# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)

_REASONER_HISTORY_KEY = "reasoner/history_images"
_REASONER_SEPARATE_VIEWS_KEY = "reasoner/separate_views"
_REASONER_PREVIEW_ACTION_KEY = "reasoner/preview_action"


def _remove_isolated_gripper_commands(
    chunk: np.ndarray, *, previous_command: int | None
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Replace one-step binary gripper pulses with matching neighbours.

    The final action is kept unchanged because its right neighbour is in the
    next server chunk.  Decisions use the original sequence (no cascading).
    """
    filtered = np.asarray(chunk).copy()
    if filtered.ndim != 2 or filtered.shape[0] < 2 or filtered.shape[1] == 0:
        return filtered, ()
    original = (filtered[:, -1] > 0.5).astype(np.int8)
    replacement = original.copy()
    changed: list[int] = []
    for index in range(filtered.shape[0] - 1):
        left = previous_command if index == 0 else int(original[index - 1])
        right = int(original[index + 1])
        if left is not None and left == right and int(original[index]) != left:
            replacement[index] = left
            changed.append(index)
    filtered[:, -1] = replacement.astype(filtered.dtype)
    return filtered, tuple(changed)


def _relative_gripper_commands(
    chunk: np.ndarray,
    *,
    current_position: float | None,
    previous_command: int | None,
    threshold: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Convert absolute gripper predictions using the current observed state.

    The server returns RoboLab-wire values (``0=open, 1=close``).  The DROID
    model itself uses the opposite convention and the server flips it before
    returning the chunk.  Comparing a returned prediction with the current
    wire-domain state therefore preserves the model's *direction*: a value
    above ``current + threshold`` means close, while a value below
    ``current - threshold`` means open.  Values inside the dead-band hold the
    previous command, which is a small hysteresis and prevents noisy state
    observations from toggling the binary simulator action.

    The final action remains binary because RoboLab's action term is itself a
    binary open/close controller; passing the continuous value through would
    still be thresholded by the simulator.
    """
    filtered = np.asarray(chunk).copy()
    if filtered.ndim != 2 or filtered.shape[0] == 0 or filtered.shape[1] == 0:
        return filtered, ()
    if current_position is None or not np.isfinite(current_position):
        return filtered, ()
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"gripper relative threshold must be finite and > 0, got {threshold!r}")

    current = float(np.clip(current_position, 0.0, 1.0))
    predictions = np.asarray(filtered[:, -1], dtype=np.float64)
    lower = current - float(threshold)
    upper = current + float(threshold)
    replacement = np.empty(predictions.shape[0], dtype=np.int8)
    changed: list[int] = []
    hold = previous_command
    if hold is None:
        # If the first prediction is in the dead-band, retaining the current
        # physical state is safer than introducing a new command.
        hold = int(current > 0.5)
    for index, prediction in enumerate(predictions):
        if np.isfinite(prediction) and prediction >= upper:
            command = 1  # RoboLab wire domain: close.
        elif np.isfinite(prediction) and prediction <= lower:
            command = 0  # RoboLab wire domain: open.
        else:
            command = int(hold)
        replacement[index] = command
        if command != int(prediction > 0.5):
            changed.append(index)
        hold = command
    filtered[:, -1] = replacement.astype(filtered.dtype)
    return filtered, tuple(changed)


def _previous_frame_relative_gripper_commands(
    chunk: np.ndarray,
    *,
    current_position: float | None,
    previous_command: int | None,
    threshold: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Convert gripper predictions relative to the preceding prediction.

    This is an opt-in variant of :func:`_relative_gripper_commands`.  The
    first prediction is compared with the current observed gripper position;
    prediction ``t > 0`` is compared with the original continuous prediction
    at ``t - 1``.  All legacy relative-mode behavior remains in the original
    helper above.
    """
    filtered = np.asarray(chunk).copy()
    if filtered.ndim != 2 or filtered.shape[0] == 0 or filtered.shape[1] == 0:
        return filtered, ()
    if current_position is None or not np.isfinite(current_position):
        return filtered, ()
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"gripper relative threshold must be finite and > 0, got {threshold!r}")

    current = float(np.clip(current_position, 0.0, 1.0))
    predictions = np.asarray(filtered[:, -1], dtype=np.float64)
    replacement = np.empty(predictions.shape[0], dtype=np.int8)
    changed: list[int] = []
    hold = previous_command
    if hold is None:
        hold = int(current > 0.5)

    reference = current
    for index, prediction in enumerate(predictions):
        if np.isfinite(prediction) and prediction >= reference + float(threshold):
            command = 1  # RoboLab wire domain: close.
        elif np.isfinite(prediction) and prediction <= reference - float(threshold):
            command = 0  # RoboLab wire domain: open.
        else:
            command = int(hold)
        replacement[index] = command
        if command != int(prediction > 0.5):
            changed.append(index)
        hold = command
        if np.isfinite(prediction):
            reference = float(prediction)

    filtered[:, -1] = replacement.astype(filtered.dtype)
    return filtered, tuple(changed)


class Cosmos3Client(InferenceClient):
    """ """

    IMAGE_W = 640
    IMAGE_H = 360
    OPEN_LOOP_HORIZON = 32

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        filter_isolated_gripper_commands: bool = False,
        relative_gripper_commands: bool = False,
        previous_frame_relative_gripper_commands: bool = False,
        gripper_relative_threshold: float = 0.10,
        reasoner_history_images: bool = False,
        reasoner_separate_views: bool = False,
        reasoner_three_group_history: bool = False,
        reasoner_include_preview_action: bool = False,
        connect: bool = True,
    ):
        """ """
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = remote_port

        self._image_w = self.IMAGE_W
        self._image_h = self.IMAGE_H

        self.open_loop_horizon = self.OPEN_LOOP_HORIZON
        # Opt-in only: ordinary Cosmos3 evaluations retain the legacy action
        # bytes and response shape when this is false.
        self.filter_isolated_gripper_commands = bool(filter_isolated_gripper_commands)
        self.relative_gripper_commands = bool(relative_gripper_commands)
        self.previous_frame_relative_gripper_commands = bool(previous_frame_relative_gripper_commands)
        self.gripper_relative_threshold = float(gripper_relative_threshold)
        if self.relative_gripper_commands and self.previous_frame_relative_gripper_commands:
            raise ValueError(
                "relative_gripper_commands and previous_frame_relative_gripper_commands "
                "are mutually exclusive"
            )
        self.reasoner_include_preview_action = bool(reasoner_include_preview_action)
        self.reasoner_three_group_history = bool(reasoner_three_group_history)
        # Training adds the previous action only to three-image examples. Make
        # history-dependent modes imply causal image history so the wire format
        # cannot drift into an unseen current-only combination.
        self.reasoner_history_images = bool(
            reasoner_history_images or self.reasoner_include_preview_action or self.reasoner_three_group_history
        )
        self.reasoner_separate_views = bool(reasoner_separate_views)
        if not np.isfinite(self.gripper_relative_threshold) or self.gripper_relative_threshold <= 0.0:
            raise ValueError(
                "gripper_relative_threshold must be finite and > 0, "
                f"got {gripper_relative_threshold!r}"
            )
        self._last_executed_gripper_commands: dict[int, int] = {}
        self._isolated_gripper_filter_indices: dict[int, tuple[int, ...]] = {}
        # ``InferenceClient`` calls _postprocess_chunk without env_id, so
        # _extract_observation records which sequential infer call is active.
        self._active_env_id: int | None = None
        self._current_observed_gripper_positions: dict[int, float] = {}
        self._reasoner_frame_history: dict[int, deque[np.ndarray]] = {}
        self._reasoner_separate_frame_history: dict[int, deque[list[np.ndarray]]] = {}
        self._previous_reasoner_actions: dict[int, str] = {}

        display = f"{self._remote_host}:{self._remote_port}"
        self.client = None
        if connect:
            print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
            self.client = self._connect()
            print(f"[{self.__class__.__name__}] Connected to {display}.")
        else:
            print(f"[{self.__class__.__name__}] Replay-only mode; skipping WAM connection ({display}).")

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        """ """
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """ """
        if self.client is None:
            raise RuntimeError(
                "Cosmos3Client has no policy connection; a replay-only episode "
                "attempted to request a live WAM chunk"
            )
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    e,
                    attempt + 1,
                    max_retries,
                )
                self.client = self._connect()
                # Flush chunk cache so all envs re-request on next step.
                self._chunks.clear()
                self._counters.clear()

    def reload_reasoner_lora(self, lora_path: str) -> dict:
        """Replace the local server's live Reasoner adapter between PPO steps.

        This is intentionally a small localhost control request, not a second
        inference model.  The server keeps its base and WAM weights resident
        and atomically copies only the updated Reasoner LoRA tensors.
        """

        response = self._infer_with_retry(
            {
                "_robolab_control": "reload_reasoner_lora",
                "reasoner_lora_path": str(lora_path),
            }
        )
        if response.get("control") != "reload_reasoner_lora":
            raise RuntimeError(f"Reasoner LoRA reload failed: {response!r}")
        return response

    def ppo_update_reasoner_lora(
        self, *, rollout_id: str, reward: float, advantage: float, output_lora: str
    ) -> dict:
        response = self._infer_with_retry(
            {
                "_robolab_control": "ppo_update_reasoner_lora",
                "rollout_id": rollout_id,
                "reward": float(reward),
                "advantage": float(advantage),
                "output_lora": str(output_lora),
            }
        )
        if response.get("status") != "passed":
            raise RuntimeError(f"server-side PPO update failed: {response!r}")
        return response

    def spin_update_reasoner_lora(
        self,
        *,
        rollout_id: str,
        chosen_response: str,
        rejected_response: str,
        output_lora: str,
        beta: float = 0.1,
        synchronize: bool = False,
        apply_update: bool = True,
        learning_rate: float = 1e-6,
        step: int | None = None,
        defer_optimizer_step: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> dict:
        """Apply or accumulate one online-DPO/SPIN pair on the resident Reasoner."""

        response = self._infer_with_retry(
            {
                "_robolab_control": "spin_update_reasoner_lora",
                "rollout_id": str(rollout_id),
                "chosen_response": str(chosen_response),
                "rejected_response": str(rejected_response),
                "output_lora": str(output_lora),
                "beta": float(beta),
                "synchronize": bool(synchronize),
                "apply_update": bool(apply_update),
                "learning_rate": float(learning_rate),
                "step": step,
                "defer_optimizer_step": bool(defer_optimizer_step),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
            }
        )
        if response.get("status") != "passed":
            raise RuntimeError(f"server-side SPIN update failed: {response!r}")
        return response

    def orpo_update_reasoner_lora(
        self,
        *,
        rollout_id: str,
        chosen_response: str,
        rejected_response: str,
        output_lora: str,
        beta: float = 0.1,
        synchronize: bool = False,
        apply_update: bool = True,
        learning_rate: float = 1e-6,
        step: int | None = None,
        orpo_front_tokens: int = 0,
        orpo_front_weight: float = 1.0,
    ) -> dict:
        """Apply one online-ORPO pair on the resident Reasoner (reference-free)."""

        response = self._infer_with_retry(
            {
                "_robolab_control": "orpo_update_reasoner_lora",
                "rollout_id": str(rollout_id),
                "chosen_response": str(chosen_response),
                "rejected_response": str(rejected_response),
                "output_lora": str(output_lora),
                "beta": float(beta),
                "synchronize": bool(synchronize),
                "apply_update": bool(apply_update),
                "learning_rate": float(learning_rate),
                "step": step,
                "orpo_front_tokens": int(orpo_front_tokens),
                "orpo_front_weight": float(orpo_front_weight),
            }
        )
        if response.get("status") != "passed":
            raise RuntimeError(f"server-side ORPO update failed: {response!r}")
        return response

    def finalize_spin_reasoner_lora(
        self,
        *,
        output_lora: str,
        synchronize: bool = False,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-6,
        step: int | None = None,
    ) -> dict:
        """Apply one optimizer step after a complete accumulated DPO batch."""

        response = self._infer_with_retry(
            {
                "_robolab_control": "finalize_spin_reasoner_lora",
                "output_lora": str(output_lora),
                "synchronize": bool(synchronize),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "learning_rate": float(learning_rate),
                "step": step,
            }
        )
        if response.get("status") != "passed":
            raise RuntimeError(f"server-side accumulated SPIN finalize failed: {response!r}")
        return response

    def discard_reasoner_rollout(self, *, rollout_id: str) -> dict:
        """Discard an unselected online preference rollout."""

        response = self._infer_with_retry(
            {
                "_robolab_control": "discard_reasoner_rollout",
                "rollout_id": str(rollout_id),
            }
        )
        if response.get("status") != "discarded":
            raise RuntimeError(f"server-side rollout discard failed: {response!r}")
        return response

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        """ """
        self._active_env_id = env_id
        # Keep the model's historical padded 360x640 images for the concat/WAM
        # path.  Separate-view Reasoner mode additionally preserves the raw
        # camera pixels so the server can apply the requested long-edge resize
        # (otherwise exterior views would already have been downscaled here).
        left_raw = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].cpu().numpy()
        right_raw = raw_obs["image_obs"]["over_shoulder_right_camera"][env_id].cpu().numpy()
        wrist_raw = raw_obs["image_obs"]["wrist_cam"][env_id].cpu().numpy()
        left_image = image_tools.resize_with_pad(left_raw, self._image_h, self._image_w)
        right_image = image_tools.resize_with_pad(right_raw, self._image_h, self._image_w)
        wrist_image = image_tools.resize_with_pad(wrist_raw, self._image_h, self._image_w)

        joint_position = raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy()
        gripper_position = raw_obs["proprio_obs"]["gripper_pos"][env_id].cpu().numpy()
        gripper_values = np.asarray(gripper_position, dtype=np.float64).reshape(-1)
        if gripper_values.size:
            self._current_observed_gripper_positions[env_id] = float(gripper_values[-1])

        extracted = {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
        if self.reasoner_separate_views:
            extracted["_separate_views"] = [
                np.ascontiguousarray(left_raw).copy(),
                np.ascontiguousarray(right_raw).copy(),
                np.ascontiguousarray(wrist_raw).copy(),
            ]
        if self.reasoner_history_images:
            # Cache every control-step frame.  Caching only when a new action
            # chunk is requested would make -16/-8 mean chunks instead of the
            # trained raw 15 Hz frame offsets.
            image = self._compose_observation_image(extracted)
            extracted["_observation_image"] = image
            self._reasoner_frame_history.setdefault(env_id, deque(maxlen=33)).append(image.copy())
            if self.reasoner_separate_views:
                self._reasoner_separate_frame_history.setdefault(env_id, deque(maxlen=33)).append(
                    [image.copy() for image in extracted["_separate_views"]]
                )
        return extracted

    def _compose_observation_image(self, extracted_obs: dict) -> np.ndarray:
        """Compose wrist/left/right views into the server input frame."""
        # Compose wrist, left, and right views into a single frame.
        wrist = extracted_obs["wrist_image"]
        size = (self._image_h // 2, self._image_w // 2)
        left = torch.from_numpy(extracted_obs["left_image"]).permute(2, 0, 1).unsqueeze(0).float()
        left = F.interpolate(left, size=size, mode="bilinear")
        left = left.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)
        right = torch.from_numpy(extracted_obs["right_image"]).permute(2, 0, 1).unsqueeze(0).float()
        right = F.interpolate(right, size=size, mode="bilinear")
        right = right.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)
        return np.ascontiguousarray(np.concatenate((wrist, np.concatenate((left, right), axis=1))))

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        """ """
        image = extracted_obs.get("_observation_image")
        if image is None:
            image = self._compose_observation_image(extracted_obs)

        request = {
            "observation/image": image,
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }
        if self.reasoner_separate_views:
            separate_views = extracted_obs.get("_separate_views")
            if separate_views is not None:
                request[_REASONER_SEPARATE_VIEWS_KEY] = np.stack(separate_views, axis=0)
        if self.reasoner_history_images:
            env_id = self._active_env_id
            history = None if env_id is None else self._reasoner_frame_history.get(env_id)
            # The server's history field contains only the two past frames;
            # it appends the current wire image itself.  Until both offsets are
            # available, omit the field so the first request is one image.
            if history is not None and len(history) >= 17:
                if self.reasoner_separate_views:
                    separate_history = self._reasoner_separate_frame_history.get(env_id)
                    if separate_history is not None and len(separate_history) >= 17:
                        history_indices = [-17, -9] if self.reasoner_three_group_history else [-17]
                        request[_REASONER_HISTORY_KEY] = np.stack(
                            [np.stack(separate_history[index], axis=0) for index in history_indices], axis=0
                        )
                        request[_REASONER_SEPARATE_VIEWS_KEY] = np.stack(separate_history[-1], axis=0)
                else:
                    request[_REASONER_HISTORY_KEY] = np.stack(
                        [history[-17], history[-9]], axis=0
                    )
                if self.reasoner_include_preview_action:
                    request[_REASONER_PREVIEW_ACTION_KEY] = self._previous_reasoner_actions.get(env_id, "")
        return request

    def _query_server(self, request: dict) -> dict:
        """ """
        response = self._infer_with_retry(request)
        if (
            self.reasoner_include_preview_action
            and not bool(request.get("skip_reasoner", False))
            and self._active_env_id is not None
            and isinstance(response, dict)
            and "subtask" in response
        ):
            subtask = response["subtask"]
            if not isinstance(subtask, str):
                raise TypeError("Reasoner response 'subtask' must be a string in prev mode")
            self._previous_reasoner_actions[self._active_env_id] = subtask.strip()
        return response

    def _unpack_response(self, response: dict) -> np.ndarray:
        """ """
        return np.asarray(response["action"])

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """ """
        chunk = chunk.copy()
        if self.previous_frame_relative_gripper_commands:
            env_id = self._active_env_id
            chunk, changed = _previous_frame_relative_gripper_commands(
                chunk,
                current_position=(
                    None if env_id is None else self._current_observed_gripper_positions.get(env_id)
                ),
                previous_command=(
                    None if env_id is None else self._last_executed_gripper_commands.get(env_id)
                ),
                threshold=self.gripper_relative_threshold,
            )
            if changed:
                logger.info(
                    "[%s] previous-frame relative gripper changed indices %s "
                    "(initial_state=%.3f, threshold=%.3f)",
                    self.__class__.__name__,
                    changed,
                    (
                        self._current_observed_gripper_positions.get(env_id, float("nan"))
                        if env_id is not None
                        else float("nan")
                    ),
                    self.gripper_relative_threshold,
                )
        elif self.relative_gripper_commands:
            env_id = self._active_env_id
            chunk, changed = _relative_gripper_commands(
                chunk,
                current_position=(
                    None if env_id is None else self._current_observed_gripper_positions.get(env_id)
                ),
                previous_command=(
                    None if env_id is None else self._last_executed_gripper_commands.get(env_id)
                ),
                threshold=self.gripper_relative_threshold,
            )
            if changed:
                logger.info(
                    "[%s] relative gripper hysteresis changed indices %s (state=%.3f, threshold=%.3f)",
                    self.__class__.__name__,
                    changed,
                    (
                        self._current_observed_gripper_positions.get(env_id, float("nan"))
                        if env_id is not None
                        else float("nan")
                    ),
                    self.gripper_relative_threshold,
                )
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        changed: tuple[int, ...] = ()
        if self.filter_isolated_gripper_commands:
            chunk, changed = _remove_isolated_gripper_commands(
                chunk,
                previous_command=self._last_executed_gripper_commands.get(env_id),
            )
            if changed:
                logger.info("[%s] env%d filtered gripper indices %s", self.__class__.__name__, env_id, changed)
        self._isolated_gripper_filter_indices[env_id] = changed
        super()._set_chunk(env_id, chunk)

    def _next_action(self, env_id: int) -> np.ndarray:
        action = super()._next_action(env_id)
        values = np.asarray(action).reshape(-1)
        if values.size:
            self._last_executed_gripper_commands[env_id] = int(values[-1] > 0.5)
        return action

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._last_executed_gripper_commands.clear()
            self._isolated_gripper_filter_indices.clear()
            self._active_env_id = None
            self._current_observed_gripper_positions.clear()
            self._reasoner_frame_history.clear()
            self._reasoner_separate_frame_history.clear()
            self._previous_reasoner_actions.clear()
        else:
            self._last_executed_gripper_commands.pop(env_id, None)
            self._isolated_gripper_filter_indices.pop(env_id, None)
            self._current_observed_gripper_positions.pop(env_id, None)
            self._reasoner_frame_history.pop(env_id, None)
            self._reasoner_separate_frame_history.pop(env_id, None)
            self._previous_reasoner_actions.pop(env_id, None)
            if self._active_env_id == env_id:
                self._active_env_id = None

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        """ """
        left = extracted_obs["left_image"]
        wrist = extracted_obs["wrist_image"]
        right = extracted_obs["right_image"]
        return np.concatenate((left, wrist, right), axis=1)
