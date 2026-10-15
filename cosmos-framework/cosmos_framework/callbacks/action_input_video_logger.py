# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


class ActionInputVideoLogger(Callback):
    """Log one action-SFT input/GT video sample to W&B.

    Action policy training uses the first frame as visual conditioning and the
    full video clip as the supervised GT trajectory. This callback logs both
    directly from the transformed training batch before the model normalizes the
    uint8 video in-place.
    """

    def __init__(
        self,
        every_n: int = 200,
        fps: int | None = None,
        log_first: bool = True,
        enabled: bool = True,
        include_gold_subgoal: bool = False,
        include_reasoner_vlm: bool = False,
        log_reasoner_immediately: bool = False,
        max_reasoner_vlm_samples: int = 4,
    ) -> None:
        super().__init__()
        self.every_n = int(every_n)
        self.fps = fps
        self.log_first = bool(log_first)
        self.enabled = bool(enabled)
        # Keep the legacy caption unchanged unless a subgoal experiment opts in.
        self.include_gold_subgoal = bool(include_gold_subgoal)
        # The ordinary action recipes stay unchanged by default.  The opt-in
        # Reasoner/WAM recipe enables this to publish a bounded CPU/PIL sample
        # captured before the Reasoner batch is freed.
        self.include_reasoner_vlm = bool(include_reasoner_vlm)
        # Standalone Reasoner has no following action-video batch to flush the
        # preview, so it logs directly at Reasoner batch end.
        self.log_reasoner_immediately = bool(log_reasoner_immediately)
        self._max_reasoner_vlm_samples = max(1, int(max_reasoner_vlm_samples))
        self._pending_reasoner_vlm: list[dict[str, Any]] = []
        self._last_logged_iteration: int | None = None

    def _should_log(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if self._last_logged_iteration == iteration:
            return False
        if not distributed.is_rank0():
            return False
        if wandb.run is None:
            return False
        if self.log_first and iteration <= 0:
            return True
        return self.every_n > 0 and iteration % self.every_n == 0

    @staticmethod
    def _first_video(video_batch: Any) -> torch.Tensor | None:
        """Extract the first sample video as [C, T, H, W]."""
        video = video_batch
        while isinstance(video, (list, tuple)):
            if len(video) == 0:
                return None
            video = video[0]
        if not isinstance(video, torch.Tensor):
            return None
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4:
            return None
        return video

    @staticmethod
    def _first_image_size(image_size_batch: Any) -> list[int] | None:
        """Extract first sample image_size as [target_h, target_w, content_h, content_w]."""
        image_size = image_size_batch
        while isinstance(image_size, (list, tuple)):
            if len(image_size) == 0:
                return None
            image_size = image_size[0]
        if not isinstance(image_size, torch.Tensor):
            return None
        image_size = image_size.detach().cpu()
        if image_size.ndim == 2 and image_size.shape[0] > 0:
            image_size = image_size[0]
        if image_size.ndim != 1 or image_size.numel() < 4:
            return None
        return [int(round(float(v))) for v in image_size[:4].tolist()]

    @staticmethod
    def _first_text(value: Any) -> str:
        """Extract the first string from a collated scalar/list value."""
        while isinstance(value, (list, tuple)):
            if not value:
                return ""
            value = value[0]
        return value if isinstance(value, str) else ""

    @staticmethod
    def _text_list(value: Any) -> list[str]:
        """Flatten collated text metadata without touching CUDA tensors."""

        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            result: list[str] = []
            for item in value:
                result.extend(ActionInputVideoLogger._text_list(item))
            return result
        return [str(value)]

    @staticmethod
    def _image_list(value: Any) -> list[Any]:
        """Flatten only list containers; PIL images are kept as host objects."""

        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            result: list[Any] = []
            for item in value:
                result.extend(ActionInputVideoLogger._image_list(item))
            return result
        return [value]

    @classmethod
    def _is_reasoner_batch(cls, data: dict[str, Any], output: dict[str, Any]) -> bool:
        task_types = cls._text_list(data.get("task_type")) + cls._text_list(output.get("task_type"))
        return any(task_type.strip().lower() == "reasoner" for task_type in task_types)

    def _capture_reasoner_vlm(
        self,
        data: dict[str, Any],
        output: dict[str, Any],
        iteration: int,
    ) -> None:
        if not self.include_reasoner_vlm or not self._is_reasoner_batch(data, output):
            return
        # A VLM cutoff intentionally stops CE/LoRA updates.  Do not publish a
        # misleading "training" sample for those consumed-but-skipped batches;
        # the WAM video logger continues normally.
        if output.get("reasoner_phase_skipped") is True or output.get("reasoner_update_performed") is False:
            return
        # Reuse the exact cadence/rank/W&B gate used by the video logger.  The
        # metadata is captured while it is still a host-side PIL/string object.
        if not self._should_log(iteration):
            return

        tasks = self._text_list(data.get("reasoner_task_text"))
        prompts = self._text_list(data.get("reasoner_user_prompt"))
        gold = self._text_list(output.get("reasoner_gold_target_text", data.get("reasoner_gold_target_text")))
        images = self._image_list(data.get("reasoner_preview_image"))
        image_counts = self._text_list(data.get("reasoner_image_count"))
        count = max(len(tasks), len(prompts), len(gold), len(images), len(image_counts))
        pending: list[dict[str, Any]] = []
        for index in range(min(count, self._max_reasoner_vlm_samples)):
            task = tasks[index] if index < len(tasks) else ""
            prompt = prompts[index] if index < len(prompts) else task
            target = gold[index] if index < len(gold) else ""
            image = images[index] if index < len(images) else None
            raw_image_count = image_counts[index] if index < len(image_counts) else ""
            image_count = int(raw_image_count) if raw_image_count else prompt.count("<image>")
            if prompt or target or image is not None:
                # Keep the original annotation text intact.  The sample count
                # and preview dimensions are bounded separately, so there is
                # no reason to truncate the supervision string.
                pending.append(
                    {
                        "task": task,
                        "user_prompt": prompt,
                        "gold_subgoal": target,
                        "image": image,
                        "image_count": image_count,
                    }
                )
        self._pending_reasoner_vlm = pending

    def _reasoner_vlm_log_payload(self) -> dict[str, Any]:
        if not self._pending_reasoner_vlm:
            return {}
        rows: list[list[Any]] = []
        for sample in self._pending_reasoner_vlm:
            task = sample["task"]
            prompt = sample["user_prompt"]
            target = sample["gold_subgoal"]
            image = sample["image"]
            logged_image = None
            if image is not None:
                logged_image = wandb.Image(image, caption=f"user:\n{prompt}\n\nassistant:\n{target}")
            rows.append([logged_image, sample["image_count"], prompt, target])

        payload: dict[str, Any] = {
            "reasoner_vlm/samples": wandb.Table(
                columns=["input_images", "image_count", "user_prompt", "gold_assistant"],
                data=rows,
            )
        }
        first = self._pending_reasoner_vlm[0]
        if first["task"]:
            payload["reasoner_vlm/task"] = first["task"]
        if first["user_prompt"]:
            payload["reasoner_vlm/user_prompt"] = first["user_prompt"]
        if first["gold_subgoal"]:
            payload["reasoner_vlm/gold_subgoal"] = first["gold_subgoal"]
        if rows[0][0] is not None:
            payload["reasoner_vlm/preview"] = rows[0][0]
        return payload

    @staticmethod
    def _crop_to_content(video: torch.Tensor, image_size: list[int] | None) -> torch.Tensor:
        """Pixel-space preview of the latent-crop content region."""
        if image_size is None:
            return video
        content_h = max(1, min(int(image_size[2]), int(video.shape[-2])))
        content_w = max(1, min(int(image_size[3]), int(video.shape[-1])))
        return video[..., :content_h, :content_w].contiguous()

    @staticmethod
    def _to_uint8_video(video: torch.Tensor) -> np.ndarray:
        """Convert [C, T, H, W] tensor to W&B's ndarray video format [T, C, H, W]."""
        video_cpu = video.detach().cpu()
        if torch.is_floating_point(video_cpu):
            min_value = float(video_cpu.min().item())
            max_value = float(video_cpu.max().item())
            if min_value >= -1.01 and max_value <= 1.01:
                video_cpu = (video_cpu + 1.0) * 127.5
            elif max_value <= 1.01:
                video_cpu = video_cpu * 255.0
            video_cpu = video_cpu.clamp(0.0, 255.0).to(torch.uint8)
        else:
            video_cpu = video_cpu.clamp(0, 255).to(torch.uint8)
        return video_cpu.permute(1, 0, 2, 3).contiguous().numpy()

    @staticmethod
    def _caption(
        data_batch: dict[str, Any],
        source_video: torch.Tensor,
        logged_video: torch.Tensor,
        fps: int,
        image_size: list[int] | None,
        include_gold_subgoal: bool = False,
    ) -> str:
        caption = data_batch.get("ai_caption", "")
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        if include_gold_subgoal and isinstance(caption, str) and caption.strip():
            gold_subgoal = ActionInputVideoLogger._first_text(data_batch.get("subtask"))
            if gold_subgoal.strip():
                caption = f"{caption}\ngold_subgoal: {gold_subgoal}"
        shape_text = (
            f"source_video_shape=[C,T,H,W]={list(source_video.shape)}, "
            f"logged_content_shape=[C,T,H,W]={list(logged_video.shape)}, fps={fps}"
        )
        if image_size is not None:
            shape_text += f", image_size={image_size}"
        return f"{shape_text}\n{caption}"

    def _resolve_fps(self, data_batch: dict[str, Any]) -> int:
        if self.fps is not None:
            return int(self.fps)
        value = data_batch.get("action_fps", data_batch.get("conditioning_fps", None))
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, torch.Tensor):
            return max(1, int(value.detach().cpu().flatten()[0].item()))
        if value is not None:
            return max(1, int(value))
        return 10

    @torch.no_grad()
    def on_training_step_batch_end(
        self,
        model,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, loss
        self._capture_reasoner_vlm(data_batch, output_batch, iteration)
        if not self.log_reasoner_immediately or not self._pending_reasoner_vlm:
            return
        try:
            wandb.log(self._reasoner_vlm_log_payload(), step=iteration)
            self._last_logged_iteration = iteration
        except Exception as exc:  # pragma: no cover - defensive logging path
            log.warning(f"Reasoner sample logger failed at iteration {iteration}: {exc}", rank0_only=True)
        finally:
            self._pending_reasoner_vlm.clear()

    @torch.no_grad()
    def on_training_step_batch_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model
        if not self._should_log(iteration):
            return

        video = self._first_video(data.get("video"))
        if video is None:
            return

        try:
            fps = self._resolve_fps(data)
            image_size = self._first_image_size(data.get("image_size"))
            content_video = self._crop_to_content(video, image_size)
            gt_video = self._to_uint8_video(content_video)
            input_frame = gt_video[0].transpose(1, 2, 0)
            caption = self._caption(
                data,
                video,
                content_video,
                fps,
                image_size,
                include_gold_subgoal=self.include_gold_subgoal,
            )
            payload: dict[str, Any] = {
                "train_input/conditioning_frame": wandb.Image(
                    input_frame,
                    caption=caption,
                ),
                "train_input/gt_video": wandb.Video(
                    gt_video,
                    caption=caption,
                    fps=fps,
                    format="mp4",
                ),
            }
            payload.update(self._reasoner_vlm_log_payload())
            wandb.log(payload, step=iteration)
            self._last_logged_iteration = iteration
            self._pending_reasoner_vlm.clear()
        except Exception as exc:  # pragma: no cover - defensive logging path
            self._pending_reasoner_vlm.clear()
            log.warning(f"ActionInputVideoLogger failed at iteration {iteration}: {exc}", rank0_only=True)
