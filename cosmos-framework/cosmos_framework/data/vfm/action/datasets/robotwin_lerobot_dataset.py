# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin LeRobot v3 dataset for Cosmos Action qpos14 training.

RoboTwin exports one LeRobot dataset per task/split.  The relevant schema is:

* ``data/chunk-*/file-*.parquet``: ``observation.state`` and ``action`` are both
  14D ALOHA qpos vectors: left 6 joints + left gripper + right 6 joints + right gripper.
* ``meta/episodes/chunk-*/file-*.parquet``: episode caption is ``tasks[0]``;
  frame slicing is ``dataset_from_index <= index < dataset_to_index``.
* ``videos/observation.images.*``: each mp4 stores many episodes concatenated;
  per-episode video frames start at ``round(from_timestamp * fps)``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

Viewpoint = Literal["concat_view", "cam_high"]

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"
_CAM_HIGH = "observation.images.cam_high"
_CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
_CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"
_CAMERA_KEYS = (_CAM_HIGH, _CAM_LEFT_WRIST, _CAM_RIGHT_WRIST)
_ACTION_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper_0",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper_0",
]


def _should_log_dataset_summary() -> bool:
    return os.environ.get("RANK", "0") in ("", "0") and os.environ.get("LOCAL_RANK", "0") in ("", "0")


def _as_caption(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


class RoboTwinLeRobotDataset(Dataset):
    """RoboTwin Action dataset.

    The raw action is qpos14.  With ``use_state=True`` the returned action tensor
    has shape ``(chunk_length + 1, 14)`` where row 0 is the current observed qpos
    and the remaining rows are future action commands.  This mirrors the DROID
    joint_pos recipe and lets ``ActionTransformPipeline`` mark the first action row
    as conditioning.
    """

    def __init__(
        self,
        root: str,
        fps: float | None = None,
        chunk_length: int = 32,
        video_fps: float | None = None,
        video_num_frames: int | None = None,
        video_sample_stride: int | None = None,
        mode: str = "policy",
        viewpoint: Viewpoint = "concat_view",
        split_name: str = "aloha-agilex_clean_50",
        task_names: list[str] | None = None,
        use_state: bool = True,
        action_normalization: str | None = None,
        use_image_augmentation: bool = False,
        sample_stride: int = 1,
        domain_name: str = "robotwin",
        resolution: str | int = "480",
    ) -> None:
        super().__init__()
        if viewpoint not in ("concat_view", "cam_high"):
            raise NotImplementedError("RoboTwinLeRobotDataset supports viewpoint='concat_view' or 'cam_high'.")
        if action_normalization not in (None, "meanstd", "minmax", "quantile"):
            raise ValueError("action_normalization must be one of None, 'meanstd', 'minmax', or 'quantile'.")
        if str(resolution) not in ("480", "robotwin_384_mode"):
            raise ValueError("RoboTwinLeRobotDataset supports resolution='480' or 'robotwin_384_mode'.")

        self._root = Path(root)
        self._target_action_fps_override = float(fps) if fps is not None else None
        self._target_video_fps_override = float(video_fps) if video_fps is not None else None
        self._chunk_length = int(chunk_length)
        self._video_num_frames_override = int(video_num_frames) if video_num_frames is not None else None
        self._video_sample_stride = int(video_sample_stride) if video_sample_stride is not None else None
        self._mode = mode
        self._viewpoint = viewpoint
        self._split_name = split_name
        self._use_state = bool(use_state)
        self._action_normalization = action_normalization
        self._use_image_augmentation = bool(use_image_augmentation)
        self._sample_stride = int(sample_stride)
        self._domain_name = domain_name
        self._domain_id = get_domain_id(self._domain_name)
        self._resolution = str(resolution)
        self._use_robotwin_384_mode = self._resolution == "robotwin_384_mode"
        self._norm_stats: dict[str, torch.Tensor] | None = None
        if self._chunk_length < 1:
            raise ValueError(f"chunk_length must be >= 1, got {self._chunk_length}")
        if self._video_num_frames_override is not None and self._video_num_frames_override < 1:
            raise ValueError(f"video_num_frames must be >= 1, got {self._video_num_frames_override}")
        if self._video_sample_stride is not None and self._video_sample_stride < 1:
            raise ValueError(f"video_sample_stride must be >= 1, got {self._video_sample_stride}")
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")

        self._dataset_roots = self._resolve_dataset_roots(self._root, split_name, task_names)
        if not self._dataset_roots:
            raise FileNotFoundError(f"No RoboTwin LeRobot datasets found under {root!r}.")

        self._dataset_infos: list[dict[str, Any]] = []
        row_dataset_parts: list[np.ndarray] = []
        row_episode_parts: list[np.ndarray] = []
        row_task_parts: list[np.ndarray] = []
        row_index_parts: list[np.ndarray] = []
        row_timestamp_parts: list[np.ndarray] = []
        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []

        for dataset_id, dataset_root in enumerate(
            tqdm(
                self._dataset_roots,
                desc="[robotwin_lerobot_dataset] loading parquet",
                disable=not _should_log_dataset_summary(),
                mininterval=5.0,
            )
        ):
            info = json.loads((dataset_root / "meta" / "info.json").read_text())
            source_fps = float(info["fps"])
            target_action_fps = float(
                self._target_action_fps_override if self._target_action_fps_override is not None else source_fps
            )
            if target_action_fps <= 0:
                raise ValueError(f"fps must be positive, got {target_action_fps:g}")
            action_frame_stride_float = source_fps / target_action_fps
            action_frame_stride = int(round(action_frame_stride_float))
            if action_frame_stride < 1 or abs(action_frame_stride_float - action_frame_stride) > 1e-6:
                raise ValueError(
                    f"RoboTwin fps={target_action_fps:g} must divide source fps={source_fps:g} for {dataset_root}; "
                    f"got source/target={action_frame_stride_float:g}."
                )
            if self._video_num_frames_override is not None:
                video_num_frames = self._video_num_frames_override
            elif self._video_sample_stride is not None:
                video_num_frames = int(np.ceil((self._chunk_length + 1) / self._video_sample_stride))
            else:
                video_num_frames = self._chunk_length + 1
            assert video_num_frames is not None

            if self._video_sample_stride is not None:
                video_frame_offsets = [
                    i * action_frame_stride * self._video_sample_stride for i in range(video_num_frames)
                ]
                target_video_fps = target_action_fps / self._video_sample_stride
            elif self._target_video_fps_override is None and self._video_num_frames_override is None:
                video_frame_offsets = [i * action_frame_stride for i in range(video_num_frames)]
                target_video_fps = target_action_fps
            else:
                target_video_fps = float(
                    self._target_video_fps_override
                    if self._target_video_fps_override is not None
                    else target_action_fps
                )
                if target_video_fps <= 0:
                    raise ValueError(f"video_fps must be positive, got {target_video_fps:g}")
                if target_video_fps > source_fps:
                    raise ValueError(
                        f"RoboTwin video_fps={target_video_fps:g} must be <= source fps={source_fps:g} "
                        f"for {dataset_root}."
                    )
                video_frame_stride_float = source_fps / target_video_fps
                video_frame_offsets = [
                    int(np.floor(i * video_frame_stride_float + 0.5)) for i in range(video_num_frames)
                ]
                if len(set(video_frame_offsets)) != len(video_frame_offsets):
                    raise ValueError(
                        f"RoboTwin video_fps={target_video_fps:g} creates duplicate source frames "
                        f"for source fps={source_fps:g}; offsets={video_frame_offsets}."
                    )

            episodes = {
                int(row["episode_index"]): row
                for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
                for row in pq.read_table(path).to_pylist()
            }
            captions = {episode_index: _as_caption(row.get("tasks", "")) for episode_index, row in episodes.items()}

            columns = ["index", "episode_index", "task_index", "timestamp", _STATE_FEATURE, _ACTION_FEATURE]
            data_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
            if not data_paths:
                raise FileNotFoundError(dataset_root / "data")
            index_parts, episode_parts, task_parts, ts_parts, local_state_parts, local_action_parts = [], [], [], [], [], []
            for path in data_paths:
                table = pq.read_table(path, columns=columns)
                index_parts.append(table["index"].to_numpy())
                episode_parts.append(table["episode_index"].to_numpy())
                task_parts.append(table["task_index"].to_numpy())
                ts_parts.append(table["timestamp"].to_numpy())
                local_state_parts.append(np.asarray(table[_STATE_FEATURE].to_pylist(), dtype=np.float32))
                local_action_parts.append(np.asarray(table[_ACTION_FEATURE].to_pylist(), dtype=np.float32))

            local_index = np.concatenate(index_parts).astype(np.int64)
            order = np.argsort(local_index, kind="stable")
            local_episode = np.concatenate(episode_parts).astype(np.int64)[order]
            local_task = np.concatenate(task_parts).astype(np.int64)[order]
            local_ts = np.concatenate(ts_parts).astype(np.float64)[order]
            local_state = np.concatenate(local_state_parts, axis=0).astype(np.float32)[order]
            local_action = np.concatenate(local_action_parts, axis=0).astype(np.float32)[order]

            if local_state.shape[1] != 14 or local_action.shape[1] != 14:
                raise ValueError(
                    f"Expected qpos14 state/action in {dataset_root}, got state={local_state.shape}, action={local_action.shape}"
                )

            self._dataset_infos.append(
                {
                    "root": dataset_root,
                    "info": info,
                    "source_fps": source_fps,
                    # Backward-compatible names refer to the action timeline.
                    "target_fps": target_action_fps,
                    "frame_stride": action_frame_stride,
                    "target_action_fps": target_action_fps,
                    "action_frame_stride": action_frame_stride,
                    "target_video_fps": target_video_fps,
                    "video_num_frames": video_num_frames,
                    "video_sample_stride": self._video_sample_stride,
                    "video_frame_offsets": video_frame_offsets,
                    "episodes": episodes,
                    "captions": captions,
                    "task_name": dataset_root.parent.name,
                    "split_name": dataset_root.name,
                }
            )
            n = len(local_index)
            row_dataset_parts.append(np.full(n, dataset_id, dtype=np.int64))
            row_episode_parts.append(local_episode)
            row_task_parts.append(local_task)
            row_index_parts.append(local_index[order])
            row_timestamp_parts.append(local_ts)
            state_parts.append(local_state)
            action_parts.append(local_action)

        self._row_dataset = np.concatenate(row_dataset_parts).astype(np.int64)
        self._row_episode = np.concatenate(row_episode_parts).astype(np.int64)
        self._row_task = np.concatenate(row_task_parts).astype(np.int64)
        self._row_index = np.concatenate(row_index_parts).astype(np.int64)
        self._row_timestamp = np.concatenate(row_timestamp_parts).astype(np.float64)
        self._state = np.concatenate(state_parts, axis=0).astype(np.float32)
        self._action = np.concatenate(action_parts, axis=0).astype(np.float32)

        self._window_starts: list[int] = []
        self._shuffle_blocks: list[tuple[int, int]] = []
        self._build_window_index()

        if _should_log_dataset_summary():
            print(
                "[robotwin_lerobot_dataset] compact index summary: "
                f"root={self._root}, datasets={len(self._dataset_infos)}, frames={len(self._row_dataset)}, "
                f"windows={len(self._window_starts)}, chunk_length={self._chunk_length}, "
                f"source_fps={self._dataset_infos[0]['source_fps']:g}, action_fps={self.fps:g}, "
                f"action_frame_stride={self._dataset_infos[0]['action_frame_stride']}, "
                f"video_fps={self.video_fps:g}, video_num_frames={self.video_num_frames}, "
                f"video_last_offset={self._dataset_infos[0]['video_frame_offsets'][-1]}, "
                f"use_state={self._use_state}, viewpoint={self._viewpoint}, "
                f"resolution={self._resolution}, "
                f"action_normalization={self._action_normalization}, split_name={self._split_name}"
            )

    @staticmethod
    def _resolve_dataset_roots(root: Path, split_name: str, task_names: list[str] | None) -> list[Path]:
        if (root / "meta" / "info.json").exists():
            return [root]

        names = task_names
        if names is None:
            names = sorted(p.name for p in root.iterdir() if p.is_dir())

        dataset_roots = []
        for task_name in names:
            candidate = root / task_name / split_name
            if (candidate / "meta" / "info.json").exists():
                dataset_roots.append(candidate)
        return dataset_roots

    @property
    def fps(self) -> float:
        return float(self._dataset_infos[0]["target_action_fps"])

    @property
    def video_fps(self) -> float:
        return float(self._dataset_infos[0]["target_video_fps"])

    @property
    def video_num_frames(self) -> int:
        return int(self._dataset_infos[0]["video_num_frames"])

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def viewpoint(self) -> str:
        return self._viewpoint

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_normalization(self) -> str | None:
        return self._action_normalization

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def action_names(self) -> list[str]:
        return list(_ACTION_NAMES)

    def _build_window_index(self) -> None:
        if len(self._row_dataset) == 0:
            return
        boundaries = np.flatnonzero(
            (np.diff(self._row_dataset) != 0)
            | (np.diff(self._row_episode) != 0)
            | (np.diff(self._row_index) != 1)
        ) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(self._row_dataset)]))
        flat_start = 0
        for start, end in zip(starts, ends, strict=True):
            count = int(end - start)
            dataset_id = int(self._row_dataset[start])
            dataset_info = self._dataset_infos[dataset_id]
            action_frame_stride = int(dataset_info["action_frame_stride"])
            video_last_offset = max(int(offset) for offset in dataset_info["video_frame_offsets"])
            required_source_span = max(self._chunk_length * action_frame_stride, video_last_offset)
            valid = max(0, count - required_source_span)
            if valid <= 0:
                continue
            # sample_stride controls window starts in source-frame units. The
            # target FPS only changes frame spacing inside each window.
            sampled = range(int(start), int(start) + valid, self._sample_stride)
            sampled_count = len(sampled)
            self._window_starts.extend(sampled)
            self._shuffle_blocks.append((flat_start, sampled_count))
            flat_start += sampled_count

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _video_path(self, dataset_info: dict[str, Any], episode: dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(episode[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode[f"videos/{video_key}/file_index"])
        rel = dataset_info["info"]["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return dataset_info["root"] / rel

    def _load_frames(self, dataset_info: dict[str, Any], episode: dict[str, Any], video_key: str, local_offsets: list[int]) -> torch.Tensor:
        video_start = int(round(float(episode[f"videos/{video_key}/from_timestamp"]) * float(dataset_info["source_fps"])))
        path = self._video_path(dataset_info, episode, video_key)
        reader = imageio.get_reader(str(path))
        try:
            frames = [
                torch.from_numpy(np.asarray(reader.get_data(video_start + int(offset)), dtype=np.uint8))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                for offset in local_offsets
            ]
        finally:
            reader.close()
        return torch.stack(frames, dim=0)

    def _load_video(self, dataset_info: dict[str, Any], episode: dict[str, Any], local_offsets: list[int]) -> torch.Tensor:
        high = self._load_frames(dataset_info, episode, _CAM_HIGH, local_offsets)  # [T,C,H,W]
        if self._viewpoint == "cam_high":
            return high

        left = self._load_frames(dataset_info, episode, _CAM_LEFT_WRIST, local_offsets)
        right = self._load_frames(dataset_info, episode, _CAM_RIGHT_WRIST, local_offsets)

        if self._use_robotwin_384_mode:
            high = F.interpolate(high, size=(256, 320), mode="bilinear", align_corners=False)
            left = F.interpolate(left, size=(128, 160), mode="bilinear", align_corners=False)
            right = F.interpolate(right, size=(128, 160), mode="bilinear", align_corners=False)
            bottom = torch.cat([left, right], dim=-1)  # [T,C,128,320]
            return torch.cat([high, bottom], dim=-2)  # [T,C,384,320]

        _, _, high_h, high_w = high.shape
        wrist_h = max(1, high_h // 2)
        wrist_w = max(1, high_w // 2)
        left = F.interpolate(left, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]
        if bottom.shape[-1] != high_w:
            bottom = F.interpolate(bottom, size=(wrist_h, high_w), mode="bilinear", align_corners=False)

        if self._use_image_augmentation:
            # Lightweight, deterministic-shape jitter shared by all frames after view composition.
            scale = 0.95
            crop_h = int((high_h + wrist_h) * scale)
            crop_w = int(high_w * scale)
            composed = torch.cat([high, bottom], dim=-2)
            if crop_h < composed.shape[-2] and crop_w < composed.shape[-1]:
                top = random.randint(0, composed.shape[-2] - crop_h)
                left_px = random.randint(0, composed.shape[-1] - crop_w)
                composed = composed[..., top : top + crop_h, left_px : left_px + crop_w]
                composed = F.interpolate(composed, size=(high_h + wrist_h, high_w), mode="bilinear", align_corners=False)
            return composed

        return torch.cat([high, bottom], dim=-2)  # [T,C,H+H/2,W]

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            actions = self._action.astype(np.float32)
            self._norm_stats = {
                "mean": torch.from_numpy(actions.mean(axis=0)).float(),
                "std": torch.from_numpy(np.maximum(actions.std(axis=0), 1e-8)).float(),
                "min": torch.from_numpy(actions.min(axis=0)).float(),
                "max": torch.from_numpy(actions.max(axis=0)).float(),
                "q01": torch.from_numpy(np.quantile(actions, 0.01, axis=0).astype(np.float32)).float(),
                "q99": torch.from_numpy(np.quantile(actions, 0.99, axis=0).astype(np.float32)).float(),
            }
        return self._norm_stats

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        if action.shape[0] <= 1:
            return int(action.shape[0])
        diffs = torch.abs(action[1:] - action[:-1]).amax(dim=-1)
        return int((diffs < (5e-3 / self.fps)).sum().item())

    def _build_result(self, *, mode: str, video: torch.Tensor, action: torch.Tensor, ai_caption: str, **extras: Any) -> dict[str, Any]:
        idle_frames = self._compute_idle_frames(action)
        if self._action_normalization is None:
            normalized_action = action
        else:
            from cosmos_framework.data.vfm.action.action_normalization import normalize_action

            normalized_action = normalize_action(action, self._action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self.video_fps, dtype=torch.float),
            "action_fps": torch.tensor(self.fps, dtype=torch.float),
            "condition_first_action": torch.tensor(self._use_state, dtype=torch.bool),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = int(self._window_starts[int(idx)])
        dataset_id = int(self._row_dataset[start])
        episode_index = int(self._row_episode[start])
        dataset_info = self._dataset_infos[dataset_id]
        episode = dataset_info["episodes"][episode_index]
        dataset_from = int(episode["dataset_from_index"])
        action_frame_stride = int(dataset_info["action_frame_stride"])
        positions = [start + i * action_frame_stride for i in range(self._chunk_length + 1)]
        action_local_offsets = [int(self._row_index[j] - dataset_from) for j in positions]
        video_local_offsets = [
            action_local_offsets[0] + int(offset) for offset in dataset_info["video_frame_offsets"]
        ]

        video = self._load_video(dataset_info, episode, video_local_offsets)
        future_action = torch.from_numpy(self._action[positions[1:]].copy()).float()
        if self._use_state:
            initial_state = torch.from_numpy(self._state[positions[:1]].copy()).float()
            action = torch.cat([initial_state, future_action], dim=0)
        else:
            action = future_action

        return self._build_result(
            mode=self._choose_mode(),
            video=video,
            action=action,
            ai_caption=dataset_info["captions"][episode_index],
            additional_view_description=(
                "The top row shows a high third-person view of the scene. "
                "The bottom row contains the left and right wrist-mounted camera views."
                if self._viewpoint == "concat_view"
                else "The video is from a high third-person camera."
            ),
            task_name=dataset_info["task_name"],
            episode_index=torch.tensor(episode_index, dtype=torch.long),
            source_fps=torch.tensor(float(dataset_info["source_fps"]), dtype=torch.float),
            frame_stride=torch.tensor(action_frame_stride, dtype=torch.long),
            video_num_frames=torch.tensor(int(dataset_info["video_num_frames"]), dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self._window_starts)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return self._shuffle_blocks
