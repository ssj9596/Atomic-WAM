# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal DROID LeRobot dataset for Cosmos Action v1.2 defaults."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlparse

import boto3
import fsspec
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from botocore.config import Config as BotoConfig
from lerobot.datasets import video_utils as lerobot_video_utils
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Joint, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.camera_condition import (
    camera_xyzrpy_to_matrix,
    compose_exterior_plucker_condition,
    scaled_camera_matrix,
)
from cosmos_framework.data.vfm.action.datasets.subtask_annotations import (
    EpisodeSubtaskFilter,
    load_subtask_annotations,
    load_subtask_episode_filters,
    lookup_subtask,
)
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    compute_idle_frames,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]
VideoDecodeBackend = Literal["imageio", "lerobot", "torchcodec", "pyav", "video_reader"]

_IMAGE_FEATURES = {
    "wrist": "observation.images.wrist_left",
    "left": "observation.images.exterior_1_left",
    "right": "observation.images.exterior_2_left",
}
_COSMOS3_DROID_IMAGE_FEATURES = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
_EXTERIOR_VIDEO_KEYS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
)
_STATE_FEATURE = "observation.state.cartesian_position"
# joint_pos (8D = 7 arm joints + gripper) features, matching the internal
# DROIDLeRobotDataset(action_space="joint_pos", use_state=...). These are
# absolute joint commands/states (no normalization is applied for joint_pos,
# matching the internal canonical run which leaves action_normalization=None).
_JOINT_ACTION_FEATURE = "action.joint_position"  # [7] commanded joints
_ACTION_GRIPPER_FEATURE = "action.gripper_position"  # [1] commanded gripper
_JOINT_STATE_FEATURE = "observation.state.joint_positions"  # [7] observed joints
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"  # [1] observed gripper
# Columns whose parquet dtype is a list<float> (need to_pylist -> stacked array).
_LIST_COLUMNS = {_STATE_FEATURE, _JOINT_ACTION_FEATURE, _JOINT_STATE_FEATURE}
_ACTION_SPACES = ("ee_pose", "joint_pos")

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production DROID wrapper conversion from Franka panda_link8 to OpenCV.
_DROID_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent / "stats/droid_lerobot_stats.json"
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_FILTER_KEY_COLUMNS = (
    "episode_id",
    "trajectory_id",
    "trajectory_path",
    "trajectory",
    "path",
    "source_path",
    "recording_path",
)
FilterDictKeyMode = Literal["auto", "metadata", "ordered"]
_CAM2BASE_SUPERSET_FILENAME = "cam2base_extrinsic_superset.json"
_CAM2BASE_STANDARD_FILENAME = "cam2base_extrinsics.json"
_CAM2CAM_FILENAME = "cam2cam_extrinsics.json"
_CAMERA_SERIALS_FILENAME = "camera_serials.json"
_EPISODE_ID_TO_PATH_FILENAME = "episode_id_to_path.json"
_INTRINSICS_FILENAME = "intrinsics.json"
_MAX_PLAUSIBLE_CAMERA_TRANSLATION_NORM_M = 2.0
_DROID_TORCHCODEC_DECODER_CACHE_SIZE = 24
StoragePath = str | Path


def _is_s3_path(path: StoragePath) -> bool:
    return str(path).startswith("s3://")


def _storage_join(root: StoragePath, *parts: str) -> StoragePath:
    if _is_s3_path(root):
        suffix = "/".join(str(part).strip("/") for part in parts)
        return f"{str(root).rstrip('/')}/{suffix}"
    return Path(root).joinpath(*parts)


def _storage_open(path: StoragePath, mode: str = "rb"):
    if _is_s3_path(path):
        return fsspec.open(str(path), mode=mode).open()
    return Path(path).open(mode=mode)


def _storage_read_json(path: StoragePath) -> Any:
    with _storage_open(path, "rt") as f:
        return json.load(f)


def _storage_is_file(path: StoragePath) -> bool:
    if _is_s3_path(path):
        fs, fs_path = fsspec.core.url_to_fs(str(path))
        return bool(fs.isfile(fs_path))
    return Path(path).is_file()


def _storage_glob(root: StoragePath, pattern: str) -> list[StoragePath]:
    if _is_s3_path(root):
        uri = str(_storage_join(root, pattern))
        fs, fs_pattern = fsspec.core.url_to_fs(uri)
        return [fs.unstrip_protocol(path) for path in sorted(fs.glob(fs_pattern))]
    return sorted(Path(root).glob(pattern))


def _storage_read_schema(path: StoragePath):
    if _is_s3_path(path):
        with _storage_open(path, "rb") as f:
            return pq.read_schema(f)
    return pq.read_schema(path)


class _BoundedTorchcodecDecoderCache:
    """Small LRU cache compatible with LeRobot's torchcodec helper.

    LeRobot's default cache has no capacity bound and holds an open file plus
    decoder for every distinct video path. DROID spans millions of videos, so
    using that cache directly causes linear FD/RSS growth. This replacement
    keeps recently reused videos (important for episode-local sampling) while
    closing both resources immediately on eviction.
    """

    def __init__(self, max_entries: int = _DROID_TORCHCODEC_DECODER_CACHE_SIZE) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = int(max_entries)
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _close_entry(entry: tuple[Any, Any]) -> None:
        decoder, file_handle = entry
        # Drop the decoder before the handle: VideoDecoder can hold an
        # internal reference to it. CPython releases the native decoder at
        # this point; closing the Python handle releases the FD deterministically.
        del decoder
        file_handle.close()

    def get_decoder(self, video_path: str):
        from torchcodec.decoders import VideoDecoder

        video_path = str(video_path)
        with self._lock:
            cached = self._cache.pop(video_path, None)
            if cached is not None:
                self._cache[video_path] = cached
                return cached[0]

            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)
            while len(self._cache) > self._max_entries:
                _old_path, old_entry = self._cache.popitem(last=False)
                self._close_entry(old_entry)
            return decoder

    def clear(self) -> None:
        with self._lock:
            entries = list(self._cache.values())
            self._cache.clear()
        for entry in entries:
            self._close_entry(entry)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def _install_bounded_torchcodec_decoder_cache() -> _BoundedTorchcodecDecoderCache:
    """Replace LeRobot's process-global unbounded cache exactly once."""

    current = getattr(lerobot_video_utils, "_default_decoder_cache", None)
    if isinstance(current, _BoundedTorchcodecDecoderCache):
        return current
    clear = getattr(current, "clear", None)
    if callable(clear):
        clear()
    bounded = _BoundedTorchcodecDecoderCache()
    lerobot_video_utils._default_decoder_cache = bounded
    return bounded


@lru_cache(maxsize=1)
def _storage_boto3_s3_client():
    profile = os.environ.get("AWS_PROFILE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_REGION"),
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 10, "mode": "adaptive"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _storage_read_table(path: StoragePath, columns: list[str] | None = None):
    if _is_s3_path(path):
        parsed = urlparse(str(path))
        response = _storage_boto3_s3_client().get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )
        try:
            payload = response["Body"].read()
        finally:
            response["Body"].close()
        return pq.read_table(BytesIO(payload), columns=columns, use_threads=False)
    return pq.read_table(path, columns=columns)


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, list):
        return bool(value[0]) if value else default
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    return bool(value)


def _should_log_dataset_summary() -> bool:
    return os.environ.get("RANK", "0") in ("", "0") and os.environ.get("LOCAL_RANK", "0") in ("", "0")


def _image_features_from_info(info: dict[str, Any]) -> dict[str, str]:
    features = info.get("features", {})
    if _COSMOS3_DROID_IMAGE_FEATURES["left"] in features:
        return _COSMOS3_DROID_IMAGE_FEATURES
    return _IMAGE_FEATURES


def _range_frame_count(ranges: Any) -> int:
    total = 0
    if not isinstance(ranges, list):
        return total
    for item in ranges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            total += max(0, int(item[1]) - int(item[0]))
    return total


def _range_window_count(ranges: Any, chunk_length: int) -> int:
    total = 0
    if not isinstance(ranges, list):
        return total
    for item in ranges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            total += max(0, int(item[1]) - int(item[0]) - int(chunk_length))
    return total


def _normalize_frame_ranges(ranges: Any, *, location: str) -> list[tuple[int, int]]:
    if not isinstance(ranges, list):
        raise ValueError(f"{location}: expected a JSON list of [start, end] frame ranges.")
    normalized: list[tuple[int, int]] = []
    for range_index, item in enumerate(ranges):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{location}[{range_index}]: expected [start, end], got {item!r}.")
        start, end = item
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"{location}[{range_index}]: frame bounds must be JSON integers, got {item!r}.")
        if end <= start:
            continue
        normalized.append((int(start), int(end)))
    normalized.sort()
    return normalized


def _intersect_frame_ranges(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    out: list[tuple[int, int]] = []
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    i = 0
    j = 0
    while i < len(left_sorted) and j < len(right_sorted):
        left_start, left_end = left_sorted[i]
        right_start, right_end = right_sorted[j]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            out.append((start, end))
        if left_end <= right_end:
            i += 1
        else:
            j += 1
    return out


def _union_frame_ranges(*range_groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(item for group in range_groups for item in group):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _subtract_frame_ranges(
    source: list[tuple[int, int]],
    removed: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract half-open ``removed`` intervals from ``source`` intervals."""

    if not source or not removed:
        return list(source)
    removed_sorted = sorted(removed)
    output: list[tuple[int, int]] = []
    for source_start, source_end in sorted(source):
        cursor = source_start
        for removed_start, removed_end in removed_sorted:
            if removed_end <= cursor:
                continue
            if removed_start >= source_end:
                break
            if removed_start > cursor:
                output.append((cursor, min(removed_start, source_end)))
            cursor = max(cursor, removed_end)
            if cursor >= source_end:
                break
        if cursor < source_end:
            output.append((cursor, source_end))
    return output


def _cam2cam_ext2_to_ext1(cam2cam_entry: Any) -> np.ndarray | None:
    """Return ext2->ext1; DROID left_cam/right_cam correspond to ext1/ext2."""
    if not isinstance(cam2cam_entry, dict):
        return None
    try:
        ext1_pose = np.asarray(cam2cam_entry["left_cam"]["pose"], dtype=np.float64)
        ext2_pose = np.asarray(cam2cam_entry["right_cam"]["pose"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if ext1_pose.shape != (4, 4) or ext2_pose.shape != (4, 4):
        return None
    if not np.isfinite(ext1_pose).all() or not np.isfinite(ext2_pose).all():
        return None
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(ext1_pose[3], expected_last_row, atol=1e-5):
        return None
    if not np.allclose(ext2_pose[3], expected_last_row, atol=1e-5):
        return None
    try:
        relative = np.linalg.inv(ext1_pose) @ ext2_pose
    except np.linalg.LinAlgError:
        return None
    return relative if np.isfinite(relative).all() else None


def _is_plausible_camera_extrinsics(extrinsics: Any) -> bool:
    values = np.asarray(extrinsics, dtype=np.float64)
    return bool(
        values.shape == (6,)
        and np.isfinite(values).all()
        and np.linalg.norm(values[:3]) <= _MAX_PLAUSIBLE_CAMERA_TRANSLATION_NORM_M
    )


def _ext1_is_physical_left_from_cam2cam(ext2_to_ext1: np.ndarray) -> bool:
    """DROID heuristic validated at 99.54% agreement: positive relative x means ext1 is physical-left."""
    return bool(float(ext2_to_ext1[0, 3]) >= 0.0)


def _episode_is_validation(*, episode_index: int, validation_fraction: float, seed: int) -> bool:
    """Return a stable episode-level train/validation assignment."""

    digest = hashlib.blake2b(
        f"reasoner-val-split:{int(seed)}:{int(episode_index)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    draw = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
    return draw < float(validation_fraction)


class DROIDLeRobotDataset(Dataset):
    """DROID Action dataset.

    Two action layouts:
      * ``action_space="ee_pose"`` (default): 10D ``[pos_delta(3), rot6d_delta(6),
        gripper(1)]``, quantile-normalized (the v1.2 midtrain default).
      * ``action_space="joint_pos"``: 8D ``[joint(7), gripper(1)]`` absolute joint
        commands, NOT normalized, with ``use_state=True`` prepending the initial
        observed joint+gripper state → ``(chunk+1, 8)`` — matching the internal
        ``Cosmos3-Nano-Policy-DROID`` post-training run.
    Filter dictionaries, temporal-segment validation, and image augmentation from
    the production wrapper are intentionally omitted.
    """

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        action_space: str = "ee_pose",
        use_state: bool = False,
        include_cartesian_state_for_language: bool = False,
        use_same_timestep_joint_action: bool = False,
        action_normalization: str | None = "quantile",
        use_image_augmentation: bool = False,
        use_filter_dict: bool = False,
        filter_dict_path: str | None = None,
        filter_dict_key_mode: FilterDictKeyMode = "auto",
        prefilter_by_filter_dict: bool = False,
        camera_order_path: str | None = None,
        cam2base_extrinsic_path: str | None = None,
        success_only: bool = False,
        video_decode_backend: VideoDecodeBackend = "torchcodec",
        wrist_rotate_180: bool = False,
        video_sample_stride: int | None = None,
        use_exterior_plucker_condition: bool = False,
        defer_plucker_generation: bool = False,
        use_all_data_with_cam2cam_fallback: bool = False,
        subtask_annotations_path: str | None = None,
        subtask_episode_filter_path: str | None = None,
        include_idle_stationary_subtasks: bool = False,
        apply_subtask_episode_filter_to_windows: bool = True,
        remove_post_completion_windows: bool = False,
        include_post_completion_move_away_windows: bool = False,
        episode_split: str = "all",
        validation_fraction: float = 0.02,
        validation_split_seed: int = 43,
        episode_indices: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if pose_convention != "backward_framewise":
            raise NotImplementedError("This minimal DROID dataset only supports backward_framewise pose deltas.")
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal DROID dataset only supports concat_view.")
        if action_space not in _ACTION_SPACES:
            raise NotImplementedError(f"action_space must be one of {_ACTION_SPACES}, got {action_space!r}.")
        if use_state and action_space != "joint_pos":
            raise NotImplementedError("use_state is only supported with action_space='joint_pos'.")
        if use_filter_dict and not filter_dict_path:
            raise ValueError("use_filter_dict=True requires filter_dict_path")
        if camera_order_path is not None and cam2base_extrinsic_path is not None:
            raise ValueError("Use either camera_order_path or cam2base_extrinsic_path, not both.")
        if use_exterior_plucker_condition and cam2base_extrinsic_path is None:
            raise ValueError("use_exterior_plucker_condition=True requires cam2base_extrinsic_path.")
        if _as_bool(use_all_data_with_cam2cam_fallback, default=False) and _as_bool(
            use_exterior_plucker_condition,
            default=False,
        ):
            raise ValueError(
                "use_all_data_with_cam2cam_fallback is an ordering-only mode and cannot be combined with "
                "use_exterior_plucker_condition=True; full DROID data does not have complete base-frame extrinsics."
            )
        if video_decode_backend not in ("imageio", "lerobot", "torchcodec", "pyav", "video_reader"):
            raise ValueError(
                "video_decode_backend must be one of {'imageio', 'lerobot', 'torchcodec', 'pyav', 'video_reader'}, "
                f"got {video_decode_backend!r}."
            )
        normalized_episode_split = str(episode_split).strip().lower()
        if normalized_episode_split not in {"all", "train", "val"}:
            raise ValueError("episode_split must be 'all', 'train', or 'val'")
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (float, int))
            or not math.isfinite(float(validation_fraction))
            or not 0.0 < float(validation_fraction) < 1.0
        ):
            raise ValueError("validation_fraction must be a finite number strictly between 0 and 1")
        if isinstance(validation_split_seed, bool) or not isinstance(validation_split_seed, int):
            raise TypeError("validation_split_seed must be an integer")
        self._episode_split = normalized_episode_split
        self._validation_fraction = float(validation_fraction)
        self._validation_split_seed = int(validation_split_seed)

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._video_sample_stride = int(video_sample_stride) if video_sample_stride is not None else None
        if self._video_sample_stride is not None and self._video_sample_stride < 1:
            raise ValueError(f"video_sample_stride must be >= 1, got {self._video_sample_stride}")
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = "droid_lerobot"
        self._domain_id = get_domain_id(self._domain_name)
        # joint_pos uses raw joint values — disable normalization.
        self._action_normalization = None if action_space == "joint_pos" else action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._action_space = action_space
        self._use_state = bool(use_state)
        self._include_cartesian_state_for_language = bool(include_cartesian_state_for_language)
        # Legacy local joint_pos windows use commands from rows [t+1, ..., t+T].
        # Opting in selects the NVIDIA main convention: [t, ..., t+T-1].
        # This only changes commanded-action row selection; the prepended state
        # remains the observed state from row t.
        self._use_same_timestep_joint_action = _as_bool(use_same_timestep_joint_action, default=False)
        self._video_decode_backend = video_decode_backend
        self._wrist_rotate_180 = _as_bool(wrist_rotate_180, default=False)
        self._use_exterior_plucker_condition = _as_bool(use_exterior_plucker_condition, default=False)
        self._defer_plucker_generation = _as_bool(defer_plucker_generation, default=False)
        self._use_all_data_with_cam2cam_fallback = _as_bool(
            use_all_data_with_cam2cam_fallback,
            default=False,
        )
        self._subtask_annotations = (
            load_subtask_annotations(subtask_annotations_path) if subtask_annotations_path is not None else None
        )
        self._subtask_episode_filters: dict[int, EpisodeSubtaskFilter] | None = (
            load_subtask_episode_filters(
                subtask_episode_filter_path,
                include_idle_stationary_subtasks=_as_bool(
                    include_idle_stationary_subtasks,
                    default=False,
                ),
            )
            if subtask_episode_filter_path is not None
            else None
        )
        # Keep the historical behavior by default: a subtask episode filter
        # trims the flat window index. WAM can opt out for a label-only index;
        # the separate post-completion switch selectively removes only groups
        # after the first completed group while retaining other fallback windows.
        self._apply_subtask_episode_filter_to_windows = _as_bool(
            apply_subtask_episode_filter_to_windows,
            default=True,
        )
        # Opt-in selective index filter for standalone WAM: physically remove
        # only groups after the first task_completed group, while the ordinary
        # six-rule keep ranges remain a label-validity mask.
        self._remove_post_completion_windows = _as_bool(remove_post_completion_windows, default=False)
        self._include_post_completion_move_away_windows = _as_bool(
            include_post_completion_move_away_windows,
            default=False,
        )
        if (
            self._remove_post_completion_windows or self._include_post_completion_move_away_windows
        ) and self._subtask_episode_filters is None:
            raise ValueError("post-completion window filtering requires subtask_episode_filter_path.")
        self._exterior_camera_calibrations: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._imageio_fallback_warned = False
        # Per-sample image augmentation (random crop+rescale + color jitter), applied
        # to all views with shared params (temporally + cross-view consistent). Lazy-built.
        self._use_image_augmentation = bool(use_image_augmentation)
        self._image_augmentor: T.Compose | None = None
        if self._use_image_augmentation and self._use_exterior_plucker_condition and not self._defer_plucker_generation:
            raise ValueError(
                "CPU image augmentation with exterior Plucker conditioning is unsafe because the random crop "
                "would not update camera intrinsics. Use IMAGE_AUGMENTATION_BACKEND=cuda so the GPU callback "
                "regenerates the ray map from updated intrinsics."
            )
        # Keep-ranges window filter (internal use_filter_dict): restrict training windows
        # to curated active segments, dropping idle/non-task frames. Off by default; the
        # keep-ranges JSON is supplied via filter_dict_path (an internal data artifact).
        self._use_filter_dict = bool(use_filter_dict)
        self._filter_dict_path = filter_dict_path
        self._filter_dict_key_mode = filter_dict_key_mode
        self._prefilter_by_filter_dict = bool(prefilter_by_filter_dict)
        if self._filter_dict_key_mode not in ("auto", "metadata", "ordered"):
            raise ValueError(
                "filter_dict_key_mode must be one of {'auto', 'metadata', 'ordered'}, "
                f"got {self._filter_dict_key_mode!r}"
            )
        self._filter_dict: dict[str, Any] | None = None
        if self._use_filter_dict:
            self._filter_dict = _storage_read_json(self._filter_dict_path)
            if not isinstance(self._filter_dict, dict):
                raise ValueError(f"Expected filter_dict_path={self._filter_dict_path!r} to contain a JSON object.")
        self._root: StoragePath = root.rstrip("/") if _is_s3_path(root) else Path(root)
        self._info = _storage_read_json(_storage_join(self._root, "meta", "info.json"))
        self._image_features = _image_features_from_info(self._info)
        self._available_s3_video_paths: set[str] | None = None
        if _is_s3_path(self._root):
            self._available_s3_video_paths = {
                str(path) for path in _storage_glob(_storage_join(self._root, "videos"), "**/*.mp4")
            }
        self._camera_order = self._load_camera_order(camera_order_path)
        self._cam2base_extrinsic_path = cam2base_extrinsic_path
        success_key = "stats/is_episode_successful/min"
        episode_columns = [
            "episode_index",
            success_key,
            "data/chunk_index",
            "data/file_index",
            *_FILTER_KEY_COLUMNS,
            *(
                f"videos/{video_key}/{field}"
                for video_key in self._image_features.values()
                for field in ("chunk_index", "file_index", "episode_chunk", "episode_file", "from_timestamp")
            ),
        ]
        episode_columns = list(dict.fromkeys(episode_columns))
        self._episodes: dict[int, dict[str, Any]] = {}
        for path in _storage_glob(_storage_join(self._root, "meta", "episodes"), "chunk-*/file-*.parquet"):
            available = set(_storage_read_schema(path).names)
            columns = [column for column in episode_columns if column in available]
            for row in _storage_read_table(path, columns=columns).to_pylist():
                self._episodes[int(row["episode_index"])] = row
        if episode_indices is not None:
            requested_episodes = {int(value) for value in episode_indices}
            self._episodes = {
                episode_index: row
                for episode_index, row in self._episodes.items()
                if episode_index in requested_episodes
            }
        n_success = 0
        n_failure = 0
        n_missing_success = 0
        for row in self._episodes.values():
            if row.get(success_key) is None:
                n_missing_success += 1
            elif _as_bool(row.get(success_key), default=True):
                n_success += 1
            else:
                n_failure += 1
        if _should_log_dataset_summary():
            print(
                "[droid_lerobot_dataset] episode metadata: "
                f"total={len(self._episodes)}, success={n_success}, failure={n_failure}, "
                f"missing_success_key={n_missing_success}, success_or_missing={n_success + n_missing_success}"
            )

        self._success_only = bool(success_only)
        if self._success_only:
            kept = {}
            n_drop = 0
            for ep_idx, row in self._episodes.items():
                if _as_bool(row.get(success_key), default=True):
                    kept[ep_idx] = row
                else:
                    n_drop += 1
            self._episodes = kept
            self._kept_episode_indices: set[int] | None = set(kept.keys())
            if _should_log_dataset_summary():
                print(f"[droid_lerobot_dataset] success_only=True: kept {len(kept)} / {len(kept) + n_drop} episodes")
        else:
            self._kept_episode_indices = None
        if self._episode_split != "all":
            before = len(self._episodes)
            keep_validation = self._episode_split == "val"
            self._episodes = {
                episode_index: row
                for episode_index, row in self._episodes.items()
                if _episode_is_validation(
                    episode_index=int(episode_index),
                    validation_fraction=self._validation_fraction,
                    seed=self._validation_split_seed,
                )
                == keep_validation
            }
            self._kept_episode_indices = set(self._episodes)
            if self._subtask_annotations is not None:
                self._subtask_annotations = {
                    episode_index: annotations
                    for episode_index, annotations in self._subtask_annotations.items()
                    if episode_index in self._kept_episode_indices
                }
            if self._subtask_episode_filters is not None:
                self._subtask_episode_filters = {
                    episode_index: filter_record
                    for episode_index, filter_record in self._subtask_episode_filters.items()
                    if episode_index in self._kept_episode_indices
                }
            if _should_log_dataset_summary():
                print(
                    f"[droid_lerobot_dataset] episode_split={self._episode_split}: "
                    f"kept {len(self._episodes)} / {before} episodes "
                    f"(validation_fraction={self._validation_fraction:g}, seed={self._validation_split_seed})"
                )
        if self._subtask_episode_filters is not None and self._apply_subtask_episode_filter_to_windows:
            annotated_episode_indices = set(self._subtask_episode_filters)
            if self._kept_episode_indices is not None:
                annotated_episode_indices &= self._kept_episode_indices
            before = len(self._episodes)
            self._episodes = {
                ep_idx: row for ep_idx, row in self._episodes.items() if int(ep_idx) in annotated_episode_indices
            }
            self._kept_episode_indices = set(self._episodes)
            if _should_log_dataset_summary():
                filter_range_count = sum(
                    len(filter_record.keep_ranges) if filter_record.keep_ranges else 1
                    for filter_record in self._subtask_episode_filters.values()
                    if int(filter_record.keep_start_frame) < int(filter_record.keep_end_frame)
                )
                print(
                    "[droid_lerobot_dataset] subtask_episode_filter_path: "
                    f"kept {len(self._episodes)} / {before} episodes with usable subtask clips, "
                    f"ranges={filter_range_count}, max_group_duration_frames=150"
                )
        elif self._subtask_episode_filters is not None and _should_log_dataset_summary():
            print(
                "[droid_lerobot_dataset] subtask_episode_filter_path: label-only mode; "
                f"not pruning the current {len(self._episodes)} episodes/windows by the subtask filter; "
                "using its keep-ranges only to validate subgoal labels"
            )
        if self._cam2base_extrinsic_path is not None:
            if self._use_all_data_with_cam2cam_fallback:
                self._apply_all_data_camera_order(Path(self._cam2base_extrinsic_path))
            else:
                self._apply_cam2base_episode_filter(Path(self._cam2base_extrinsic_path))
        if self._prefilter_by_filter_dict:
            if not self._use_filter_dict or self._filter_dict is None:
                raise ValueError("prefilter_by_filter_dict=True requires use_filter_dict=True and filter_dict_path.")
            episode_filter_keys = self._resolve_episode_filter_keys(self._filter_dict)
            kept_by_filter = set(episode_filter_keys)
            if self._kept_episode_indices is not None:
                kept_by_filter &= self._kept_episode_indices
            before = len(self._episodes)
            self._episodes = {ep_idx: row for ep_idx, row in self._episodes.items() if ep_idx in kept_by_filter}
            self._kept_episode_indices = set(self._episodes)
            self._prefilter_episode_filter_keys = {
                int(ep_idx): key for ep_idx, key in episode_filter_keys.items() if ep_idx in self._episodes
            }
            if _should_log_dataset_summary():
                print(
                    "[droid_lerobot_dataset] prefilter_by_filter_dict=True: "
                    f"kept {len(self._episodes)} / {before} episodes before parquet loading"
                )
        else:
            self._prefilter_episode_filter_keys = None
        self._tasks = {
            int(row["task_index"]): str(row["task"])
            for row in _storage_read_table(_storage_join(self._root, "meta", "tasks.parquet")).to_pylist()
        }

        # Compact, lazy frame index. Materializing every frame as a Python dict
        # (``sorted(... pq.read_table(path).to_pylist() ...)``) does not scale:
        # the full DROID success shard is ~18M frames, which is tens of GB of
        # dicts plus an 18M-element Python sort at construction, and each
        # DataLoader worker faults in its own copy. Instead we read only the
        # columns the sample builder needs into contiguous numpy arrays
        # (~1 GB total) -- read-only after init, so worker forks share them
        # copy-on-write.
        if action_space == "joint_pos":
            feature_cols = [
                _JOINT_ACTION_FEATURE,
                _ACTION_GRIPPER_FEATURE,
                _JOINT_STATE_FEATURE,
                _GRIPPER_STATE_FEATURE,
            ]
            if self._include_cartesian_state_for_language:
                feature_cols.append(_STATE_FEATURE)
        else:
            feature_cols = [_STATE_FEATURE, _ACTION_GRIPPER_FEATURE]
        columns = ["index", "episode_index", "task_index", "timestamp", *feature_cols]
        index_parts, episode_parts, task_parts, ts_parts = [], [], [], []
        feature_parts: dict[str, list] = {c: [] for c in feature_cols}
        data_paths = _storage_glob(_storage_join(self._root, "data"), "chunk-*/file-*.parquet")
        if self._prefilter_by_filter_dict and self._kept_episode_indices:
            selected_data_paths = set()
            for episode in self._episodes.values():
                data_chunk = episode.get("data/chunk_index")
                data_file = episode.get("data/file_index")
                if data_chunk is None or data_file is None:
                    selected_data_paths = set()
                    break
                selected_data_paths.add(
                    _storage_join(
                        self._root,
                        "data",
                        f"chunk-{int(data_chunk):03d}",
                        f"file-{int(data_file):03d}.parquet",
                    )
                )
            if selected_data_paths:
                data_paths = sorted(
                    (path for path in selected_data_paths if _storage_is_file(path)),
                    key=str,
                )
                if _should_log_dataset_summary():
                    print(
                        "[droid_lerobot_dataset] prefilter_by_filter_dict=True: "
                        f"reading {len(data_paths)} selected parquet files"
                    )
        kept_episode_array = (
            np.fromiter(self._kept_episode_indices, dtype=np.int64)
            if self._prefilter_by_filter_dict
            and self._kept_episode_indices is not None
            and len(self._kept_episode_indices) > 0
            else None
        )
        for path in tqdm(
            data_paths,
            desc="[droid_lerobot_dataset] loading parquet",
            disable=not _should_log_dataset_summary(),
            mininterval=5.0,
        ):
            table = _storage_read_table(path, columns=columns)
            episode_np = table["episode_index"].to_numpy().astype(np.int64)
            if kept_episode_array is not None:
                row_mask = np.isin(episode_np, kept_episode_array)
                if not row_mask.any():
                    continue
            else:
                row_mask = None
            index_np = table["index"].to_numpy()
            task_np = table["task_index"].to_numpy()
            ts_np = table["timestamp"].to_numpy()
            if row_mask is not None:
                index_np = index_np[row_mask]
                episode_np = episode_np[row_mask]
                task_np = task_np[row_mask]
                ts_np = ts_np[row_mask]
            index_parts.append(index_np)
            episode_parts.append(episode_np)
            task_parts.append(task_np)
            ts_parts.append(ts_np)
            for c in feature_cols:
                if c in _LIST_COLUMNS:
                    values = np.asarray(table[c].to_pylist(), dtype=np.float32)
                else:
                    values = np.asarray(table[c].to_numpy(), dtype=np.float32)
                feature_parts[c].append(values[row_mask] if row_mask is not None else values)
        if not index_parts:
            raise ValueError("No DROID parquet rows remain after episode filtering.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        # Per-feature arrays keyed by parquet column name (read-only after init).
        self._feat = {c: np.concatenate(feature_parts[c], axis=0).astype(np.float32)[order] for c in feature_cols}

        # Group frames into episodes and keep only within-episode chunk windows.
        # The global frame index is ordered by episode in LeRobot v3, so episodes
        # are contiguous blocks once sorted by ``index``. The previous code sliced
        # the flat row list (``rows[idx : idx + chunk + 1]``) with no boundary
        # guard, so ~one chunk of samples per episode silently mixed two episodes;
        # restricting to in-episode windows yields ``total - n_episodes * chunk``
        # valid samples (matching the production dataset).
        assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        if self._kept_episode_indices is not None:
            ep_counts = np.where(
                np.isin(self._ep_vals, np.fromiter(self._kept_episode_indices, dtype=np.int64)),
                ep_counts,
                0,
            )
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)
        self._use_segment_window_filter = bool(
            self._use_filter_dict
            or (self._subtask_episode_filters is not None and self._apply_subtask_episode_filter_to_windows)
            or (self._subtask_episode_filters is not None and self._remove_post_completion_windows)
        )
        if self._use_segment_window_filter:
            filter_dict = self._filter_dict
            if self._use_filter_dict:
                assert filter_dict is not None
                n_filter_clips = 0
                n_filter_frames = 0
                n_filter_malformed_values = 0
                for ranges in filter_dict.values():
                    if not isinstance(ranges, list):
                        n_filter_malformed_values += 1
                        continue
                    for item in ranges:
                        if not isinstance(item, (list, tuple)) or len(item) != 2:
                            n_filter_malformed_values += 1
                            continue
                        s, e = item
                        n_filter_clips += 1
                        n_filter_frames += max(0, int(e) - int(s))
                if _should_log_dataset_summary():
                    print(
                        "[droid_lerobot_dataset] filter_dict input: "
                        f"episodes={len(filter_dict)}, clips={n_filter_clips}, "
                        f"clip_frames={n_filter_frames}, malformed_values={n_filter_malformed_values}"
                    )
                episode_filter_keys = self._prefilter_episode_filter_keys or self._resolve_episode_filter_keys(
                    filter_dict
                )
            else:
                n_filter_clips = 0
                n_filter_frames = 0
                episode_filter_keys = {}

            seg_ep_pos, seg_win_start, seg_len = [], [], []
            n_episode_with_ranges = 0
            n_ranges_seen = 0
            n_ranges_kept = 0
            n_annotation_matched_episodes = 0
            for pos in tqdm(
                range(len(self._ep_vals)),
                desc="[droid_lerobot_dataset] applying window filters",
                disable=not _should_log_dataset_summary(),
                mininterval=5.0,
            ):
                episode_index = int(self._ep_vals[pos])
                episode_frame_count = int(ep_counts[pos])
                valid = int(max(0, episode_frame_count - self._chunk_length))
                if valid <= 0:
                    continue
                key = episode_filter_keys.get(episode_index)
                frame_ranges = self._effective_frame_ranges_for_episode(
                    episode_index=episode_index,
                    episode_frame_count=episode_frame_count,
                    filter_key=key,
                )
                if not frame_ranges:
                    continue
                n_episode_with_ranges += 1
                if self._subtask_episode_filters is not None and episode_index in self._subtask_episode_filters:
                    n_annotation_matched_episodes += 1
                for s, e in frame_ranges:
                    n_ranges_seen += 1
                    ws = max(int(s), 0)
                    we = min(int(e) - self._chunk_length, valid)
                    if we - ws > 0:
                        seg_ep_pos.append(pos)
                        seg_win_start.append(ws)
                        seg_len.append(we - ws)
                        n_ranges_kept += 1
            self._seg_ep_pos = np.asarray(seg_ep_pos, dtype=np.int64)
            self._seg_win_start = np.asarray(seg_win_start, dtype=np.int64)
            self._seg_cum = np.cumsum(seg_len).astype(np.int64) if seg_len else np.zeros(0, dtype=np.int64)
            if _should_log_dataset_summary():
                print(
                    "[droid_lerobot_dataset] window-filter summary: "
                    f"use_filter_dict={self._use_filter_dict}, "
                    f"use_subtask_episode_filter={self._subtask_episode_filters is not None}, "
                    f"apply_subtask_episode_filter_to_windows={self._apply_subtask_episode_filter_to_windows}, "
                    f"remove_post_completion_windows={self._remove_post_completion_windows}, "
                    f"matched_episodes={n_episode_with_ranges}, "
                    f"annotation_matched_episodes={n_annotation_matched_episodes}, "
                    f"kept_ranges={n_ranges_kept}/{n_ranges_seen}, "
                    f"kept_windows={int(self._seg_cum[-1]) if self._seg_cum.size else 0}"
                )

        if _should_log_dataset_summary():
            compact_bytes = (
                self._row_episode.nbytes
                + self._row_task.nbytes
                + self._row_timestamp.nbytes
                + sum(arr.nbytes for arr in self._feat.values())
                + self._ep_vals.nbytes
                + self._ep_starts.nbytes
                + self._valid_cum.nbytes
            )
            if self._use_segment_window_filter:
                compact_bytes += self._seg_ep_pos.nbytes + self._seg_win_start.nbytes + self._seg_cum.nbytes
            raw_windows = int(self._valid_cum[-1]) if self._valid_cum.size else 0
            kept_windows = len(self)
            print(
                "[droid_lerobot_dataset] compact index summary: "
                f"root={self._root}, episodes={len(self._episodes)}, "
                f"frames={len(self._row_episode)}, raw_windows={raw_windows}, "
                f"kept_windows={kept_windows}, chunk_length={self._chunk_length}, "
                f"fps={self._fps:g}, action_space={self._action_space}, "
                f"video_sample_stride={self._video_sample_stride}, video_fps={self.video_fps:g}, "
                f"use_state={self._use_state}, "
                f"use_same_timestep_joint_action={self._use_same_timestep_joint_action}, "
                f"use_filter_dict={self._use_filter_dict}, "
                f"use_segment_window_filter={self._use_segment_window_filter}, "
                f"apply_subtask_episode_filter_to_windows={self._apply_subtask_episode_filter_to_windows}, "
                f"remove_post_completion_windows={self._remove_post_completion_windows}, "
                f"feature_cols={list(self._feat.keys())}, approx_index_mem={compact_bytes / 1024**2:.1f}MiB"
            )

    def _filter_key_candidates(self, episode: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for column in _FILTER_KEY_COLUMNS:
            value = episode.get(column)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None:
                    continue
                text = str(item)
                if not text:
                    continue
                candidates.append(text)
                if not text.startswith("gs://"):
                    candidates.append(
                        "gs://xembodiment_data/r2d2/r2d2-data-full/"
                        f"{text}/recordings/MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{text}/trajectory.h5"
                    )
        return candidates

    def _resolve_episode_filter_keys(self, filter_dict: dict[str, Any]) -> dict[int, str]:
        resolved: dict[int, str] = {}
        for ep_idx, episode in self._episodes.items():
            for candidate in self._filter_key_candidates(episode):
                if candidate in filter_dict:
                    resolved[int(ep_idx)] = candidate
                    break
        if resolved:
            if _should_log_dataset_summary():
                print(
                    "[droid_lerobot_dataset] filter_dict key mode: metadata "
                    f"matched {len(resolved)} / {len(self._episodes)} episodes"
                )
            return resolved

        if self._filter_dict_key_mode == "ordered":
            if len(filter_dict) != len(self._episodes):
                raise ValueError(
                    "filter_dict_key_mode='ordered' requires exactly one filter key per episode: "
                    f"filter_keys={len(filter_dict)}, episodes={len(self._episodes)}"
                )
            if _should_log_dataset_summary():
                print(
                    "[droid_lerobot_dataset] WARNING: filter_dict_key_mode='ordered' is using JSON insertion order "
                    "as episode_index order. Use only when the filter file was generated in the same LeRobot order."
                )
            return {int(ep_idx): key for ep_idx, key in zip(sorted(self._episodes), filter_dict.keys(), strict=True)}

        available_columns = sorted(self._episodes[next(iter(self._episodes))].keys()) if self._episodes else []
        raise ValueError(
            "use_filter_dict=True but this DROID LeRobot metadata has no trajectory id/path column that can be "
            "matched to the keep-ranges JSON keys. The current dataset exposes columns "
            f"{available_columns}, while the filter JSON uses keys like {next(iter(filter_dict))!r}. "
            "Use a LeRobot export that preserves episode_id/trajectory_path, disable use_filter_dict, or set "
            "filter_dict_key_mode='ordered' only if JSON order is known to match episode_index order."
        )

    def _annotation_keep_ranges(self, episode_index: int) -> list[tuple[int, int]] | None:
        if self._subtask_episode_filters is None:
            return None
        keep_window = self._subtask_episode_filters.get(int(episode_index))
        if keep_window is None:
            return []
        if keep_window.keep_ranges:
            return [(int(start), int(end)) for start, end in keep_window.keep_ranges]
        if int(keep_window.keep_end_frame) <= int(keep_window.keep_start_frame):
            return []
        return [(int(keep_window.keep_start_frame), int(keep_window.keep_end_frame))]

    def _annotation_post_completion_ranges(self, episode_index: int) -> list[tuple[int, int]]:
        if self._subtask_episode_filters is None:
            return []
        filter_record = self._subtask_episode_filters.get(int(episode_index))
        if filter_record is None:
            return []
        return [
            (int(start), int(end))
            for start, end in getattr(filter_record, "post_completion_ranges", ())
        ]

    def _annotation_post_completion_move_away_ranges(self, episode_index: int) -> list[tuple[int, int]]:
        if self._subtask_episode_filters is None:
            return []
        filter_record = self._subtask_episode_filters.get(int(episode_index))
        if filter_record is None:
            return []
        return [
            (int(start), int(end))
            for start, end in getattr(filter_record, "post_completion_move_away_ranges", ())
        ]

    def _effective_frame_ranges_for_episode(
        self,
        *,
        episode_index: int,
        episode_frame_count: int,
        filter_key: str | None,
    ) -> list[tuple[int, int]]:
        if self._use_filter_dict:
            filter_dict = self._filter_dict
            assert filter_dict is not None
            if filter_key is None:
                return []
            raw_ranges = filter_dict.get(filter_key)
            if raw_ranges is None:
                return []
            frame_ranges = _normalize_frame_ranges(raw_ranges, location=f"filter_dict[{filter_key!r}]")
        else:
            frame_ranges = [(0, int(episode_frame_count))]

        # In label-only mode the annotation filter remains available to the
        # subgoal lookup as a validity mask, but must not intersect the parent
        # window index.  The default path keeps the historical intersection.
        annotation_ranges = (
            self._annotation_keep_ranges(episode_index)
            if getattr(self, "_apply_subtask_episode_filter_to_windows", True)
            else None
        )
        if annotation_ranges is not None:
            frame_ranges = _intersect_frame_ranges(frame_ranges, annotation_ranges)
        if getattr(self, "_remove_post_completion_windows", False):
            frame_ranges = _subtract_frame_ranges(
                frame_ranges,
                self._annotation_post_completion_ranges(episode_index),
            )
        if getattr(self, "_include_post_completion_move_away_windows", False):
            frame_ranges = _union_frame_ranges(
                frame_ranges,
                self._annotation_post_completion_move_away_ranges(episode_index),
            )
        return frame_ranges

    def _subtask_window_is_label_eligible(
        self,
        *,
        episode_index: int,
        window_start_frame: int,
        window_end_frame: int,
    ) -> bool:
        """Whether a requested half-open target interval lies in a keep range.

        This check is only active for the opt-in label-only mode.  The requested
        half-open interval must be covered continuously by the union of the keep
        ranges. Callers pass either the preview/window interval or the
        action-aligned target interval; the subgoal dataset uses the latter so
        label eligibility follows the exact command timeline. Missing episodes
        and gaps therefore retain their WAM flow sample but cannot provide a
        gold subgoal condition.
        """

        filters = getattr(self, "_subtask_episode_filters", None)
        if filters is None or getattr(self, "_apply_subtask_episode_filter_to_windows", True):
            return True
        start = int(window_start_frame)
        end = int(window_end_frame)
        if start < 0 or end <= start:
            return False
        ranges = self._annotation_keep_ranges(int(episode_index))
        if not ranges:
            return False
        cursor = start
        for range_start, range_end in ranges:
            range_start = int(range_start)
            range_end = int(range_end)
            if range_end <= cursor:
                continue
            if range_start > cursor:
                return False
            cursor = max(cursor, range_end)
            if cursor >= end:
                return True
        return False

    @staticmethod
    def _relative_path_to_keep_range_key(relative_path: str) -> str:
        return (
            "gs://xembodiment_data/r2d2/r2d2-data-full/"
            f"{relative_path}/recordings/MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{relative_path}/trajectory.h5"
        )

    def _apply_cam2base_episode_filter(self, cam2base_extrinsic_path: Path) -> None:
        calib_dir = cam2base_extrinsic_path.parent
        camera_serials_path = calib_dir / _CAMERA_SERIALS_FILENAME
        episode_id_to_path_path = calib_dir / _EPISODE_ID_TO_PATH_FILENAME
        intrinsics_path = calib_dir / _INTRINSICS_FILENAME
        if not cam2base_extrinsic_path.is_file():
            raise FileNotFoundError(cam2base_extrinsic_path)
        if not camera_serials_path.is_file():
            raise FileNotFoundError(camera_serials_path)
        if not episode_id_to_path_path.is_file():
            raise FileNotFoundError(episode_id_to_path_path)
        if self._use_exterior_plucker_condition and not intrinsics_path.is_file():
            raise FileNotFoundError(intrinsics_path)

        with cam2base_extrinsic_path.open() as f:
            cam2base = json.load(f)
        with camera_serials_path.open() as f:
            camera_serials = json.load(f)
        with episode_id_to_path_path.open() as f:
            episode_id_to_path = json.load(f)
        if getattr(self, "_use_exterior_plucker_condition", False):
            with intrinsics_path.open() as f:
                intrinsics = json.load(f)
        else:
            intrinsics = None
        if (
            not isinstance(cam2base, dict)
            or not isinstance(camera_serials, dict)
            or not isinstance(episode_id_to_path, dict)
        ):
            raise ValueError("cam2base calibration files must contain JSON objects.")
        if intrinsics is not None and not isinstance(intrinsics, dict):
            raise ValueError(f"{intrinsics_path} must contain a JSON object.")
        path_to_episode_id = {str(value): str(key) for key, value in episode_id_to_path.items()}

        filter_dict = self._filter_dict
        new_filter_dict: dict[str, Any] | None = {} if filter_dict is not None else None
        new_prefilter_keys: dict[int, str] = {}
        selected: dict[int, dict[str, Any]] = {}
        camera_order: dict[int, tuple[str, str]] = {}
        exterior_camera_calibrations: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        counts = {
            "input_episodes": len(self._episodes),
            "matched_episode_id_to_path": 0,
            "matched_keep_ranges": 0,
            "matched_cam2base": 0,
            "selected_episodes": 0,
            "selected_keep_range_frames": 0,
            "selected_training_windows_chunk_length": 0,
            "selected_ext1_bottom_left": 0,
            "selected_ext2_bottom_left": 0,
            "dropped_missing_episode_id_path": 0,
            "dropped_missing_keep_ranges": 0,
            "dropped_missing_camera_serials": 0,
            "dropped_missing_cam2base": 0,
            "dropped_missing_intrinsics": 0,
            "dropped_invalid_intrinsics": 0,
            "dropped_missing_video": 0,
            "dropped_no_keep_range_windows": 0,
        }

        required_video_keys = (
            self._image_features["wrist"],
            self._image_features["left"],
            self._image_features["right"],
        )
        for ep_idx, episode in self._episodes.items():
            raw_episode_id = episode.get("episode_id")
            raw_episode_id_text = str(raw_episode_id) if raw_episode_id is not None else ""
            if raw_episode_id_text in episode_id_to_path:
                calib_episode_id = raw_episode_id_text
                relative_path = str(episode_id_to_path[raw_episode_id_text])
            elif raw_episode_id_text in path_to_episode_id:
                calib_episode_id = path_to_episode_id[raw_episode_id_text]
                relative_path = raw_episode_id_text
            else:
                counts["dropped_missing_episode_id_path"] += 1
                continue
            counts["matched_episode_id_to_path"] += 1

            keep_key = self._relative_path_to_keep_range_key(str(relative_path))
            ranges = filter_dict.get(keep_key) if isinstance(filter_dict, dict) else None
            if filter_dict is not None and ranges is None:
                counts["dropped_missing_keep_ranges"] += 1
                continue
            if ranges is not None:
                counts["matched_keep_ranges"] += 1
                if _range_window_count(ranges, self._chunk_length) <= 0:
                    counts["dropped_no_keep_range_windows"] += 1
                    continue

            serial_entry = camera_serials.get(calib_episode_id)
            if not isinstance(serial_entry, dict):
                counts["dropped_missing_camera_serials"] += 1
                continue
            ext1_serial = serial_entry.get("ext1_cam_serial")
            ext2_serial = serial_entry.get("ext2_cam_serial")
            calib_entry = cam2base.get(calib_episode_id)
            if not isinstance(calib_entry, dict):
                counts["dropped_missing_cam2base"] += 1
                continue
            ext1_extrinsics = calib_entry.get(str(ext1_serial))
            ext2_extrinsics = calib_entry.get(str(ext2_serial))
            if ext1_extrinsics is None or ext2_extrinsics is None:
                counts["dropped_missing_cam2base"] += 1
                continue
            counts["matched_cam2base"] += 1

            ext1_intrinsics: dict[str, Any] | None = None
            ext2_intrinsics: dict[str, Any] | None = None
            if intrinsics is not None:
                episode_intrinsics = intrinsics.get(calib_episode_id)
                if not isinstance(episode_intrinsics, dict):
                    counts["dropped_missing_intrinsics"] += 1
                    continue
                ext1_intrinsics = episode_intrinsics.get(str(ext1_serial))
                ext2_intrinsics = episode_intrinsics.get(str(ext2_serial))
                if not isinstance(ext1_intrinsics, dict) or not isinstance(ext2_intrinsics, dict):
                    counts["dropped_missing_intrinsics"] += 1
                    continue
                try:
                    for intrinsic_entry in (ext1_intrinsics, ext2_intrinsics):
                        scaled_camera_matrix(
                            intrinsic_entry["cameraMatrix"],
                            source_width=int(intrinsic_entry["width"]),
                            source_height=int(intrinsic_entry["height"]),
                            target_width=1,
                            target_height=1,
                        )
                except (KeyError, TypeError, ValueError, OverflowError):
                    counts["dropped_invalid_intrinsics"] += 1
                    continue

            missing_video = False
            for video_key in required_video_keys:
                if not self._video_is_file(self._video_path(episode, video_key)):
                    missing_video = True
                    break
            if missing_video:
                counts["dropped_missing_video"] += 1
                continue

            # exterior_1/2 这个名字本身不能可靠表示物理左右。这里用
            # 标定得到的 camera-to-base y 坐标判断：y 更大的相机放到
            # 机器人真实物理左侧。这个判断和视频是原始大 mp4 还是
            # per-episode H.264 切片无关。
            ext1_y = float(ext1_extrinsics[1])
            ext2_y = float(ext2_extrinsics[1])
            if ext1_y >= ext2_y:
                bottom_left_key = self._image_features["left"]
                bottom_right_key = self._image_features["right"]
                bottom_left_calibration = (ext1_extrinsics, ext1_intrinsics)
                bottom_right_calibration = (ext2_extrinsics, ext2_intrinsics)
                counts["selected_ext1_bottom_left"] += 1
            else:
                bottom_left_key = self._image_features["right"]
                bottom_right_key = self._image_features["left"]
                bottom_left_calibration = (ext2_extrinsics, ext2_intrinsics)
                bottom_right_calibration = (ext1_extrinsics, ext1_intrinsics)
                counts["selected_ext2_bottom_left"] += 1

            selected[int(ep_idx)] = episode
            camera_order[int(ep_idx)] = (bottom_left_key, bottom_right_key)
            if intrinsics is not None:
                assert bottom_left_calibration[1] is not None and bottom_right_calibration[1] is not None
                exterior_camera_calibrations[int(ep_idx)] = (
                    {
                        "extrinsics": bottom_left_calibration[0],
                        "intrinsics": bottom_left_calibration[1],
                    },
                    {
                        "extrinsics": bottom_right_calibration[0],
                        "intrinsics": bottom_right_calibration[1],
                    },
                )
            counts["selected_episodes"] += 1
            if ranges is not None and new_filter_dict is not None:
                new_filter_dict[keep_key] = ranges
                new_prefilter_keys[int(ep_idx)] = keep_key
                counts["selected_keep_range_frames"] += _range_frame_count(ranges)
                counts["selected_training_windows_chunk_length"] += _range_window_count(ranges, self._chunk_length)

        self._episodes = selected
        self._camera_order = camera_order
        self._exterior_camera_calibrations = exterior_camera_calibrations
        if self._kept_episode_indices is not None:
            self._kept_episode_indices &= set(selected)
        else:
            self._kept_episode_indices = set(selected)
        if new_filter_dict is not None:
            self._filter_dict = new_filter_dict
            self._prefilter_episode_filter_keys = new_prefilter_keys
        self._cam2base_filter_counts = counts
        if _should_log_dataset_summary():
            print(
                "[droid_lerobot_dataset] cam2base_extrinsic_path filter: "
                f"selected={counts['selected_episodes']} / {counts['input_episodes']}, "
                f"matched_cam2base={counts['matched_cam2base']}, "
                f"missing_video={counts['dropped_missing_video']}, "
                f"keep_frames={counts['selected_keep_range_frames']}, "
                f"windows_chunk{self._chunk_length}={counts['selected_training_windows_chunk_length']}, "
                f"bottom_left ext1/ext2={counts['selected_ext1_bottom_left']}/{counts['selected_ext2_bottom_left']}, "
                f"plucker={self._use_exterior_plucker_condition}, "
                f"missing/invalid_intrinsics={counts['dropped_missing_intrinsics']}/"
                f"{counts['dropped_invalid_intrinsics']}"
            )
        if not selected:
            raise ValueError(
                "cam2base_extrinsic_path selected 0 DROID episodes. "
                f"Counts: {counts}. This usually means the required wrist/exterior videos are not downloaded yet."
            )

    def _apply_all_data_camera_order(self, cam2base_extrinsic_path: Path) -> None:
        """Keep all usable episodes while resolving physical left/right from available calibration.

        Priority:
        1. Two plausible direct camera-to-base poses: compare base y.
        2. One plausible direct pose: its base-y sign fixes ordering.
        3. No plausible direct pose: cam2cam relative x fixes ordering under DROID's
           one-camera-per-side prior.
        4. No usable calibration: preserve ext1/ext2 order.

        This mode is deliberately ordering-only and never creates a camera condition.
        """
        calib_dir = cam2base_extrinsic_path.parent
        standard_cam2base_path = calib_dir / _CAM2BASE_STANDARD_FILENAME
        cam2cam_path = calib_dir / _CAM2CAM_FILENAME
        camera_serials_path = calib_dir / _CAMERA_SERIALS_FILENAME
        episode_id_to_path_path = calib_dir / _EPISODE_ID_TO_PATH_FILENAME
        required_paths = [
            cam2base_extrinsic_path,
            standard_cam2base_path,
            cam2cam_path,
            camera_serials_path,
            episode_id_to_path_path,
        ]
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        def load_json(path: Path) -> dict[str, Any]:
            with path.open() as f:
                value = json.load(f)
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a JSON object")
            return value

        cam2base_superset = load_json(cam2base_extrinsic_path)
        cam2base_standard = load_json(standard_cam2base_path)
        cam2cam = load_json(cam2cam_path)
        camera_serials = load_json(camera_serials_path)
        episode_id_to_path = load_json(episode_id_to_path_path)
        path_to_episode_id = {str(value): str(key) for key, value in episode_id_to_path.items()}
        cam2cam_by_path = {
            str(entry["relative_path"]): (str(episode_id), entry)
            for episode_id, entry in cam2cam.items()
            if isinstance(entry, dict) and entry.get("relative_path") is not None
        }

        filter_dict = self._filter_dict
        new_filter_dict: dict[str, Any] | None = {} if filter_dict is not None else None
        new_prefilter_keys: dict[int, str] = {}
        selected: dict[int, dict[str, Any]] = {}
        camera_order: dict[int, tuple[str, str]] = {}
        counts = {
            "input_episodes": len(self._episodes),
            "selected_episodes": 0,
            "selected_keep_range_frames": 0,
            "selected_training_windows_chunk_length": 0,
            "order_from_two_cam2base": 0,
            "order_from_single_ext1_cam2base": 0,
            "order_from_single_ext2_cam2base": 0,
            "order_from_cam2cam": 0,
            "order_from_default": 0,
            "selected_ext1_bottom_left": 0,
            "selected_ext2_bottom_left": 0,
            "invalid_direct_cam2base": 0,
            "ambiguous_single_cam2base_y": 0,
            "ambiguous_cam2cam_x": 0,
            "dropped_missing_keep_ranges": 0,
            "dropped_no_keep_range_windows": 0,
            "dropped_missing_video": 0,
        }

        required_video_keys = (
            self._image_features["wrist"],
            self._image_features["left"],
            self._image_features["right"],
        )

        def direct_extrinsics(episode_id: str | None, serial: Any) -> tuple[Any | None, bool]:
            if episode_id is None or serial is None:
                return None, False
            serial = str(serial)
            was_present = False
            for table in (cam2base_superset, cam2base_standard):
                entry = table.get(episode_id)
                if not isinstance(entry, dict) or serial not in entry:
                    continue
                was_present = True
                value = entry[serial]
                if _is_plausible_camera_extrinsics(value):
                    return value, True
            return None, was_present

        for ep_idx, episode in self._episodes.items():
            raw_episode_id = str(episode.get("episode_id") or "")
            if raw_episode_id in episode_id_to_path:
                calib_episode_id: str | None = raw_episode_id
                relative_path = str(episode_id_to_path[raw_episode_id])
            else:
                relative_path = raw_episode_id
                calib_episode_id = path_to_episode_id.get(relative_path)

            cam2cam_pair = cam2cam_by_path.get(relative_path)
            if calib_episode_id is None and cam2cam_pair is not None:
                calib_episode_id = cam2cam_pair[0]
            cam2cam_entry = cam2cam.get(calib_episode_id) if calib_episode_id is not None else None
            if not isinstance(cam2cam_entry, dict) and cam2cam_pair is not None:
                cam2cam_entry = cam2cam_pair[1]
            ext2_to_ext1 = _cam2cam_ext2_to_ext1(cam2cam_entry)

            keep_key = self._relative_path_to_keep_range_key(relative_path)
            ranges = filter_dict.get(keep_key) if isinstance(filter_dict, dict) else None
            if filter_dict is not None and ranges is None:
                counts["dropped_missing_keep_ranges"] += 1
                continue
            if ranges is not None and _range_window_count(ranges, self._chunk_length) <= 0:
                counts["dropped_no_keep_range_windows"] += 1
                continue

            if any(not self._video_is_file(self._video_path(episode, key)) for key in required_video_keys):
                counts["dropped_missing_video"] += 1
                continue

            serial_entry = camera_serials.get(calib_episode_id, {}) if calib_episode_id is not None else {}
            ext1_serial = serial_entry.get("ext1_cam_serial") if isinstance(serial_entry, dict) else None
            ext2_serial = serial_entry.get("ext2_cam_serial") if isinstance(serial_entry, dict) else None
            ext1_extrinsics, ext1_was_present = direct_extrinsics(calib_episode_id, ext1_serial)
            ext2_extrinsics, ext2_was_present = direct_extrinsics(calib_episode_id, ext2_serial)
            counts["invalid_direct_cam2base"] += int(ext1_was_present and ext1_extrinsics is None)
            counts["invalid_direct_cam2base"] += int(ext2_was_present and ext2_extrinsics is None)

            direct_count = int(ext1_extrinsics is not None) + int(ext2_extrinsics is not None)
            if direct_count == 2:
                ext1_is_left = float(ext1_extrinsics[1]) >= float(ext2_extrinsics[1])
                counts["order_from_two_cam2base"] += 1
            elif ext1_extrinsics is not None:
                ext1_y = float(ext1_extrinsics[1])
                ext1_is_left = ext1_y >= 0.0
                counts["order_from_single_ext1_cam2base"] += 1
                counts["ambiguous_single_cam2base_y"] += int(abs(ext1_y) < 0.02)
            elif ext2_extrinsics is not None:
                ext2_y = float(ext2_extrinsics[1])
                ext1_is_left = ext2_y < 0.0
                counts["order_from_single_ext2_cam2base"] += 1
                counts["ambiguous_single_cam2base_y"] += int(abs(ext2_y) < 0.02)
            elif ext2_to_ext1 is not None:
                ext1_is_left = _ext1_is_physical_left_from_cam2cam(ext2_to_ext1)
                counts["order_from_cam2cam"] += 1
                counts["ambiguous_cam2cam_x"] += int(abs(float(ext2_to_ext1[0, 3])) < 0.02)
            else:
                ext1_is_left = True
                counts["order_from_default"] += 1

            if ext1_is_left:
                bottom_left_key = self._image_features["left"]
                bottom_right_key = self._image_features["right"]
                counts["selected_ext1_bottom_left"] += 1
            else:
                bottom_left_key = self._image_features["right"]
                bottom_right_key = self._image_features["left"]
                counts["selected_ext2_bottom_left"] += 1

            selected[int(ep_idx)] = episode
            camera_order[int(ep_idx)] = (bottom_left_key, bottom_right_key)
            counts["selected_episodes"] += 1
            if ranges is not None and new_filter_dict is not None:
                new_filter_dict[keep_key] = ranges
                new_prefilter_keys[int(ep_idx)] = keep_key
                counts["selected_keep_range_frames"] += _range_frame_count(ranges)
                counts["selected_training_windows_chunk_length"] += _range_window_count(ranges, self._chunk_length)

        self._episodes = selected
        self._camera_order = camera_order
        self._exterior_camera_calibrations = {}
        self._kept_episode_indices = set(selected)
        if new_filter_dict is not None:
            self._filter_dict = new_filter_dict
            self._prefilter_episode_filter_keys = new_prefilter_keys
        self._cam2base_filter_counts = counts
        if _should_log_dataset_summary():
            print(
                "[droid_lerobot_dataset] all-data camera-order fallback: "
                f"selected={counts['selected_episodes']}/{counts['input_episodes']}, "
                f"order two/single1/single2/cam2cam/default="
                f"{counts['order_from_two_cam2base']}/{counts['order_from_single_ext1_cam2base']}/"
                f"{counts['order_from_single_ext2_cam2base']}/{counts['order_from_cam2cam']}/"
                f"{counts['order_from_default']}, "
                "camera_condition=disabled, "
                f"invalid_direct_cam2base={counts['invalid_direct_cam2base']}, "
                f"windows_chunk{self._chunk_length}={counts['selected_training_windows_chunk_length']}"
            )
        if not selected:
            raise ValueError(f"All-data camera-order fallback selected 0 DROID episodes. Counts: {counts}")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def video_fps(self) -> float:
        stride = self._video_sample_stride or 1
        return self._fps / stride

    @property
    def video_num_frames(self) -> int:
        stride = self._video_sample_stride or 1
        return int(np.ceil((self._chunk_length + 1) / stride))

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
        return 8 if self._action_space == "joint_pos" else 10

    def _action_spec(self) -> ActionSpec:
        if self._action_space == "joint_pos":
            return build_action_spec(Joint(n=7, label="joint"), Gripper())
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    @property
    def action_names(self) -> list[str]:
        return self._action_spec().names

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(value).float() for key, value in load_action_stats(str(cls._stats_path())).items()
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = self.load_action_stats()
        return self._norm_stats

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _load_camera_order(self, camera_order_path: str | None) -> dict[int, tuple[str, str]]:
        if camera_order_path is None:
            return {}
        with open(camera_order_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected camera_order_path={camera_order_path!r} to contain a JSON object.")
        episodes = data.get("episodes", data)
        if not isinstance(episodes, dict):
            raise ValueError(f"Expected camera_order_path={camera_order_path!r} to contain an 'episodes' object.")
        out: dict[int, tuple[str, str]] = {}
        for key, value in episodes.items():
            if not isinstance(value, dict):
                continue
            left_key = value.get("bottom_left_video_key") or value.get("left_video_key")
            right_key = value.get("bottom_right_video_key") or value.get("right_video_key")
            if left_key not in _EXTERIOR_VIDEO_KEYS or right_key not in _EXTERIOR_VIDEO_KEYS:
                raise ValueError(
                    f"Invalid camera order for episode {key}: "
                    f"left={left_key!r}, right={right_key!r}, allowed={_EXTERIOR_VIDEO_KEYS}"
                )
            if left_key == right_key:
                raise ValueError(f"Invalid camera order for episode {key}: left and right both use {left_key!r}")
            out[int(key)] = (str(left_key), str(right_key))
        if _should_log_dataset_summary():
            print(f"[droid_lerobot_dataset] camera_order_path={camera_order_path}: loaded {len(out)} episode orders")
        return out

    def _video_path(self, episode: dict[str, Any], video_key: str) -> StoragePath:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return _storage_join(self._root, rel)

    def _video_is_file(self, path: StoragePath) -> bool:
        available_s3_video_paths = getattr(self, "_available_s3_video_paths", None)
        if available_s3_video_paths is not None:
            return str(path) in available_s3_video_paths
        return _storage_is_file(path)

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        idle_frames = compute_idle_frames(
            action,
            self._action_spec(),
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )
        if self._action_normalization is None:
            normalized_action = action
        else:
            normalized_action = normalize_action(action, self._action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self.video_fps, dtype=torch.float),
            "action_fps": torch.tensor(self._fps, dtype=torch.float),
            "condition_first_action": torch.tensor(self._use_state, dtype=torch.bool),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    def _window_rows(self, start: int, stop: int, episode_index: int) -> list[dict[str, Any]]:
        """Reconstruct the per-frame dicts the sample builder consumes for the
        half-open frame window ``[start, stop)`` from the compact column arrays.
        ``start``/``stop`` are guaranteed to lie within a single episode."""
        return [
            {
                "episode_index": episode_index,
                "task_index": int(self._row_task[j]),
                "timestamp": float(self._row_timestamp[j]),
                **{c: self._feat[c][j] for c in self._feat},
            }
            for j in range(start, stop)
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        # Map the flat sample index to a within-episode frame window.
        if getattr(self, "_use_segment_window_filter", self._use_filter_dict):
            seg = int(np.searchsorted(self._seg_cum, idx, side="right"))
            base = int(self._seg_cum[seg - 1]) if seg > 0 else 0
            ep = int(self._seg_ep_pos[seg])
            start = int(self._ep_starts[ep]) + int(self._seg_win_start[seg]) + (idx - base)
        else:
            ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
            prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
            start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        observation_rows = self._window_rows(start, start + self._chunk_length + 1, episode_index)

        video_rows = (
            observation_rows[:: self._video_sample_stride]
            if self._video_sample_stride is not None
            else observation_rows
        )
        video = self._load_concat_video(episode, video_rows)
        plucker_condition = None
        camera_metadata: dict[str, torch.Tensor] = {}
        if getattr(self, "_use_exterior_plucker_condition", False):
            if getattr(self, "_defer_plucker_generation", False):
                camera_metadata = self._build_exterior_camera_metadata(episode_index)
            else:
                plucker_condition = self._build_exterior_plucker_condition(
                    episode_index,
                    height=int(video.shape[-2]),
                    width=int(video.shape[-1]),
                )
        if self._action_space == "joint_pos":
            raw_action = self._build_joint_action(observation_rows)
            extras: dict[str, Any] = {}
        else:
            action_rows = observation_rows[: self._chunk_length]
            raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)
            extras = {"initial_pose": initial_pose}
        if plucker_condition is not None:
            extras["plucker_condition"] = plucker_condition
        extras.update(camera_metadata)
        subtask_annotations = getattr(self, "_subtask_annotations", None)
        if subtask_annotations is not None:
            action_start_frame = start - int(self._ep_starts[ep])
            extras["subtask"] = lookup_subtask(
                subtask_annotations,
                episode_index=episode_index,
                action_start_frame=action_start_frame,
                action_end_frame=action_start_frame + self._chunk_length,
                boundary_trim_frames=getattr(self, "_subtask_boundary_trim_frames", 5),
            )
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice(task.split(" | "))

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            additional_view_description=(
                "The top row is from the wrist-mounted camera. "
                "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
            ),
            **extras,
        )

    def _build_exterior_plucker_condition(self, episode_index: int, *, height: int, width: int) -> torch.Tensor:
        """Build a first-frame camera condition for the physical-left/right bottom views."""
        calibration = self._exterior_camera_calibrations.get(int(episode_index))
        if calibration is None:
            raise KeyError(f"Missing exterior camera calibration for DROID episode_index={episode_index}")
        # ``concat_view`` stacks a full-width wrist view above one row of two exterior views.
        # Each exterior view is half the wrist height, so the layout is 2/3 wrist + 1/3 exterior.
        # Current DROID example: (height, width)=(540, 640) -> top/wrist=360 px,
        # bottom/exterior=180 px, physical-left/right widths=320/320 px.
        bottom_height = height // 3  # 540 // 3 = 180 for the current raw concat video.
        left_width = width // 2
        right_width = width - left_width

        def parse(entry: dict[str, Any], *, target_width: int) -> tuple[torch.Tensor, torch.Tensor]:
            intrinsic_entry = entry["intrinsics"]
            camera_to_base = camera_xyzrpy_to_matrix(entry["extrinsics"])
            intrinsic = scaled_camera_matrix(
                intrinsic_entry["cameraMatrix"],
                source_width=int(intrinsic_entry["width"]),
                source_height=int(intrinsic_entry["height"]),
                target_width=target_width,
                target_height=bottom_height,
            )
            return camera_to_base, intrinsic

        left_pose, left_intrinsic = parse(calibration[0], target_width=left_width)
        right_pose, right_intrinsic = parse(calibration[1], target_width=right_width)
        # Shape is [6, height, width], currently [6, 540, 640]:
        #   [:, :360, :]      = 0 (wrist has no usable extrinsics),
        #   [:, 360:, :320]   = physical-left raymap [6, 180, 320],
        #   [:, 360:, 320:]   = physical-right raymap [6, 180, 320].
        return compose_exterior_plucker_condition(
            left_pose,
            left_intrinsic,
            right_pose,
            right_intrinsic,
            height=height,
            width=width,
        )

    def _build_exterior_camera_metadata(self, episode_index: int) -> dict[str, torch.Tensor]:
        """Return compact physical-left/right calibration for CUDA ray-map generation."""
        calibration = self._exterior_camera_calibrations.get(int(episode_index))
        if calibration is None:
            raise KeyError(f"Missing exterior camera calibration for DROID episode_index={episode_index}")

        camera_to_base: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        source_sizes: list[torch.Tensor] = []
        for entry in calibration:
            intrinsic_entry = entry["intrinsics"]
            source_width = int(intrinsic_entry["width"])
            source_height = int(intrinsic_entry["height"])
            camera_to_base.append(camera_xyzrpy_to_matrix(entry["extrinsics"]))
            intrinsics.append(
                scaled_camera_matrix(
                    intrinsic_entry["cameraMatrix"],
                    source_width=source_width,
                    source_height=source_height,
                    target_width=source_width,
                    target_height=source_height,
                )
            )
            source_sizes.append(torch.tensor([source_height, source_width], dtype=torch.long))

        return {
            "exterior_camera_to_base": torch.stack(camera_to_base),
            "exterior_camera_intrinsics": torch.stack(intrinsics),
            "exterior_camera_source_size": torch.stack(source_sizes),
        }

    def _build_joint_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """8D joint-position action ``[joint(7), gripper(1)]`` over the chunk, matching
        the internal ``action_space='joint_pos'``. The window is ``chunk+1`` frames:
        ``row[0]`` is the initial observed state (prepended when ``use_state``).
        By default, legacy behavior uses commanded actions from ``rows[1:]``. With
        ``use_same_timestep_joint_action=True``, commanded actions instead come from
        ``rows[:chunk_length]``, matching NVIDIA main's ``t..t+T-1`` timestamps.
        Gripper is flipped (1 - g). No normalization is applied."""
        if getattr(self, "_use_same_timestep_joint_action", False):
            action_rows = observation_rows[: self._chunk_length]
        else:
            action_rows = observation_rows[1:]
        joints = np.asarray([r[_JOINT_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 7]
        gripper = np.asarray([r[_ACTION_GRIPPER_FEATURE] for r in action_rows], dtype=np.float32).reshape(-1, 1)
        gripper = 1.0 - gripper
        action = np.concatenate([joints, gripper], axis=-1)  # [chunk, 8]
        if self._use_state:
            init = observation_rows[0]
            init_joint = np.asarray(init[_JOINT_STATE_FEATURE], dtype=np.float32)  # [7]
            init_gripper = np.asarray([1.0 - float(init[_GRIPPER_STATE_FEATURE])], dtype=np.float32)  # [1]
            initial_state = np.concatenate([init_joint, init_gripper])[None, :]  # [1, 8]
            action = np.concatenate([initial_state, action], axis=0)  # [chunk + 1, 8]
        return torch.from_numpy(action).float()

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        episode_index = int(observation_rows[0]["episode_index"])
        bottom_left_key, bottom_right_key = self._camera_order.get(
            episode_index,
            (self._image_features["left"], self._image_features["right"]),
        )

        def decode(video_key: str) -> torch.Tensor:
            path = self._video_path(episode, video_key)
            # 按 LeRobot 元数据取帧：video 的 from_timestamp 加上
            # data parquet 中 episode 内相对 timestamp。
            query_timestamps = [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps]
            return self._decode_video_frames(path, query_timestamps, video_key)

        left = decode(bottom_left_key)
        right = decode(bottom_right_key)

        wrist_key = self._image_features["wrist"]
        wrist_path = self._video_path(episode, wrist_key)
        if not self._video_is_file(wrist_path):
            raise FileNotFoundError(
                f"Missing wrist video for DROID concat_view episode_index={episode_index}: {wrist_path}"
            )
        wrist = decode(wrist_key)
        if self._image_features["left"] == _COSMOS3_DROID_IMAGE_FEATURES["left"]:
            wrist_h, wrist_w = wrist.shape[-2:]
            side_h, side_w = wrist_h // 2, wrist_w // 2
        else:
            side_h, side_w = left.shape[-2:]
            wrist_h, wrist_w = side_h * 2, side_w * 2
        if wrist.shape[-2:] != (wrist_h, wrist_w):
            wrist = F.interpolate(wrist, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
        if self._wrist_rotate_180:
            wrist = torch.flip(wrist, dims=(-2, -1))

        if self._use_image_augmentation:
            # Random crop+rescale (spatial jitter) + color jitter, BEFORE the concat.
            # All three views are stacked so one sampled set of params is applied
            # uniformly across every frame and view (temporally + cross-view consistent),
            # while each __getitem__ resamples. Matches the internal DROID recipe.
            if self._image_augmentor is None:
                self._image_augmentor = T.Compose(
                    [
                        # 只保留颜色增强；先关掉随机 crop/resize，避免改变图像几何。
                        # T.RandomCrop((int(wrist_h * 0.95), int(wrist_w * 0.95))),
                        # T.Resize((wrist_h, wrist_w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            left = F.interpolate(left, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
            right = F.interpolate(right, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
            n, m = wrist.shape[0], wrist.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([wrist, left, right], dim=0))
            wrist, left, right = combined[:n], combined[n:m], combined[m:]

        left = F.interpolate(left, size=(side_h, side_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(side_h, side_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist, bottom], dim=-2)

    def _load_separate_reasoner_views(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
        *,
        long_edge: int | None = None,
    ) -> torch.Tensor:
        """Decode Reasoner views as separate images in LEFT/RIGHT/WRIST order.

        ``concat_view`` is useful for the action/VAE path, but it shrinks each
        exterior camera to half height when it is composed with the wrist view.
        The Reasoner can instead receive the three camera views independently,
        preserving the exterior camera resolution.  The returned tensor is
        ``[N * 3, C, H, W]`` with the view order repeated for every frame:
        ``LEFT, RIGHT, WRIST, LEFT, RIGHT, WRIST, ...``.
        """
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        episode_index = int(observation_rows[0]["episode_index"])
        bottom_left_key, bottom_right_key = self._camera_order.get(
            episode_index,
            (self._image_features["left"], self._image_features["right"]),
        )

        def decode(video_key: str) -> torch.Tensor:
            path = self._video_path(episode, video_key)
            query_timestamps = [
                float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps
            ]
            return self._decode_video_frames(path, query_timestamps, video_key)

        left = decode(bottom_left_key)
        right = decode(bottom_right_key)
        wrist_key = self._image_features["wrist"]
        wrist_path = self._video_path(episode, wrist_key)
        if not self._video_is_file(wrist_path):
            raise FileNotFoundError(
                f"Missing wrist video for DROID separate-view Reasoner episode_index={episode_index}: {wrist_path}"
            )
        wrist = decode(wrist_key)
        if self._wrist_rotate_180:
            wrist = torch.flip(wrist, dims=(-2, -1))

        views = [left, right, wrist]
        if long_edge is not None:
            edge = int(long_edge)
            if edge < 1:
                raise ValueError(f"reasoner_image_long_edge must be positive, got {long_edge!r}")
            resized: list[torch.Tensor] = []
            for view in views:
                height, width = (int(view.shape[-2]), int(view.shape[-1]))
                scale = edge / max(height, width)
                target_h = max(1, int(round(height * scale)))
                target_w = max(1, int(round(width * scale)))
                if (target_h, target_w) != (height, width):
                    view = F.interpolate(view, size=(target_h, target_w), mode="bilinear", align_corners=False)
                resized.append(view)
            views = resized

        # Interleave camera views within each timepoint rather than grouping
        # all LEFT frames followed by all RIGHT/WRIST frames.
        frames: list[torch.Tensor] = []
        for frame_index in range(len(observation_rows)):
            frames.extend(view[frame_index : frame_index + 1] for view in views)
        return torch.cat(frames, dim=0)

    def _decode_video_frames(self, path: StoragePath, query_timestamps: list[float], video_key: str) -> torch.Tensor:
        if self._video_decode_backend == "imageio":
            if _is_s3_path(path):
                raise ValueError("video_decode_backend='imageio' does not support S3; use 'torchcodec'.")
            try:
                return self._decode_video_frames_imageio(path, query_timestamps, video_key)
            except Exception:
                if not self._imageio_fallback_warned and _should_log_dataset_summary():
                    print(
                        "[droid_lerobot_dataset] imageio video decode failed; "
                        "falling back to lerobot decode_video_frames for this worker."
                    )
                    self._imageio_fallback_warned = True
                return self._decode_video_frames_lerobot(path, query_timestamps, backend=None)

        backend = None if self._video_decode_backend == "lerobot" else self._video_decode_backend
        return self._decode_video_frames_lerobot(path, query_timestamps, backend=backend)

    def _decode_video_frames_lerobot(
        self,
        path: StoragePath,
        query_timestamps: list[float],
        backend: str | None,
    ) -> torch.Tensor:
        if backend == "torchcodec":
            # Keep recently reused episode videos, but never allow LeRobot's
            # default unbounded per-path cache to accumulate across training.
            _install_bounded_torchcodec_decoder_cache()
        return decode_video_frames(path, query_timestamps, self._tolerance_s, backend=backend)

    def _decode_video_frames_imageio(
        self,
        path: StoragePath,
        query_timestamps: list[float],
        video_key: str,
    ) -> torch.Tensor:
        fps = self._video_fps(video_key)
        frame_indices = [int(round(ts * fps)) for ts in query_timestamps]
        reader = imageio.get_reader(str(path))
        try:
            frames = [
                torch.from_numpy(np.asarray(reader.get_data(frame_idx), dtype=np.uint8))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                for frame_idx in frame_indices
            ]
        finally:
            reader.close()
        return torch.stack(frames, dim=0)

    def _video_fps(self, video_key: str) -> float:
        feature = self._info.get("features", {}).get(video_key, {})
        info = feature.get("info", {}) if isinstance(feature, dict) else {}
        fps = info.get("video.fps", self._fps) if isinstance(info, dict) else self._fps
        return float(fps)

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DROID_TO_OPENCV

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        gripper = np.asarray([row[_ACTION_GRIPPER_FEATURE] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        gripper = 1.0 - gripper
        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose

    def __len__(self) -> int:
        if getattr(self, "_use_segment_window_filter", self._use_filter_dict):
            return int(self._seg_cum[-1]) if self._seg_cum.size else 0
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode (or per kept-segment, when ``use_filter_dict``) flat-index blocks
        ``(start, length)``. ``ActionIterableShuffleDataset`` shuffles the ORDER of these
        blocks and shards them disjointly across ranks, while keeping windows *within* a
        block sequential -> decorrelates batches across ranks without random-access I/O
        (preserves locality + copy-on-write memory sharing across workers)."""
        cum = self._seg_cum if getattr(self, "_use_segment_window_filter", self._use_filter_dict) else self._valid_cum
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks
