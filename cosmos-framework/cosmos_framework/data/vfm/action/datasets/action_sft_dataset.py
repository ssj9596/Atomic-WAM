# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: ``DROIDLeRobotDataset`` → ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.vfm.action.datasets.robotwin_lerobot_dataset import RoboTwinLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)

    def get_shuffle_blocks(self):
        """Delegate to the inner DROIDLeRobotDataset (per-episode/segment flat-index blocks)."""
        return self._dataset.get_shuffle_blocks()


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view of a map-style ``ActionSFTDataset``.

    Each ``(rank, worker)`` is assigned a DISJOINT subset of episodes (sharded over
    ``shard_world_size * num_workers``), shuffles its episode ORDER, and streams the
    windows WITHIN each episode sequentially -> within-rank batch diversity (the N
    workers of a rank stream N different episodes) AND cross-rank diversity, while
    keeping reads sequential (I/O locality + COW; no RandomSampler random-access OOM).
    Re-shuffles each epoch and streams indefinitely (the trainer stops at ``max_iter``).

    ``shard_world_size`` / ``shard_rank`` are set by ``RankPartitionedDataLoader``.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42, resume_name: str = ""):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self._resume_name = str(resume_name).strip()
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        import torch

        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        prefix = f"COSMOS_DL_STATE_{self._resume_name}_" if self._resume_name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{wid}_EPOCH", 0)) if self._resume_name else 0
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{wid}_INDEX", -1)) if self._resume_name else -1
        epoch = resume_epoch
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)  # same permutation across all (rank,worker) -> disjoint shard
            order = torch.randperm(len(blocks), generator=g).tolist()
            stream_pos = 0
            for b in order[global_shard::total_shards]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    if epoch == resume_epoch and stream_pos <= resume_pos:
                        stream_pos += 1
                        continue
                    item = self._dataset[idx]
                    if self._resume_name and isinstance(item, dict):
                        yield {"_dp_epoch": epoch, "_dp_stream_pos": stream_pos, **item}
                    else:
                        yield item
                    stream_pos += 1
            epoch += 1


def _validate_non_navigation_oversample_factor(value: float) -> float:
    """Validate the opt-in non-navigation repetition factor.

    A factor of one is deliberately a no-op.  Values below one would
    downsample navigation/non-navigation buckets and would make it impossible
    to guarantee that every baseline window remains in the stream, so they are
    rejected just like the existing primary-grip sampler.
    """

    if isinstance(value, bool):
        raise TypeError(f"non_navigation_oversample_factor must be a number >= 1.0, got {value!r}.")
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"non_navigation_oversample_factor must be a number >= 1.0, got {value!r}.") from exc
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(
            f"non_navigation_oversample_factor must be finite and >= 1.0; got {value!r}. Downsampling is not supported."
        )
    return factor


def _validate_navigation_keep_ratio(value: float) -> float:
    """Validate the optional fixed-length navigation retention ratio."""

    if isinstance(value, bool):
        raise TypeError(f"navigation_keep_ratio must be a number in [0, 1], got {value!r}.")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"navigation_keep_ratio must be a number in [0, 1], got {value!r}.") from exc
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"navigation_keep_ratio must be finite and in [0, 1], got {value!r}.")
    return ratio


def _normalized_primary_subgoal(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip().rstrip(".!?。！？").rstrip().casefold()


_NON_NAVIGATION_EXCLUDED_PREFIXES = ("idle", "remain stationary", "remain idle")
_IDLE_WORDS = frozenset(("idle", "idling", "idled"))


def _is_non_navigation_sampling_excluded_label(text: str) -> bool:
    return bool(_IDLE_WORDS.intersection(text.split())) or any(
        text == prefix or text.startswith(f"{prefix} ") for prefix in _NON_NAVIGATION_EXCLUDED_PREFIXES
    )


def _is_available_non_navigation_subgoal(value: Any) -> bool:
    """Return whether an available primary label is outside move/approach.

    Empty labels represent an annotation gap, a missing episode, or a
    six-rule-filtered window.  They must stay at their baseline frequency and
    therefore are *not* considered non-navigation samples. Idle/stationary
    labels are excluded defensively; this sampler-side rule is intentionally
    at least as strict as the six-rule label filter. Any other non-empty label
    whose first action is not ``move``/``approach`` belongs to the oversampled
    bucket.
    """

    text = _normalized_primary_subgoal(value)
    return (
        bool(text)
        and not _is_non_navigation_sampling_excluded_label(text)
        and not text.startswith(("move", "approach"))
    )


def _primary_subgoal_bucket(value: Any) -> str:
    """Classify a primary label without conflating unavailable text.

    ``unavailable`` covers empty/missing/filter-fallback windows and explicit
    idle/stationary labels excluded from repetition; only the other two
    buckets participate in balanced replacement accounting.
    """

    text = _normalized_primary_subgoal(value)
    if not text or _is_non_navigation_sampling_excluded_label(text):
        return "unavailable"
    return "navigation" if text.startswith(("move", "approach")) else "non_navigation"


class NonNavigationOversampleIterableDataset(IterableDataset):
    """Repeat available non-navigation subgoal windows deterministically.

    The source order, episode permutation, worker/rank sharding, and resume
    cursor are identical to :class:`PrimaryGripOversampleIterableDataset`.
    Only windows whose *available* executable primary subgoal does not start
    with ``move``/``approach`` receive extra copies. Thus ``grip`` and ``lift``
    can be emphasized, while navigation labels, idle/stationary labels, and
    windows falling back to the task instruction remain at factor one.

    This class lives beside the ordinary action iterable rather than inside
    the joint Reasoner/WAM module so the standalone WAM launcher does not need
    to import or construct a Reasoner stream.  ``emit_resume_metadata=False``
    is used by the standalone factory to preserve the ordinary WAM sample
    schema; direct users/tests keep the metadata default for cursor inspection.
    """

    def __init__(
        self,
        dataset: "ActionSFTDataset",
        *,
        seed: int = 42,
        resume_name: str = "wam_non_navigation",
        non_navigation_oversample_factor: float = 1.0,
        non_navigation_oversample_seed: int | None = None,
        emit_resume_metadata: bool = True,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self._resume_name = str(resume_name).strip()
        if not isinstance(emit_resume_metadata, bool):
            raise TypeError(f"emit_resume_metadata must be a bool, got {emit_resume_metadata!r}.")
        self._emit_resume_metadata = emit_resume_metadata
        self._oversample_factor = _validate_non_navigation_oversample_factor(non_navigation_oversample_factor)
        raw_seed = self._seed if non_navigation_oversample_seed is None else non_navigation_oversample_seed
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError(f"non_navigation_oversample_seed must be an integer, got {raw_seed!r}.")
        self._oversample_seed = int(raw_seed)
        extra = self._oversample_factor - 1.0
        self._guaranteed_extra = int(math.floor(extra))
        self._fractional_extra = float(extra - self._guaranteed_extra)
        self.shard_world_size = 1
        self.shard_rank = 0

        raw_dataset = getattr(dataset, "_dataset", None)
        self._primary_subgoal = getattr(raw_dataset, "primary_subgoal", None)
        self._is_primary_navigation_window = getattr(raw_dataset, "is_primary_navigation_window", None)
        if (
            self._oversample_factor > 1.0
            and not callable(self._primary_subgoal)
            and not callable(self._is_primary_navigation_window)
        ):
            raise TypeError(
                "NonNavigationOversampleIterableDataset requires the wrapped raw DROID dataset "
                "to expose primary_subgoal(index) or is_primary_navigation_window(index)."
            )

        # Compact metadata cache used by both the ordinary and balanced
        # streams. Planning and emission should not repeat annotation interval
        # lookups for every raw window.  0xff=unknown, 0=unavailable,
        # 1=navigation, 2=non-navigation.
        self._bucket_cache = bytearray([0xFF]) * len(dataset)

    def __len__(self) -> int:
        """Return the unweighted baseline length; iteration is intentionally infinite."""

        return len(self._dataset)

    def _fractional_draw(self, *, epoch: int, raw_index: int) -> float:
        payload = f"{self._oversample_seed}:{int(epoch)}:{int(raw_index)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    def _bucket_for_raw_index(self, raw_index: int) -> str:
        """Resolve and cache the three-way metadata-only label bucket."""

        raw_index = int(raw_index)
        cached = self._bucket_cache[raw_index]
        if cached == 0:
            return "unavailable"
        if cached == 1:
            return "navigation"
        if cached == 2:
            return "non_navigation"

        if callable(self._primary_subgoal):
            bucket = _primary_subgoal_bucket(self._primary_subgoal(raw_index))
        elif callable(self._is_primary_navigation_window):
            # Real DROID subgoal datasets expose primary_subgoal, which lets us
            # distinguish an empty fallback label. Keep this predicate-only
            # fallback for lightweight test/dummy datasets.
            bucket = "navigation" if bool(self._is_primary_navigation_window(raw_index)) else "non_navigation"
        else:
            bucket = "unavailable"
        self._bucket_cache[raw_index] = {"unavailable": 0, "navigation": 1, "non_navigation": 2}[bucket]
        return bucket

    def _is_non_navigation_window(self, raw_index: int) -> bool:
        return self._bucket_for_raw_index(raw_index) == "non_navigation"

    def _extra_copy_count(self, *, epoch: int, raw_index: int, bucket: str | None = None) -> int:
        if self._oversample_factor <= 1.0:
            return 0
        if bucket is None:
            bucket = self._bucket_for_raw_index(raw_index)
        if bucket != "non_navigation":
            return 0
        count = self._guaranteed_extra
        if (
            self._fractional_extra > 0.0
            and self._fractional_draw(epoch=epoch, raw_index=raw_index) < self._fractional_extra
        ):
            count += 1
        return count

    def _stream_coordinates(self) -> tuple[int, int, int, int]:
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        global_shard = int(self.shard_rank) * num_workers + worker_id
        total_shards = max(1, int(self.shard_world_size) * num_workers)
        return worker_id, num_workers, global_shard, total_shards

    def __iter__(self):
        blocks = self._dataset.get_shuffle_blocks()
        worker_id, _, global_shard, total_shards = self._stream_coordinates()
        prefix = f"COSMOS_DL_STATE_{self._resume_name}_" if self._resume_name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0)) if self._resume_name else 0
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1)) if self._resume_name else -1
        epoch = resume_epoch
        while True:
            import torch

            generator = torch.Generator().manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=generator).tolist()
            stream_pos = 0
            for block_index in order[global_shard::total_shards]:
                start, length = blocks[block_index]
                for raw_index in range(int(start), int(start) + int(length)):
                    bucket = self._bucket_for_raw_index(raw_index)
                    emit_count = 1 + self._extra_copy_count(
                        epoch=epoch,
                        raw_index=raw_index,
                        bucket=bucket,
                    )
                    skipped = 0
                    if epoch == resume_epoch and stream_pos <= resume_pos:
                        skipped = min(emit_count, resume_pos - stream_pos + 1)
                        stream_pos += skipped
                    if skipped == emit_count:
                        continue

                    item = self._dataset[raw_index]
                    for _ in range(skipped, emit_count):
                        if self._emit_resume_metadata:
                            yield {
                                "_dp_epoch": epoch,
                                "_dp_stream_pos": stream_pos,
                                **item,
                            }
                        else:
                            yield dict(item)
                        stream_pos += 1
            epoch += 1


class BalancedNonNavigationReplacementIterableDataset(NonNavigationOversampleIterableDataset):
    """Fixed-length non-navigation repetition with navigation replacement.

    When enabled, a metadata-only planning pass over this rank/worker's
    shuffled episode blocks counts navigation windows and the deterministic
    non-navigation extra copies requested by the factor.  The emission pass
    caps extras at ``floor((1-navigation_keep_ratio) * navigation_count)`` and
    skips exactly that many navigation windows.  Consequently every local
    epoch emits the same number of samples as the unweighted source stream and
    ``extra_non_navigation == skipped_navigation``.  If the factor requests
    fewer extras than the ratio quota, the effective keep ratio is higher than
    requested; if it requests more, the quota caps the extras and keeps at
    least the configured navigation fraction.

    The planning pass only calls the raw subgoal metadata helper, never the
    transformed dataset, so it does not decode video.  Re-running that pass on
    resume reconstructs the same epoch plan before applying the saved emitted
    cursor.  Each rank/worker balances its disjoint shard independently; the
    sum across ranks remains fixed-length, but no cross-rank synchronization or
    global pre-scan is performed.
    """

    def __init__(
        self,
        dataset: "ActionSFTDataset",
        *,
        seed: int = 42,
        resume_name: str = "wam_non_navigation_balanced",
        non_navigation_oversample_factor: float = 1.0,
        non_navigation_oversample_seed: int | None = None,
        navigation_keep_ratio: float = 1.0,
        emit_resume_metadata: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            resume_name=resume_name,
            non_navigation_oversample_factor=non_navigation_oversample_factor,
            non_navigation_oversample_seed=non_navigation_oversample_seed,
            emit_resume_metadata=emit_resume_metadata,
        )
        self._navigation_keep_ratio = _validate_navigation_keep_ratio(navigation_keep_ratio)

    def _epoch_plan(
        self,
        *,
        epoch: int,
        block_order: list[int],
        blocks: list[tuple[int, int]],
        global_shard: int,
        total_shards: int,
    ) -> tuple[list[int], int, int, int]:
        """Return shard blocks, navigation skip quota, and metadata counts."""

        shard_blocks = block_order[global_shard::total_shards]
        navigation_count = 0
        requested_extra_count = 0
        for block_index in shard_blocks:
            start, length = blocks[block_index]
            for raw_index in range(int(start), int(start) + int(length)):
                bucket = self._bucket_for_raw_index(raw_index)
                if bucket == "non_navigation":
                    requested_extra_count += self._extra_copy_count(
                        epoch=epoch,
                        raw_index=raw_index,
                        bucket=bucket,
                    )
                elif bucket == "navigation":
                    navigation_count += 1
        ratio_quota = int(math.floor((1.0 - self._navigation_keep_ratio) * navigation_count))
        skip_quota = min(requested_extra_count, max(0, ratio_quota))
        return shard_blocks, skip_quota, navigation_count, requested_extra_count

    def _candidate_draw(self, *, epoch: int, raw_index: int, candidate_index: int, bucket: str) -> float:
        payload = f"{self._oversample_seed}:{epoch}:{raw_index}:{candidate_index}:{bucket}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    def __iter__(self):
        blocks = self._dataset.get_shuffle_blocks()
        worker_id, _, global_shard, total_shards = self._stream_coordinates()
        prefix = f"COSMOS_DL_STATE_{self._resume_name}_" if self._resume_name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0)) if self._resume_name else 0
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1)) if self._resume_name else -1
        epoch = resume_epoch
        while True:
            import torch

            generator = torch.Generator().manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=generator).tolist()
            shard_blocks, skip_quota, navigation_count, requested_extra_count = self._epoch_plan(
                epoch=epoch,
                block_order=order,
                blocks=blocks,
                global_shard=global_shard,
                total_shards=total_shards,
            )
            remaining_navigation_skips = skip_quota
            remaining_extra_budget = skip_quota
            # These candidate counts are known from the metadata planning pass;
            # dynamic reservoir probabilities below select exactly the quota
            # without materializing a potentially huge index set.
            remaining_navigation_candidates = navigation_count
            remaining_extra_candidates = requested_extra_count
            stream_pos = 0
            for block_index in shard_blocks:
                start, length = blocks[block_index]
                for raw_index in range(int(start), int(start) + int(length)):
                    bucket = self._bucket_for_raw_index(raw_index)
                    if bucket == "non_navigation":
                        requested_extra = self._extra_copy_count(
                            epoch=epoch,
                            raw_index=raw_index,
                            bucket=bucket,
                        )
                        extra_count = 0
                        for candidate_index in range(requested_extra):
                            if remaining_extra_candidates <= 0:
                                break
                            choose = remaining_extra_budget >= remaining_extra_candidates or (
                                remaining_extra_budget > 0
                                and self._candidate_draw(
                                    epoch=epoch,
                                    raw_index=raw_index,
                                    candidate_index=candidate_index,
                                    bucket="extra",
                                )
                                < remaining_extra_budget / remaining_extra_candidates
                            )
                            remaining_extra_candidates -= 1
                            if choose:
                                extra_count += 1
                                remaining_extra_budget -= 1
                        emit_count = 1 + extra_count
                    elif bucket == "navigation":
                        choose_skip = remaining_navigation_skips >= remaining_navigation_candidates or (
                            remaining_navigation_skips > 0
                            and self._candidate_draw(
                                epoch=epoch,
                                raw_index=raw_index,
                                candidate_index=0,
                                bucket="navigation",
                            )
                            < remaining_navigation_skips / remaining_navigation_candidates
                        )
                        remaining_navigation_candidates -= 1
                        if choose_skip:
                            remaining_navigation_skips -= 1
                            continue
                        emit_count = 1
                    else:
                        # Unavailable/empty labels are ordinary instruction
                        # fallbacks and are never counted as navigation or
                        # non-navigation replacement candidates.
                        emit_count = 1

                    skipped = 0
                    if epoch == resume_epoch and stream_pos <= resume_pos:
                        skipped = min(emit_count, resume_pos - stream_pos + 1)
                        stream_pos += skipped
                    if skipped == emit_count:
                        continue

                    item = self._dataset[raw_index]
                    for _ in range(skipped, emit_count):
                        if self._emit_resume_metadata:
                            yield {
                                "_dp_epoch": epoch,
                                "_dp_stream_pos": stream_pos,
                                **item,
                            }
                        else:
                            yield dict(item)
                        stream_pos += 1
            epoch += 1


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "policy",
    use_state: bool = True,
    use_same_timestep_joint_action: bool = False,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    prefilter_by_filter_dict: bool = False,
    camera_order_path: str | None = None,
    cam2base_extrinsic_path: str | None = None,
    success_only: bool = False,
    video_decode_backend: str = "torchcodec",
    wrist_rotate_180: bool = False,
    video_sample_stride: int | None = None,
    use_exterior_plucker_condition: bool = False,
    defer_plucker_generation: bool = False,
    use_all_data_with_cam2cam_fallback: bool = False,
    subtask_annotations_path: str | None = None,
    precise_subtask_annotations_path: str | None = None,
    subtask_episode_filter_path: str | None = None,
    use_subgoal_dataset: bool = False,
    subgoal_training_mode: str | None = None,
    condition_sources: Sequence[str] | None = None,
    condition_weights: Sequence[float] | None = None,
    condition_seed: int = 0,
    lap_control_include_translation: bool = True,
    lap_control_include_rotation: bool = True,
    lap_control_rotation_precision: int = 10,
    lap_control_min_translation_cm: float = 2.0,
    lap_control_min_rotation_degrees: float = 10.0,
    lap_force_condition_rotation_degrees: float | None = None,
    rotation_aware_subgoal_degrees: float | None = None,
    subtask_label_source: str = "effective",
    max_subtasks_per_window: int = 100,
    max_subtasks_count_segments: bool = False,
    subtask_boundary_trim_frames: int = 5,
    pregrasp_boundary_trim_frames: int = 0,
    contact_label_delay_frames: int = 0,
    object_position_augment_prob: float = 0.0,
    object_position_augment_style: str = "random",
    object_position_augment_seed: int = 42,
    include_idle_stationary_subtasks: bool = False,
    apply_subtask_episode_filter_to_windows: bool = True,
    remove_post_completion_windows: bool = False,
    include_post_completion_move_away_windows: bool = False,
    include_execution_success: bool = False,
    include_task_instruction_context: bool = False,
    execution_success_only: bool = False,
    primary_grip_oversample_factor: float = 1.0,
    primary_grip_oversample_seed: int | None = None,
    non_navigation_oversample_factor: float = 1.0,
    non_navigation_oversample_seed: int | None = None,
    navigation_keep_ratio: float = 1.0,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    camera_cfg_dropout_rate: float = 0.0,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    include_vit_first_frame: bool = False,
    vit_processor_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    vit_processor_config_variant: str = "hf",
    vit_processor_cache_dir: str | None = None,
    vit_first_frame_max_pixels: int | None = None,
    vit_first_frame_min_pixels: int | None = None,
    defer_vit_pixel_values: bool = False,
    vit_frame_offsets: Sequence[int] = (0,),
    vit_frame_offset_choices: Sequence[Sequence[int]] | None = None,
    vit_frame_choice_seed: int = 0,
    subgoal_task_variant_policy: str = "first",
    mid_soft: bool = False,
    tokenize_subtask: bool = False,
    subtask_max_length: int = 128,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the DROID action SFT dataset: ``action_space='joint_pos'`` (8D) +
    ``use_state`` (raw/un-normalized), concat_view, chunk_length 32."""
    # ``subgoal_training_mode`` is an explicit opt-in convenience flag.  It
    # selects the isolated subgoal dataset even when callers only update a
    # single LazyCall argument (the legacy ``use_subgoal_dataset`` default is
    # still False).  The standalone ``wam`` mode never constructs Reasoner
    # records; the joint ``reasoner_wam`` factories remain in
    # ``droid_reasoner_wam_streams.py``.
    if subgoal_training_mode is not None:
        use_subgoal_dataset = True

    if include_task_instruction_context and not format_prompt_as_json:
        raise ValueError(
            "include_task_instruction_context=True requires format_prompt_as_json=True."
        )

    if use_subgoal_dataset:
        from cosmos_framework.data.vfm.action.datasets.droid_lerobot_subgoal_dataset import (
            DROIDLeRobotSubgoalDataset as DatasetClass,
        )
    else:
        from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset as DatasetClass

    dataset_kwargs: dict[str, Any] = dict(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        use_same_timestep_joint_action=use_same_timestep_joint_action,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        prefilter_by_filter_dict=prefilter_by_filter_dict,
        camera_order_path=camera_order_path,
        cam2base_extrinsic_path=cam2base_extrinsic_path,
        success_only=success_only,
        video_decode_backend=video_decode_backend,
        wrist_rotate_180=wrist_rotate_180,
        video_sample_stride=video_sample_stride,
        use_exterior_plucker_condition=use_exterior_plucker_condition,
        defer_plucker_generation=defer_plucker_generation,
        use_all_data_with_cam2cam_fallback=use_all_data_with_cam2cam_fallback,
        subtask_annotations_path=subtask_annotations_path,
        subtask_episode_filter_path=subtask_episode_filter_path,
        include_idle_stationary_subtasks=include_idle_stationary_subtasks,
        apply_subtask_episode_filter_to_windows=apply_subtask_episode_filter_to_windows,
        remove_post_completion_windows=remove_post_completion_windows,
        include_post_completion_move_away_windows=include_post_completion_move_away_windows,
    )
    if use_subgoal_dataset:
        resolved_training_mode = "mid_soft" if subgoal_training_mode is None else str(subgoal_training_mode)
        if resolved_training_mode not in ("mid_soft", "wam"):
            raise ValueError(
                "get_action_droid_sft_dataset supports subgoal_training_mode='mid_soft' or 'wam'; "
                "use get_droid_reasoner_stream_dataset/get_droid_wam_stream_dataset for the joint streams. "
                f"Got {resolved_training_mode!r}."
            )
        if include_task_instruction_context and resolved_training_mode != "wam":
            raise ValueError(
                "include_task_instruction_context=True is only supported for subgoal_training_mode='wam'."
            )
        dataset_kwargs["training_mode"] = resolved_training_mode
        dataset_kwargs["precise_subtask_annotations_path"] = precise_subtask_annotations_path
        if condition_sources is not None:
            dataset_kwargs["condition_sources"] = condition_sources
        if condition_weights is not None:
            dataset_kwargs["condition_weights"] = condition_weights
        dataset_kwargs["condition_seed"] = int(condition_seed)
        dataset_kwargs["lap_control_include_translation"] = bool(lap_control_include_translation)
        dataset_kwargs["lap_control_include_rotation"] = bool(lap_control_include_rotation)
        dataset_kwargs["lap_control_rotation_precision"] = int(lap_control_rotation_precision)
        dataset_kwargs["lap_control_min_translation_cm"] = float(lap_control_min_translation_cm)
        dataset_kwargs["lap_control_min_rotation_degrees"] = float(lap_control_min_rotation_degrees)
        dataset_kwargs["lap_force_condition_rotation_degrees"] = lap_force_condition_rotation_degrees
        dataset_kwargs["rotation_aware_subgoal_degrees"] = rotation_aware_subgoal_degrees
        dataset_kwargs["subtask_label_source"] = str(subtask_label_source)
        dataset_kwargs["max_subtasks_per_window"] = int(max_subtasks_per_window)
        dataset_kwargs["max_subtasks_count_segments"] = bool(max_subtasks_count_segments)
        dataset_kwargs["subtask_boundary_trim_frames"] = int(subtask_boundary_trim_frames)
        dataset_kwargs["pregrasp_boundary_trim_frames"] = int(pregrasp_boundary_trim_frames)
        dataset_kwargs["contact_label_delay_frames"] = int(contact_label_delay_frames)
        dataset_kwargs["include_execution_success"] = bool(include_execution_success)
        dataset_kwargs["include_task_instruction_context"] = bool(include_task_instruction_context)
        dataset_kwargs["execution_success_only"] = bool(execution_success_only)
        dataset_kwargs["object_position_augment_prob"] = object_position_augment_prob
        dataset_kwargs["object_position_augment_style"] = object_position_augment_style
        dataset_kwargs["object_position_augment_seed"] = object_position_augment_seed
        dataset_kwargs["vit_frame_offsets"] = vit_frame_offsets
        dataset_kwargs["vit_frame_offset_choices"] = vit_frame_offset_choices
        dataset_kwargs["vit_frame_choice_seed"] = vit_frame_choice_seed
        dataset_kwargs["task_variant_policy"] = subgoal_task_variant_policy
    dataset = DatasetClass(**dataset_kwargs)
    pad_keys = ["video"]
    if use_exterior_plucker_condition:
        pad_keys.append("plucker_condition")
    if use_subgoal_dataset and (tuple(vit_frame_offsets) != (0,) or vit_frame_offset_choices is not None):
        pad_keys.append("vit_images")
    transform = ActionTransformPipeline(
        pad_keys=pad_keys,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        camera_cfg_dropout_rate=camera_cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
        include_vit_first_frame=include_vit_first_frame,
        vit_processor_model_name=vit_processor_model_name,
        vit_processor_config_variant=vit_processor_config_variant,
        vit_processor_cache_dir=vit_processor_cache_dir,
        vit_first_frame_max_pixels=vit_first_frame_max_pixels,
        vit_first_frame_min_pixels=vit_first_frame_min_pixels,
        defer_vit_pixel_values=defer_vit_pixel_values,
        vit_frame_offsets=vit_frame_offsets,
        mid_soft=mid_soft,
        tokenize_subtask=tokenize_subtask,
        subtask_max_length=subtask_max_length,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)

    # The primary-grip repetition stream is deliberately imported lazily: the
    # regular action dataset (and all historical settings) must not import the
    # joint Reasoner/WAM module or pay for its extra sampler.  Factor 1.0 keeps
    # the exact legacy iterable path below.
    if use_subgoal_dataset and dataset_kwargs.get("training_mode") == "wam":
        if isinstance(primary_grip_oversample_factor, bool):
            raise TypeError(
                f"primary_grip_oversample_factor must be a number >= 1.0, got {primary_grip_oversample_factor!r}."
            )
        try:
            oversample_factor = float(primary_grip_oversample_factor)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"primary_grip_oversample_factor must be a number >= 1.0, got {primary_grip_oversample_factor!r}."
            ) from exc
        if not math.isfinite(oversample_factor) or oversample_factor < 1.0:
            raise ValueError(
                f"primary_grip_oversample_factor must be finite and >= 1.0; got {primary_grip_oversample_factor!r}."
            )
        non_navigation_factor = _validate_non_navigation_oversample_factor(non_navigation_oversample_factor)
        navigation_ratio = _validate_navigation_keep_ratio(navigation_keep_ratio)
        if oversample_factor > 1.0 and non_navigation_factor > 1.0:
            raise ValueError(
                "primary_grip_oversample_factor and non_navigation_oversample_factor cannot both exceed 1.0; "
                "choose one repetition strategy to avoid double-counting grip/non-navigation windows."
            )
        if navigation_ratio < 1.0 and oversample_factor > 1.0:
            raise ValueError(
                "navigation_keep_ratio<1 cannot be combined with primary_grip_oversample_factor>1; "
                "set the grip factor to 1.0 when enabling balanced non-navigation replacement."
            )
        if navigation_ratio < 1.0 and non_navigation_factor > 1.0:
            if not iterable_shuffle:
                raise ValueError("navigation_keep_ratio<1 requires iterable_shuffle=True.")
            return BalancedNonNavigationReplacementIterableDataset(
                sft,
                seed=episode_shuffle_seed,
                resume_name="",
                non_navigation_oversample_factor=non_navigation_factor,
                non_navigation_oversample_seed=non_navigation_oversample_seed,
                navigation_keep_ratio=navigation_ratio,
                emit_resume_metadata=False,
            )
        if non_navigation_factor != 1.0:
            if not iterable_shuffle:
                raise ValueError("non_navigation_oversample_factor>1 requires iterable_shuffle=True.")
            return NonNavigationOversampleIterableDataset(
                sft,
                seed=episode_shuffle_seed,
                resume_name="",
                non_navigation_oversample_factor=non_navigation_factor,
                non_navigation_oversample_seed=non_navigation_oversample_seed,
                emit_resume_metadata=False,
            )
        if oversample_factor != 1.0:
            if not iterable_shuffle:
                raise ValueError("primary_grip_oversample_factor>1 requires iterable_shuffle=True.")
            from cosmos_framework.data.vfm.action.datasets.droid_reasoner_wam_streams import (
                PrimaryGripOversampleIterableDataset,
            )

            return PrimaryGripOversampleIterableDataset(
                sft,
                seed=episode_shuffle_seed,
                # The ordinary single-stream launcher has no paired-loader
                # cursor; keep internal _dp_* metadata out of its legacy
                # collator/model batch.  Reasoner-WAM uses its own factory with
                # resume_name="wam" and retains checkpoint cursors there.
                resume_name="",
                primary_grip_oversample_factor=oversample_factor,
                primary_grip_oversample_seed=primary_grip_oversample_seed,
                emit_resume_metadata=False,
            )
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_robotwin_sft_dataset(
    *,
    root: str,
    fps: float | None = None,
    chunk_length: int = 32,
    video_fps: float | None = None,
    video_num_frames: int | None = None,
    video_sample_stride: int | None = None,
    mode: str = "policy",
    viewpoint: str = "concat_view",
    split_name: str = "aloha-agilex_clean_50",
    task_names: list[str] | None = None,
    action_space: str | None = None,
    use_state: bool = True,
    action_normalization: str | None = None,
    use_image_augmentation: bool = False,
    sample_stride: int = 1,
    domain_name: str = "robotwin",
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    success_only: bool = False,
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    include_vit_first_frame: bool = False,
    vit_processor_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    vit_processor_config_variant: str = "hf",
    vit_processor_cache_dir: str | None = None,
    vit_first_frame_max_pixels: int | None = None,
    vit_first_frame_min_pixels: int | None = None,
    mid_soft: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the RoboTwin action SFT dataset.

    RoboTwin actions are raw qpos14.  The default ``use_state=True`` prepends the
    current qpos state as a conditioning action row, matching the DROID joint_pos
    sequence-plan behavior.  ``root`` may be either one task split directory
    (``.../beat_block_hammer/aloha-agilex_clean_50``) or the RoboTwin root that
    contains many task directories.

    ``domain_name`` defaults to ``"robotwin"``. The DROID action SFT config skips
    loading the action projection heads, so the RoboTwin domain row is trained
    from scratch together with ``action2llm`` / ``llm2action``.

    ``action_space``, ``use_filter_dict``, ``filter_dict_path``, and
    ``success_only`` are accepted as DROID-config compatibility no-ops so an
    existing DROID LazyCall can be retargeted to this function from a launch
    script.
    """
    del action_space, use_filter_dict, filter_dict_path, success_only

    use_robotwin_384_mode = str(resolution) == "robotwin_384_mode"
    transform_resolution: str | int = "384" if use_robotwin_384_mode else resolution

    dataset = RoboTwinLeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        video_fps=video_fps,
        video_num_frames=video_num_frames,
        video_sample_stride=video_sample_stride,
        mode=mode,
        viewpoint=viewpoint,
        split_name=split_name,
        task_names=task_names,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        sample_stride=sample_stride,
        domain_name=domain_name,
        resolution=resolution,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        include_vit_first_frame=include_vit_first_frame,
        vit_processor_model_name=vit_processor_model_name,
        vit_processor_config_variant=vit_processor_config_variant,
        vit_processor_cache_dir=vit_processor_cache_dir,
        vit_first_frame_max_pixels=vit_first_frame_max_pixels,
        vit_first_frame_min_pixels=vit_first_frame_min_pixels,
        mid_soft=mid_soft,
    )
    sft = ActionSFTDataset(dataset, transform, transform_resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
