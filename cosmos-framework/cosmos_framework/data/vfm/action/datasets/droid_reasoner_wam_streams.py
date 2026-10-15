# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in DROID streams for homogeneous Reasoner/WAM joint training.

The two factories intentionally share the same raw DROID window settings, but
produce different schemas:

* :func:`get_droid_reasoner_stream_dataset` yields one- or three-image ShareGPT
  records consumed by the ordinary VLM language-model head.
* :func:`get_droid_wam_stream_dataset` applies the established action SFT
  transform and yields the video/action records consumed by OmniMoT flow
  matching.  Its caption is sampled deterministically from task/gold-subgoal
  with the configured weights.

Both use the established ``ActionIterableShuffleDataset`` by default.  WAM may
opt into a local deterministic primary-grip repetition stream without changing
that shared class.  When placed under a one-source
``RankPartitionedDistributor``, every rank/worker receives disjoint episodes,
episode order is shuffled, and overlapping windows remain sequential on disk.
The outer ``ReasonerWAMPairedDataLoader`` consumes one complete batch from each
child on every logical step; there is no ratio-based global Reasoner/WAM
alternation in this mode.
"""

from __future__ import annotations

import hashlib
import math
import os
from array import array
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_subgoal_dataset import (
    DROIDLeRobotSubgoalDataset,
)
from cosmos_framework.data.vfm.action.datasets.droid_reasoner_rationale_dataset import (
    DROIDStartFrameRationaleDataset,
)
from cosmos_framework.data.vfm.action.datasets.subtask_annotations import (
    object_position_augmentation_parts,
)
from cosmos_framework.data.vfm.action.reasoner_targets import reasoner_final_subtask
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

_REASONER_STRATIFIED_FAMILIES = (
    "navigation",
    "approach",
    "move_away",
    "contact",
    "transport",
    "placement",
    "other",
)
# Bucket order: navigation (excluding Approach), Approach, Move away,
# contact, transport, placement, and other.  Approach remains navigation
# semantically, but has a protected quota so ordinary Move windows cannot
# overwhelm it.
_REASONER_DEFAULT_FAMILY_RATIOS = (0.25, 0.10, 0.05, 0.24, 0.24, 0.09, 0.03)
# Keep locator-backed windows as a bounded share of each priority family.
# This is an opt-in train-sampler preference; the source labels and validation
# stream are unchanged.
_REASONER_POSITION_AUGMENT_PRIORITY_FRACTION = 0.50


def _reasoner_has_position_augmentable_segment(lookup: object) -> bool:
    """Whether a window contains a valid locator-backed action segment."""

    # Scan all segments so ``Move away`` followed by ``Approach`` or
    # ``Move toward`` is eligible while the dataset formatter preserves the
    # leading Move-away text in the final prompt.
    return any(
        object_position_augmentation_parts(segment) is not None
        for segment in getattr(lookup, "segments", ())
    )


def _reasoner_target_text(item: dict) -> str:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return ""
    target = conversations[-1]
    return str(target.get("value", "")) if isinstance(target, dict) else ""


def _reasoner_sampling_family(item: dict) -> str | None:
    """Return the primary-action sampling family for one Reasoner record."""

    text = " ".join(reasoner_final_subtask(_reasoner_target_text(item)).strip().split()).casefold()
    if not text:
        return None
    # The raw Reasoner dataset drops ordinary standalone Move-away windows,
    # but deliberately keeps Move away immediately before Grasp as a valid
    # single-step avoidance example.
    if text.startswith("move away"):
        return "move_away"
    if text.startswith("approach"):
        return "approach"
    if text.startswith(("move ", "move toward", "move to")):
        return "navigation"
    if text.startswith(("grasp", "close", "hold", "pick up")):
        return "contact"
    if text.startswith(("lift", "carry", "pull", "push")):
        return "transport"
    if text.startswith(("place", "release", "open", "drop")):
        return "placement"
    return "other"

class StratifiedReasonerIterableDataset(IterableDataset):
    """Deterministic per-epoch family-balanced Reasoner stream.

    The source dataset remains unchanged.  This wrapper builds a metadata-only
    global candidate index and samples each action family without replacement.
    A common shortage scale keeps all family ratios aligned when one family has
    fewer candidates than its requested quota.
    It is deliberately an iterable stream so rank/worker sharding and resume
    metadata follow the existing Reasoner loader contract.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int = 43,
        samples_per_epoch: int = 1_000_000,
        family_ratios: Sequence[float] = _REASONER_DEFAULT_FAMILY_RATIOS,
        candidate_pool_multiplier: float = 2.0,
        position_augment_priority: bool = False,
        resume_name: str = "reasoner",
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self._samples_per_epoch = int(samples_per_epoch)
        if self._samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        ratios = tuple(float(value) for value in family_ratios)
        if len(ratios) != len(_REASONER_STRATIFIED_FAMILIES) or any(value < 0 for value in ratios):
            raise ValueError(
                f"family_ratios must contain {len(_REASONER_STRATIFIED_FAMILIES)} non-negative values"
            )
        total = sum(ratios)
        if total <= 0:
            raise ValueError("family_ratios must have a positive sum")
        self._family_ratios = tuple(value / total for value in ratios)
        # Kept for config/API compatibility.  Candidate indices are now always
        # collected from the complete metadata source; using this value to cap
        # the pool caused an index-order prefix bias in training.
        self._candidate_pool_multiplier = float(candidate_pool_multiplier)
        if not math.isfinite(self._candidate_pool_multiplier) or self._candidate_pool_multiplier < 1.0:
            raise ValueError("candidate_pool_multiplier must be finite and >= 1.0")
        if not isinstance(position_augment_priority, bool):
            raise TypeError(f"position_augment_priority must be a bool, got {position_augment_priority!r}.")
        self._position_augment_priority = position_augment_priority
        self._resume_name = str(resume_name).strip()
        self.shard_world_size = 1
        self.shard_rank = 0
        # ``array('q')`` keeps each raw index at eight bytes and, unlike a
        # Python ``list[int]``, does not dirty a page in every forked worker
        # merely because an integer object's refcount is read.  This index is
        # built in the rank parent before DataLoader workers are forked.
        self._family_indices: dict[str, array] | None = None
        # Priority candidates are stored as the leading slice of their family
        # index.  Retaining the split lets ``__iter__`` sample the requested
        # share directly rather than losing it to a permutation of the full
        # family pool.
        self._family_priority_counts: dict[str, int] = {}

    def __len__(self) -> int:
        return self._samples_per_epoch

    @staticmethod
    def _quotas(total: int, ratios: Sequence[float]) -> list[int]:
        raw = [total * float(ratio) for ratio in ratios]
        values = [int(value) for value in raw]
        for index in sorted(range(len(values)), key=lambda i: raw[i] - values[i], reverse=True)[: total - sum(values)]:
            values[index] += 1
        return values

    def _build_family_indices(self) -> dict[str, array]:
        """Build a complete metadata-only candidate index once per process.

        The index uses ``array('q')`` so retaining all raw indices costs only
        eight bytes per candidate and remains copy-on-write friendly when the
        DataLoader workers fork.  Keeping the complete index is important:
        truncating at the first N source indices creates an episode-order
        prefix bias that is not fixed by the per-epoch permutation.
        """

        if self._family_indices is not None:
            return self._family_indices
        blocks = self._dataset.get_shuffle_blocks()
        families = {family: array("q") for family in _REASONER_STRATIFIED_FAMILIES}
        priority_families = {"navigation", "approach", "move_away"} if self._position_augment_priority else set()
        priority_indices = {family: array("q") for family in priority_families}
        fallback_indices = {family: array("q") for family in priority_families}
        for start, length in blocks:
            for raw_index in range(int(start), int(start) + int(length)):
                window, lookup = self._dataset._metadata_for_raw_index(int(raw_index))
                if not lookup.available:
                    continue
                family = _reasoner_sampling_family(
                    {"conversations": [{}, {"value": lookup.text}]}
                )
                if family is None:
                    continue
                # Keep every candidate.  Priority families are split into two
                # arrays only so __iter__ can reserve their requested share
                # without changing the complete candidate pool.
                if family in priority_families:
                    if _reasoner_has_position_augmentable_segment(lookup):
                        priority_indices[family].append(int(raw_index))
                    else:
                        fallback_indices[family].append(int(raw_index))
                else:
                    families[family].append(int(raw_index))
        for family in priority_families:
            families[family].extend(priority_indices[family])
            families[family].extend(fallback_indices[family])
        self._family_priority_counts = {
            family: len(priority_indices[family]) for family in priority_families
        }
        self._family_indices = families
        return families

    def get_sampling_plan(self, *, total_shards: int = 1) -> dict[str, object]:
        """Return a metadata-only preflight for the requested epoch size.

        ``target_reachable`` is true only when every positive-ratio family has
        enough distinct candidates to satisfy its exact quota.  A false value
        is expected for a long-tail shortage: the stream then applies one
        common scale factor and reports the smaller effective epoch.
        """

        total_shards = int(total_shards)
        if total_shards <= 0:
            raise ValueError("total_shards must be positive")
        families = self._build_family_indices()
        family_names = list(_REASONER_STRATIFIED_FAMILIES)
        target_quotas = dict(
            zip(
                family_names,
                self._quotas(self._samples_per_epoch, self._family_ratios),
                strict=True,
            )
        )
        available = {family: len(families[family]) for family in family_names}
        priority_available = {
            family: self._family_priority_counts.get(family, 0) for family in family_names
        }
        target_reachable = all(
            available[family] >= quota for family, quota in target_quotas.items() if quota > 0
        )
        scale = 1.0
        for family, quota in target_quotas.items():
            if quota > 0:
                scale = min(scale, available[family] / quota)
        effective_quotas = {
            family: min(available[family], int(quota * scale))
            for family, quota in target_quotas.items()
        }
        remainder = sum(effective_quotas.values()) % total_shards
        for family in sorted(family_names, key=lambda name: effective_quotas[name], reverse=True):
            if remainder <= 0:
                break
            if effective_quotas[family] > 0:
                effective_quotas[family] -= 1
                remainder -= 1
        return {
            "available": available,
            "priority_available": priority_available,
            "target": target_quotas,
            "target_reachable": target_reachable,
            "scale": scale,
            "effective": effective_quotas,
            "effective_total": sum(effective_quotas.values()),
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        global_shard = int(self.shard_rank) * num_workers + worker_id
        total_shards = max(1, int(self.shard_world_size) * num_workers)
        family_names = list(_REASONER_STRATIFIED_FAMILIES)
        plan = self.get_sampling_plan(total_shards=total_shards)
        families = self._family_indices
        assert families is not None
        target_quotas = plan["target"]
        effective_quotas = plan["effective"]
        scale = float(plan["scale"])
        available = plan["available"]
        priority_available = plan["priority_available"]
        # Shortage policy: use one common scale factor for every family.  This
        # preserves the requested ratios without replacement or cross-family
        # substitution.  If one family has only 80% of its target quota, all
        # family quotas are reduced to 80% for that epoch.
        missing: list[str] = []
        for family, quota in target_quotas.items():
            if available[family] == 0 and quota > 0:
                missing.append(family)
        if missing:
            raise RuntimeError(
                "Stratified Reasoner stream has no candidates for required families: "
                f"{missing}. Set that ratio to zero or disable stratified_sampling."
            )
        if worker_id == 0 and int(self.shard_rank) == 0:
            available_text = ", ".join(f"{family}={available[family]}" for family in family_names)
            target_text = ", ".join(f"{family}={target_quotas[family]}" for family in family_names)
            effective_text = ", ".join(f"{family}={effective_quotas[family]}" for family in family_names)
            print(
                "[reasoner_stratified] "
                f"target_reachable={plan['target_reachable']} scale={scale:.4f} "
                f"effective_total={plan['effective_total']} "
                f"available=[{available_text}] "
                f"target=[{target_text}] effective=[{effective_text}]",
                flush=True,
            )
            if self._position_augment_priority:
                priority_text = ", ".join(
                    f"{family}={priority_available[family]}" for family in family_names if priority_available[family]
                )
                priority_selected_text = ", ".join(
                    f"{family}={min(priority_available[family], int(round(effective_quotas[family] * _REASONER_POSITION_AUGMENT_PRIORITY_FRACTION)))}"
                    for family in family_names
                    if priority_available[family]
                )
                print(
                    "[reasoner_stratified] "
                    f"position_augmentable_available=[{priority_text}] "
                    f"position_augmentable_selected=[{priority_selected_text}]",
                    flush=True,
                )
        prefix = f"COSMOS_DL_STATE_{self._resume_name}_" if self._resume_name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0)) if self._resume_name else 0
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1)) if self._resume_name else -1
        epoch = resume_epoch
        while True:
            generator = torch.Generator().manual_seed(self._seed + epoch)
            # Build only this rank/worker's selection.  The old implementation
            # materialized, in *every* worker, Python lists for every global
            # family permutation, the full global selection, a second global
            # permutation, and finally the local shard.  With the default one
            # million samples and eight workers this multiplied transient RAM
            # by many gigabytes.  All workers still generate the same compact
            # per-family permutation, but take disjoint positions from it
            # before storing anything.
            local_selected = array("q")
            for family, quota in effective_quotas.items():
                candidates = families[family]
                priority_count = self._family_priority_counts.get(family, 0)
                if priority_count:
                    # Enforce the opt-in proportion in the actual epoch.  If
                    # locator-backed windows are scarce, use every available
                    # one and fill the remaining quota from ordinary windows.
                    priority_quota = min(
                        priority_count,
                        int(round(int(quota) * _REASONER_POSITION_AUGMENT_PRIORITY_FRACTION)),
                    )
                    fallback_count = len(candidates) - priority_count
                    fallback_quota = min(int(quota) - priority_quota, fallback_count)
                    # If the ordinary pool is short (including an entirely
                    # priority-backed family), use additional priority
                    # candidates to keep the family quota exact.
                    remaining = int(quota) - priority_quota - fallback_quota
                    extra_priority = min(priority_count - priority_quota, remaining)
                    priority_quota += extra_priority
                    remaining -= extra_priority
                    if remaining:
                        fallback_quota += min(fallback_count - fallback_quota, remaining)
                    priority_order = torch.randperm(priority_count, generator=generator)
                    fallback_order = torch.randperm(fallback_count, generator=generator)
                    for order_index in range(global_shard, priority_quota, total_shards):
                        local_selected.append(candidates[int(priority_order[order_index])])
                    for order_index in range(global_shard, fallback_quota, total_shards):
                        local_selected.append(candidates[priority_count + int(fallback_order[order_index])])
                else:
                    family_order = torch.randperm(len(candidates), generator=generator)
                    for order_index in range(global_shard, int(quota), total_shards):
                        local_selected.append(candidates[int(family_order[order_index])])
            local_order = torch.randperm(len(local_selected), generator=generator)
            for stream_pos, selected_index in enumerate(local_order):
                if self._resume_name and epoch == resume_epoch and stream_pos <= resume_pos:
                    continue
                raw_index = local_selected[int(selected_index)]
                item = self._dataset[raw_index]
                if self._resume_name:
                    yield {"_dp_epoch": epoch, "_dp_stream_pos": stream_pos, **item}
                else:
                    yield item
            epoch += 1


def _validate_primary_grip_oversample_factor(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"primary_grip_oversample_factor must be a number >= 1.0, got {value!r}.")
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"primary_grip_oversample_factor must be a number >= 1.0, got {value!r}.") from exc
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(
            f"primary_grip_oversample_factor must be finite and >= 1.0; got {value!r}. Downsampling is not supported."
        )
    return factor


class PrimaryGripOversampleIterableDataset(IterableDataset):
    """WAM-only ordered stream with deterministic primary-grip repetition.

    This intentionally duplicates the established episode/segment stream rather
    than modifying :class:`ActionIterableShuffleDataset`.  Every source window
    is emitted at least once.  For a primary-grip window, the requested factor
    ``r >= 1`` emits ``floor(r - 1)`` guaranteed extra copies and one additional
    copy with probability ``r - 1 - floor(r - 1)``.  The Bernoulli draw is a
    stable hash of ``(seed, epoch, raw_index)``; it therefore survives worker
    restarts and checkpoint resume without relying on Python's randomized hash.

    The wrapped dataset must be an :class:`ActionSFTDataset` whose raw dataset
    exposes ``is_primary_grip_window(index)``.  That method performs the
    boundary-trimmed (five-frame rule) subgoal lookup without decoding video.
    """

    def __init__(
        self,
        dataset: ActionSFTDataset,
        *,
        seed: int = 42,
        resume_name: str = "wam",
        primary_grip_oversample_factor: float = 1.0,
        primary_grip_oversample_seed: int | None = None,
        emit_resume_metadata: bool = True,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self._resume_name = str(resume_name).strip()
        # The joint Reasoner/WAM loader needs provenance to checkpoint a
        # stream cursor.  Standalone legacy PackingDataLoader does not consume
        # that metadata, so its factory opts out explicitly to keep the old
        # sample schema unchanged.  Keeping this separate from resume_name
        # also preserves the small, direct sampler API used by tests/tools.
        if not isinstance(emit_resume_metadata, bool):
            raise TypeError(f"emit_resume_metadata must be a bool, got {emit_resume_metadata!r}.")
        self._emit_resume_metadata = emit_resume_metadata
        self._oversample_factor = _validate_primary_grip_oversample_factor(primary_grip_oversample_factor)
        raw_seed = self._seed if primary_grip_oversample_seed is None else primary_grip_oversample_seed
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError(f"primary_grip_oversample_seed must be an integer, got {raw_seed!r}.")
        self._oversample_seed = int(raw_seed)
        extra = self._oversample_factor - 1.0
        self._guaranteed_extra = int(math.floor(extra))
        self._fractional_extra = float(extra - self._guaranteed_extra)
        self.shard_world_size = 1
        self.shard_rank = 0

        raw_dataset = getattr(dataset, "_dataset", None)
        self._is_primary_grip_window = getattr(raw_dataset, "is_primary_grip_window", None)
        if self._oversample_factor > 1.0 and not callable(self._is_primary_grip_window):
            raise TypeError(
                "PrimaryGripOversampleIterableDataset requires the wrapped raw DROID "
                "dataset to expose is_primary_grip_window(index)."
            )

    def __len__(self) -> int:
        """Return the unweighted base length (the stream itself is infinite)."""

        return len(self._dataset)

    def _fractional_draw(self, *, epoch: int, raw_index: int) -> float:
        payload = f"{self._oversample_seed}:{int(epoch)}:{int(raw_index)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    def _extra_copy_count(self, *, epoch: int, raw_index: int) -> int:
        if self._oversample_factor <= 1.0:
            return 0
        assert callable(self._is_primary_grip_window)
        if not bool(self._is_primary_grip_window(int(raw_index))):
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
            generator = torch.Generator().manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=generator).tolist()
            stream_pos = 0
            for block_index in order[global_shard::total_shards]:
                start, length = blocks[block_index]
                for raw_index in range(int(start), int(start) + int(length)):
                    emit_count = 1 + self._extra_copy_count(epoch=epoch, raw_index=raw_index)

                    # Resume metadata records the *emitted* position, so a
                    # skipped primary-grip sample may account for multiple
                    # copies.  Handle a cursor that falls inside the copies as
                    # well as one that falls before the source sample.
                    skipped = 0
                    if epoch == resume_epoch and stream_pos <= resume_pos:
                        skipped = min(emit_count, resume_pos - stream_pos + 1)
                        stream_pos += skipped
                    if skipped == emit_count:
                        continue

                    item = self._dataset[raw_index]
                    for _ in range(skipped, emit_count):
                        # Build a fresh top-level mapping for every copy.  The
                        # loader/transform pipeline mutates sample mappings in
                        # place; reusing one dictionary for a repeated grip
                        # window would otherwise corrupt the next duplicate.
                        if self._emit_resume_metadata:
                            yield {
                                "_dp_epoch": epoch,
                                "_dp_stream_pos": stream_pos,
                                **item,
                            }
                        else:
                            # Standalone WAM has no paired-loader cursor.  Do
                            # not leak internal resume metadata into the
                            # ordinary action collator/model batch.
                            yield dict(item)
                        stream_pos += 1
            epoch += 1


def _raw_subgoal_dataset(
    *,
    root: str,
    reasoner_wam_stream: str,
    fps: float,
    chunk_length: int,
    mode: str,
    viewpoint: str,
    action_space: str,
    use_state: bool,
    use_same_timestep_joint_action: bool,
    action_normalization: str | None,
    use_image_augmentation: bool,
    use_filter_dict: bool,
    filter_dict_path: str | None,
    prefilter_by_filter_dict: bool,
    camera_order_path: str | None,
    cam2base_extrinsic_path: str | None,
    success_only: bool,
    video_decode_backend: str,
    wrist_rotate_180: bool,
    video_sample_stride: int | None,
    use_exterior_plucker_condition: bool,
    defer_plucker_generation: bool,
    use_all_data_with_cam2cam_fallback: bool,
    subtask_annotations_path: str,
    precise_subtask_annotations_path: str | None = None,
    subtask_episode_filter_path: str | None,
    include_idle_stationary_subtasks: bool,
    apply_subtask_episode_filter_to_windows: bool,
    remove_post_completion_windows: bool,
    include_post_completion_move_away_windows: bool,
    reasoner_history_offsets: Sequence[int],
    reasoner_history_boundary: str,
    reasoner_history_choices: Sequence[Sequence[int]] | None,
    reasoner_history_choice_weights: Sequence[float] | None,
    reasoner_history_choice_seed: int,
    reasoner_view_mode: str = "concat",
    reasoner_image_long_edge: int | None = None,
    vit_frame_offsets: Sequence[int],
    condition_sources: Sequence[str],
    condition_weights: Sequence[float],
    condition_seed: int,
    rotation_aware_subgoal_degrees: float | None = None,
    max_subtasks_per_window: int = 100,
    subtask_boundary_trim_frames: int = 5,
    contact_label_delay_frames: int = 0,
    release_label_advance_frames: int = 0,
    force_move_away_followup: bool = False,
    drop_standalone_move_away: bool = False,
    include_preview_action_in_prompt: bool = False,
    object_position_augment_prob: float = 0.0,
    object_position_augment_style: str = "random",
    object_position_augment_seed: int = 42,
    open_gripper_move_augment: bool = False,
    open_gripper_single_subtask_augment: bool = False,
    execution_success_only: bool = False,
    episode_split: str = "all",
    validation_fraction: float = 0.02,
    validation_split_seed: int = 43,
    task_variant_policy: str,
) -> DROIDLeRobotSubgoalDataset:
    """Construct the common raw dataset without introducing a second index."""

    return DROIDLeRobotSubgoalDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_space=action_space,
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
        precise_subtask_annotations_path=precise_subtask_annotations_path,
        subtask_episode_filter_path=subtask_episode_filter_path,
        include_idle_stationary_subtasks=include_idle_stationary_subtasks,
        apply_subtask_episode_filter_to_windows=apply_subtask_episode_filter_to_windows,
        remove_post_completion_windows=remove_post_completion_windows,
        include_post_completion_move_away_windows=include_post_completion_move_away_windows,
        training_mode="reasoner_wam",
        reasoner_wam_stream=reasoner_wam_stream,
        reasoner_history_offsets=reasoner_history_offsets,
        reasoner_history_boundary=reasoner_history_boundary,
        reasoner_history_choices=reasoner_history_choices,
        reasoner_history_choice_weights=reasoner_history_choice_weights,
        reasoner_history_choice_seed=reasoner_history_choice_seed,
        reasoner_view_mode=reasoner_view_mode,
        reasoner_image_long_edge=reasoner_image_long_edge,
        vit_frame_offsets=vit_frame_offsets,
        condition_sources=condition_sources,
        condition_weights=condition_weights,
        condition_seed=condition_seed,
        rotation_aware_subgoal_degrees=rotation_aware_subgoal_degrees,
        max_subtasks_per_window=max_subtasks_per_window,
        subtask_boundary_trim_frames=subtask_boundary_trim_frames,
        contact_label_delay_frames=contact_label_delay_frames,
        release_label_advance_frames=release_label_advance_frames,
        force_move_away_followup=force_move_away_followup,
        drop_standalone_move_away=drop_standalone_move_away,
        include_preview_action_in_prompt=include_preview_action_in_prompt,
        object_position_augment_prob=object_position_augment_prob,
        object_position_augment_style=object_position_augment_style,
        object_position_augment_seed=object_position_augment_seed,
        open_gripper_move_augment=open_gripper_move_augment,
        open_gripper_single_subtask_augment=open_gripper_single_subtask_augment,
        execution_success_only=execution_success_only,
        episode_split=episode_split,
        validation_fraction=validation_fraction,
        validation_split_seed=validation_split_seed,
        task_variant_policy=task_variant_policy,
    )


def get_droid_reasoner_stream_dataset(
    *,
    root: str,
    subtask_annotations_path: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "policy",
    viewpoint: str = "concat_view",
    action_space: str = "joint_pos",
    use_state: bool = True,
    use_same_timestep_joint_action: bool = False,
    action_normalization: str | None = None,
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
    use_all_data_with_cam2cam_fallback: bool = False,
    subtask_episode_filter_path: str | None = None,
    precise_subtask_annotations_path: str | None = None,
    include_idle_stationary_subtasks: bool = False,
    remove_post_completion_windows: bool = False,
    include_post_completion_move_away_windows: bool = False,
    reasoner_history_offsets: Sequence[int] = (0,),
    reasoner_history_boundary: str = "repeat_first",
    reasoner_history_choices: Sequence[Sequence[int]] | None = None,
    reasoner_history_choice_weights: Sequence[float] | None = None,
    reasoner_history_choice_seed: int = 0,
    rotation_aware_subgoal_degrees: float | None = None,
    max_subtasks_per_window: int = 100,
    subtask_boundary_trim_frames: int = 5,
    contact_label_delay_frames: int = 0,
    release_label_advance_frames: int = 0,
    force_move_away_followup: bool = False,
    drop_standalone_move_away: bool = False,
    include_preview_action_in_prompt: bool = False,
    object_position_augment_prob: float = 0.0,
    object_position_augment_style: str = "random",
    object_position_augment_seed: int = 42,
    open_gripper_move_augment: bool = False,
    open_gripper_single_subtask_augment: bool = False,
    execution_success_only: bool = False,
    episode_split: str = "all",
    validation_fraction: float = 0.02,
    validation_split_seed: int = 43,
    task_variant_policy: str = "first",
    episode_shuffle_seed: int = 42,
    resume_name: str = "reasoner",
    stratified_sampling: bool = False,
    stratified_samples_per_epoch: int = 1_000_000,
    stratified_family_ratios: Sequence[float] = _REASONER_DEFAULT_FAMILY_RATIOS,
    stratified_candidate_pool_multiplier: float = 2.0,
    position_augment_priority: bool = False,
    rationale_annotations_path: str | None = None,
    rationale_only: bool = False,
    rationale_strict_window_match: bool = False,
    exact_action_sampling: bool = False,
    open_gripper_sampling_ratio: float = 0.0,
    separate_placement_actions: bool = False,
    reasoner_view_mode: str = "concat",
    reasoner_image_long_edge: int | None = None,
) -> Dataset:
    """Return the independently shuffled assistant-subgoal Reasoner stream.

    The factory requires an annotation source.  When
    ``subtask_episode_filter_path`` is set, its clip-level ranges are
    applied by the shared parent DROID dataset; missing/gap labels inside the
    retained ranges stay in the stream and are masked by
    ``DROIDReasonerSupervisionProcessor``.
    """

    if rationale_only and not rationale_annotations_path:
        raise ValueError("rationale_only=True requires rationale_annotations_path.")
    if rationale_annotations_path and not rationale_only:
        raise ValueError(
            "rationale_annotations_path is a separate replay stream; set rationale_only=True and mix that "
            "stream with the ordinary Reasoner loader at an explicit batch ratio."
        )
    if exact_action_sampling:
        raise ValueError(
            "exact_action_sampling is not part of the restored baseline stream; "
            "set REASONER_EXACT_ACTION_SAMPLING=false to use legacy stratified sampling."
        )
    if float(open_gripper_sampling_ratio) != 0.0 or separate_placement_actions:
        raise ValueError(
            "additive placement samplers are not part of the restored baseline stream; "
            "leave open_gripper_sampling_ratio=0 and separate_placement_actions=false."
        )

    dataset: Dataset = _raw_subgoal_dataset(
        root=root,
        reasoner_wam_stream="reasoner",
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_space=action_space,
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
        use_exterior_plucker_condition=False,
        defer_plucker_generation=False,
        use_all_data_with_cam2cam_fallback=use_all_data_with_cam2cam_fallback,
        subtask_annotations_path=subtask_annotations_path,
        precise_subtask_annotations_path=precise_subtask_annotations_path,
        subtask_episode_filter_path=subtask_episode_filter_path,
        include_idle_stationary_subtasks=include_idle_stationary_subtasks,
        apply_subtask_episode_filter_to_windows=True,
        remove_post_completion_windows=remove_post_completion_windows,
        include_post_completion_move_away_windows=include_post_completion_move_away_windows,
        reasoner_history_offsets=reasoner_history_offsets,
        reasoner_history_boundary=reasoner_history_boundary,
        reasoner_history_choices=reasoner_history_choices,
        reasoner_history_choice_weights=reasoner_history_choice_weights,
        reasoner_history_choice_seed=reasoner_history_choice_seed,
        reasoner_view_mode=reasoner_view_mode,
        reasoner_image_long_edge=reasoner_image_long_edge,
        vit_frame_offsets=(0,),
        condition_sources=("task",),
        condition_weights=(1.0,),
        condition_seed=0,
        rotation_aware_subgoal_degrees=rotation_aware_subgoal_degrees,
        max_subtasks_per_window=max_subtasks_per_window,
        subtask_boundary_trim_frames=subtask_boundary_trim_frames,
        contact_label_delay_frames=contact_label_delay_frames,
        release_label_advance_frames=release_label_advance_frames,
        force_move_away_followup=force_move_away_followup,
        drop_standalone_move_away=drop_standalone_move_away,
        include_preview_action_in_prompt=include_preview_action_in_prompt,
        object_position_augment_prob=object_position_augment_prob,
        object_position_augment_style=object_position_augment_style,
        object_position_augment_seed=object_position_augment_seed,
        open_gripper_move_augment=open_gripper_move_augment,
        open_gripper_single_subtask_augment=open_gripper_single_subtask_augment,
        execution_success_only=execution_success_only,
        episode_split=episode_split,
        validation_fraction=validation_fraction,
        validation_split_seed=validation_split_seed,
        task_variant_policy=task_variant_policy,
    )
    if rationale_only:
        assert rationale_annotations_path is not None
        dataset = DROIDStartFrameRationaleDataset(
            dataset,
            rationale_annotations_path=rationale_annotations_path,
            strict_window_match=rationale_strict_window_match,
        )
    if stratified_sampling:
        stream = StratifiedReasonerIterableDataset(
            dataset,
            seed=episode_shuffle_seed,
            samples_per_epoch=stratified_samples_per_epoch,
            family_ratios=stratified_family_ratios,
            candidate_pool_multiplier=stratified_candidate_pool_multiplier,
            position_augment_priority=position_augment_priority,
            resume_name=resume_name,
        )
        # Build once before DataLoader workers fork/spawn.  Otherwise every
        # worker would independently scan the full metadata index.
        stream._build_family_indices()
        return stream
    return ActionIterableShuffleDataset(dataset, seed=episode_shuffle_seed, resume_name=resume_name)


def get_droid_wam_stream_dataset(
    *,
    root: str,
    subtask_annotations_path: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "policy",
    viewpoint: str = "concat_view",
    action_space: str = "joint_pos",
    use_state: bool = True,
    use_same_timestep_joint_action: bool = False,
    action_normalization: str | None = None,
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
    subtask_episode_filter_path: str | None = None,
    precise_subtask_annotations_path: str | None = None,
    include_idle_stationary_subtasks: bool = False,
    apply_subtask_episode_filter_to_windows: bool = True,
    remove_post_completion_windows: bool = False,
    include_post_completion_move_away_windows: bool = False,
    condition_sources: Sequence[str] = ("task", "gold_subgoal"),
    # Source order is (task, gold_subgoal): default effective ratio is 2:8.
    condition_weights: Sequence[float] = (2.0, 8.0),
    condition_seed: int = 42,
    rotation_aware_subgoal_degrees: float | None = None,
    max_subtasks_per_window: int = 100,
    object_position_augment_prob: float = 0.0,
    object_position_augment_style: str = "random",
    object_position_augment_seed: int = 42,
    subtask_boundary_trim_frames: int = 5,
    contact_label_delay_frames: int = 0,
    release_label_advance_frames: int = 0,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    camera_cfg_dropout_rate: float = 0.0,
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
    defer_vit_pixel_values: bool = False,
    vit_frame_offsets: Sequence[int] = (0,),
    episode_shuffle_seed: int = 42,
    primary_grip_oversample_factor: float = 1.0,
    primary_grip_oversample_seed: int | None = None,
) -> Dataset:
    """Return the transformed WAM stream for mode 2.

    The WAM text input is always selected from task/gold-subgoal according to
    the supplied weights.  Gold-unavailable windows fall back to task text;
    no window is removed.  ``primary_grip_oversample_factor`` is an opt-in
    sample weight implemented as deterministic repetition.  Its default 1.0
    returns the unchanged legacy iterable stream.
    """

    raw = _raw_subgoal_dataset(
        root=root,
        reasoner_wam_stream="wam",
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_space=action_space,
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
        precise_subtask_annotations_path=precise_subtask_annotations_path,
        subtask_episode_filter_path=subtask_episode_filter_path,
        include_idle_stationary_subtasks=include_idle_stationary_subtasks,
        apply_subtask_episode_filter_to_windows=apply_subtask_episode_filter_to_windows,
        remove_post_completion_windows=remove_post_completion_windows,
        include_post_completion_move_away_windows=include_post_completion_move_away_windows,
        reasoner_history_offsets=(0,),
        reasoner_history_boundary="repeat_first",
        reasoner_history_choices=None,
        reasoner_history_choice_weights=None,
        reasoner_history_choice_seed=0,
        vit_frame_offsets=vit_frame_offsets,
        condition_sources=condition_sources,
        condition_weights=condition_weights,
        condition_seed=condition_seed,
        rotation_aware_subgoal_degrees=rotation_aware_subgoal_degrees,
        max_subtasks_per_window=max_subtasks_per_window,
        subtask_boundary_trim_frames=subtask_boundary_trim_frames,
        contact_label_delay_frames=contact_label_delay_frames,
        release_label_advance_frames=release_label_advance_frames,
        object_position_augment_prob=object_position_augment_prob,
        object_position_augment_style=object_position_augment_style,
        object_position_augment_seed=object_position_augment_seed,
        execution_success_only=False,
        episode_split="all",
        validation_fraction=0.02,
        validation_split_seed=43,
        # Preserve the parent DROID action dataset's random task-variant
        # choice for WAM.  Reasoner may use a deterministic first variant, but
        # the WAM stream must remain behaviorally identical to viddown4.
        task_variant_policy="random",
    )
    pad_keys = ["video"]
    if use_exterior_plucker_condition:
        pad_keys.append("plucker_condition")
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
        include_vit_first_frame=include_vit_first_frame,
        vit_processor_model_name=vit_processor_model_name,
        vit_processor_config_variant=vit_processor_config_variant,
        vit_processor_cache_dir=vit_processor_cache_dir,
        vit_first_frame_max_pixels=vit_first_frame_max_pixels,
        vit_first_frame_min_pixels=vit_first_frame_min_pixels,
        defer_vit_pixel_values=defer_vit_pixel_values,
        vit_frame_offsets=vit_frame_offsets,
        mid_soft=False,
        tokenize_subtask=False,
    )
    transformed = ActionSFTDataset(raw, transform, resolution)
    resolved_oversample_factor = _validate_primary_grip_oversample_factor(primary_grip_oversample_factor)
    if resolved_oversample_factor == 1.0:
        # Preserve the exact legacy WAM stream and its resume implementation
        # when the opt-in sampler is disabled.
        return ActionIterableShuffleDataset(transformed, seed=episode_shuffle_seed, resume_name="wam")
    return PrimaryGripOversampleIterableDataset(
        transformed,
        seed=episode_shuffle_seed,
        resume_name="wam",
        primary_grip_oversample_factor=resolved_oversample_factor,
        primary_grip_oversample_seed=primary_grip_oversample_seed,
    )


__all__ = [
    "get_droid_reasoner_stream_dataset",
    "get_droid_wam_stream_dataset",
    "PrimaryGripOversampleIterableDataset",
    "StratifiedReasonerIterableDataset",
]
