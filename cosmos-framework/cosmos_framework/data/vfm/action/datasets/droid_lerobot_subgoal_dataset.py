# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in DROID LeRobot datasets for subgoal-conditioned training modes.

The sample-building style mirrors :mod:`robotwin_lerobot_dataset`, while the
already validated DROID video/action/index implementation is reused from
``DROIDLeRobotDataset``.  Keeping this as a separate class is intentional: the
legacy DROID dataset and its launchers retain their exact defaults and output
contract.

``mid_soft``
    The WAM record plus the optional ``subtask`` text consumed by the existing
    mid-soft subtask decoder.
``reasoner_wam``
    Two concrete stream records used by the joint experiment: a Reasoner
    ``image``/``conversations`` record and a WAM video/action record whose text
    condition is selected from task or gold subgoal.  The streams are kept
    separate so each batch has one schema; they still share one optimizer and
    checkpoint.
``wam``
    Standalone WAM action/video records.  Every parent window is retained when
    ``apply_subtask_episode_filter_to_windows=False``; windows without a
    complete six-rule subtask label use the task instruction as their text
    condition.  ``remove_post_completion_windows=True`` is the selective
    exception: only windows after the first completed group are removed. This
    mode never creates a Reasoner conversation or subtask CE target.

WAM condition sources are ``task``, ``gold_subgoal``, and
``gold_subgoal_lap``. The last one preserves the gold subgoal and prefixes a
compact LAP-style EEF cue: qualitative translation directions plus rounded
rotation angles, without centimetre magnitudes or gripper open/close text.

Subgoal annotation intervals are episode-local and half-open.  By default the
target follows the action timeline: the target query uses the same frame offset
as the commanded action rows.  This keeps subgoal supervision causally aligned
with the command sequence while retaining an explicit ``target_frame_offset``
override for observation-window experiments.  If the target spans adjacent
subgoal intervals, the label preserves their order as ``subgoal1 then
subgoal2`` after omitting a leading or trailing interval that contributes at
most five target frames.  Annotation coverage never changes the parent dataset
length.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch

from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import (
    DROIDLeRobotDataset,
    FilterDictKeyMode,
    PoseConvention,
    VideoDecodeBackend,
    Viewpoint,
    _as_bool,
    _episode_is_validation,
    _should_log_dataset_summary,
)
from cosmos_framework.data.vfm.action.datasets.lap_control_text import (
    append_lap_control,
    append_rotation_purpose,
    eef_window_action,
    format_lap_control,
    format_rotation_control,
)
from cosmos_framework.data.vfm.action.datasets.precise_subtask_overrides import (
    PreciseSubtaskEpisode,
    load_precise_subtask_points,
)
from cosmos_framework.data.vfm.action.datasets.subtask_annotations import (
    EpisodeSubtaskAnnotations,
    SubtaskAnnotation,
    advance_release_subtask_boundaries,
    augment_subtask_with_object_position,
    compose_subtask_text,
    delay_contact_subtask_boundaries,
    object_position_augmentation_parts,
    relabel_subtask_annotations,
    resolve_subtask_segments,
)
from cosmos_framework.data.vfm.action.prompt_templates import format_action_prediction_prompt
from cosmos_framework.data.vfm.action.reasoner_targets import format_reasoner_rationale_target

# ``wam`` is the standalone, action-only subgoal condition mode.  Keep the
# historical ``reasoner_wam`` value (and its explicit stream selector) for
# existing joint experiments; callers that only want WAM should use ``wam``
# so no Reasoner record is ever constructed.
TrainingMode = Literal["mid_soft", "reasoner_wam", "wam"]
ReasonerWAMStream = Literal["reasoner", "wam"]
SubgoalFrameAlignment = Literal["window", "action"]
ConditionSource = Literal["task", "gold_subgoal", "gold_subgoal_lap"]
HistoryBoundary = Literal["repeat_first", "error"]
_SCHEMA_VERSION = "droid-lerobot-subgoal-v1"
_EVENT_MIN_START_COVERAGE_FRAMES = 8
_MOVE_OPEN_GRIPPER_RE = re.compile(r"\bmove\s+the\s+open\s+gripper\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DROIDSubgoalWindow:
    """Resolved provenance for one raw DROID training window."""

    raw_index: int
    episode_position: int
    episode_index: int
    global_start: int
    episode_global_start: int
    episode_frame_count: int
    window_start_frame: int
    window_end_frame: int
    action_start_frame: int
    action_end_frame: int
    target_start_frame: int
    target_end_frame: int

    @property
    def sample_id(self) -> str:
        return f"droid:{self.episode_index}:{self.window_start_frame}"


@dataclass(frozen=True, slots=True)
class _SubgoalLookup:
    text: str
    status: str
    segments: tuple[SubtaskAnnotation, ...]

    @property
    def available(self) -> bool:
        return bool(self.text)


def _rewrite_move_open_gripper_text(text: str) -> str:
    """Rewrite ``move the open gripper`` into an explicit open-then-move action.

    The rewrite is deliberately text-only and leaves every other word and
    punctuation mark unchanged.  Continuations after a clause stay lowercase,
    while a sentence-initial replacement is capitalized.
    """

    if not isinstance(text, str) or not text:
        return text

    def replacement(match: re.Match[str]) -> str:
        prefix = text[: match.start()]
        value = "open the gripper and move the gripper"
        if not prefix.strip() or prefix.rstrip().endswith((".", "!", "?", ";", ":", "</think>")):
            value = value.capitalize()
        return value

    return _MOVE_OPEN_GRIPPER_RE.sub(replacement, text)


def _rewrite_single_move_open_gripper_text(text: str) -> str:
    """Rewrite the phrase for a single-subtask open-then-move target."""

    if not isinstance(text, str) or not text:
        return text

    def replacement(match: re.Match[str]) -> str:
        prefix = text[: match.start()]
        value = "open the gripper, next move the open gripper"
        if not prefix.strip() or prefix.rstrip().endswith((".", "!", "?", ";", ":", "</think>")):
            value = value.capitalize()
        return value

    return _MOVE_OPEN_GRIPPER_RE.sub(replacement, text)


def _contiguous_event_end_frames(segments: Sequence[SubtaskAnnotation]) -> np.ndarray:
    """Return the exclusive end frame of each row's contiguous event.

    One goal-level event commonly spans several refined source groups.  Event
    eligibility is defined by coverage from the target start to the end of the
    *event*, not merely to the end of its first source group.
    """

    event_ends = np.empty(len(segments), dtype=np.int64)
    for index in range(len(segments) - 1, -1, -1):
        current = segments[index]
        if index + 1 < len(segments):
            following = segments[index + 1]
            continues_event = (
                current.event_caption is not None
                and current.event_id is not None
                and current.event_id == following.event_id
                and current.event_caption == following.event_caption
                and current.end_frame == following.start_frame
            )
            if continues_event:
                event_ends[index] = event_ends[index + 1]
                continue
        event_ends[index] = current.end_frame
    return event_ends


def _effective_subtask_counts_for_windows(
    annotations: EpisodeSubtaskAnnotations | None,
    *,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    boundary_trim_frames: int = 5,
    pregrasp_boundary_trim_frames: int = 0,
    count_annotation_segments: bool = False,
) -> np.ndarray:
    """Count semantic labels after the same boundary trim used by training.

    ``count_annotation_segments`` is an opt-in event-training mode.  Historical
    subgoal recipes count only label changes, so adjacent rows with identical
    captions behave as one semantic label.  Goal-level events must preserve
    every annotation boundary even when their natural-language captions happen
    to match, otherwise a window could train across two independent events.
    """

    starts = np.asarray(target_starts, dtype=np.int64)
    ends = np.asarray(target_ends, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError(
            f"target_starts and target_ends must have matching shapes, got {starts.shape} and {ends.shape}"
        )
    counts = np.zeros(starts.shape, dtype=np.int64)
    if annotations is None or not annotations.segments or starts.size == 0:
        return counts
    if (
        isinstance(pregrasp_boundary_trim_frames, bool)
        or not isinstance(pregrasp_boundary_trim_frames, int)
        or pregrasp_boundary_trim_frames < 0
    ):
        raise ValueError(
            "pregrasp_boundary_trim_frames must be a non-negative integer, "
            f"got {pregrasp_boundary_trim_frames!r}."
        )

    annotation_starts = np.asarray(annotations.starts, dtype=np.int64)
    annotation_ends = np.fromiter(
        (segment.end_frame for segment in annotations.segments),
        dtype=np.int64,
        count=len(annotations.segments),
    )
    label_keys = np.asarray([segment.subtask.casefold() for segment in annotations.segments], dtype=object)
    # The exporter populates ``source_subtask`` on every refined row, even
    # rows outside accepted events (whose event_caption is null).  This lets
    # us distinguish event-enriched files from legacy annotations and reject
    # explicit null-caption gaps.
    event_mode = any(bool(segment.source_subtask) for segment in annotations.segments)
    event_keys = np.asarray(
        [segment.event_id if segment.event_caption is not None else None for segment in annotations.segments],
        dtype=object,
    )
    event_end_frames = _contiguous_event_end_frames(annotations.segments)
    source_label_keys = np.asarray(
        [str(segment.source_subtask or "").strip().casefold() for segment in annotations.segments],
        dtype=object,
    )
    # The dedicated pre-grasp rule is a coarse-boundary rule.  Refined JSON
    # rows may split one coarse Move/Approach segment into several fine rows;
    # those rows must not hide the following coarse Grasp boundary.
    coarse_label_keys = np.asarray(
        [str(segment.coarse_subtask or segment.subtask).casefold() for segment in annotations.segments],
        dtype=object,
    )
    gap_prefix = np.zeros(len(annotations.segments), dtype=np.int64)
    label_change_prefix = np.zeros(len(annotations.segments), dtype=np.int64)
    if len(annotations.segments) > 1:
        gap_prefix[1:] = np.cumsum(annotation_starts[1:] != annotation_ends[:-1])
        label_change_prefix[1:] = np.cumsum(label_keys[1:] != label_keys[:-1])

    first = np.searchsorted(annotation_starts, starts, side="right") - 1
    last = np.searchsorted(annotation_starts, ends - 1, side="right") - 1
    valid = (first >= 0) & (last >= first) & (last < len(annotations.segments)) & (ends > starts)
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return counts

    first_valid = first[positions]
    last_valid = last[positions]
    covered = (
        (annotation_starts[first_valid] <= starts[positions])
        & (annotation_ends[first_valid] > starts[positions])
        & (annotation_starts[last_valid] < ends[positions])
        & (annotation_ends[last_valid] >= ends[positions])
        & ((gap_prefix[last_valid] - gap_prefix[first_valid]) == 0)
    )
    positions = positions[covered]
    if positions.size == 0:
        return counts

    effective_first = first[positions].copy()
    effective_last = last[positions].copy()
    if pregrasp_boundary_trim_frames > 0:
        coarse_next_distinct = np.full(len(annotations.segments), -1, dtype=np.int64)
        for position in range(len(annotations.segments) - 2, -1, -1):
            coarse_next_distinct[position] = (
                position + 1
                if coarse_label_keys[position + 1] != coarse_label_keys[position]
                else coarse_next_distinct[position + 1]
            )
        candidate_next = coarse_next_distinct[effective_first]
        valid_next = (candidate_next >= 0) & (candidate_next <= effective_last)
        safe_next = np.clip(candidate_next, 0, len(annotations.segments) - 1)
        leading_overlap = annotation_starts[safe_next] - starts[positions]
        first_labels = coarse_label_keys[effective_first]
        next_labels = coarse_label_keys[safe_next]
        is_approach = np.asarray(
            [
                (str(label).startswith("approach") or str(label).startswith("move"))
                and not str(label).startswith("move away")
                for label in first_labels
            ],
            dtype=bool,
        )
        is_grasp = np.asarray(
            [str(label).startswith("grasp") for label in next_labels],
            dtype=bool,
        )
        apply_pregrasp = valid_next & is_approach & is_grasp & (leading_overlap <= pregrasp_boundary_trim_frames)
        effective_first[apply_pregrasp] = candidate_next[apply_pregrasp]
    multi_segment = effective_first < effective_last
    leading_overlap = np.minimum(ends[positions], annotation_ends[effective_first]) - starts[positions]
    effective_first[multi_segment & (leading_overlap <= boundary_trim_frames)] += 1

    multi_segment = effective_first < effective_last
    trailing_overlap = ends[positions] - np.maximum(starts[positions], annotation_starts[effective_last])
    effective_last[multi_segment & (trailing_overlap <= boundary_trim_frames)] -= 1
    if count_annotation_segments:
        counts[positions] = 1 + effective_last - effective_first
    else:
        counts[positions] = 1 + label_change_prefix[effective_last] - label_change_prefix[effective_first]
    if event_mode:
        # An event WAM sample emits only the event at target_start.  The
        # fixed 32-frame action target can extend into the next accepted event
        # without becoming a multi-event caption.  It is rejected only if the
        # starting event contributes fewer than eight frames, or if any later
        # target frame enters an explicit null-caption gap.
        for local_index, position in enumerate(positions):
            first_i = int(first[positions[local_index]])
            last_i = int(last[positions[local_index]])
            start_coverage = int(min(event_end_frames[first_i], ends[position]) - starts[position])
            starts_idle = source_label_keys[first_i] in {"idle", "remain stationary", "stationary"}
            contains_missing_event = any(event_keys[i] is None for i in range(first_i, last_i + 1))
            counts[position] = (
                1
                if (
                    event_keys[first_i] is not None
                    and start_coverage >= _EVENT_MIN_START_COVERAGE_FRAMES
                    and not starts_idle
                    and not contains_missing_event
                )
                else 0
            )
    return counts


def _uniform_execution_success_for_windows(
    annotations: EpisodeSubtaskAnnotations | None,
    *,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> np.ndarray:
    """Resolve a single execution outcome for each fully covered window.

    Values are 1 when every covered source group is successful, 0 when every
    covered source group is explicitly unsuccessful, and -1 for a coverage
    gap, missing outcome, or true/false mixture. A 32-frame WAM target needs
    one unambiguous execution-outcome label. Unlike semantic label
    composition, no five-frame boundary trimming is applied: every physical
    frame in the action-aligned window must have annotation coverage.
    """

    starts = np.asarray(target_starts, dtype=np.int64)
    ends = np.asarray(target_ends, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError(
            f"target_starts and target_ends must have matching shapes, got {starts.shape} and {ends.shape}"
        )
    outcomes = np.full(starts.shape, -1, dtype=np.int8)
    if annotations is None or not annotations.segments or starts.size == 0:
        return outcomes

    annotation_starts = np.asarray(annotations.starts, dtype=np.int64)
    annotation_ends = np.fromiter(
        (segment.end_frame for segment in annotations.segments),
        dtype=np.int64,
        count=len(annotations.segments),
    )
    values = np.asarray(
        [
            -1
            if segment.execution_success is None
            else int(segment.execution_success)
            for segment in annotations.segments
        ],
        dtype=np.int8,
    )
    gap_prefix = np.zeros(len(annotations.segments), dtype=np.int64)
    unknown_prefix = np.cumsum(values < 0, dtype=np.int64)
    true_prefix = np.cumsum(values == 1, dtype=np.int64)
    false_prefix = np.cumsum(values == 0, dtype=np.int64)
    if len(annotations.segments) > 1:
        gap_prefix[1:] = np.cumsum(annotation_starts[1:] != annotation_ends[:-1])

    first = np.searchsorted(annotation_starts, starts, side="right") - 1
    last = np.searchsorted(annotation_starts, ends - 1, side="right") - 1
    valid = (first >= 0) & (last >= first) & (last < len(annotations.segments)) & (ends > starts)
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return outcomes
    first_valid = first[positions]
    last_valid = last[positions]
    covered = (
        (annotation_starts[first_valid] <= starts[positions])
        & (annotation_ends[first_valid] > starts[positions])
        & (annotation_starts[last_valid] < ends[positions])
        & (annotation_ends[last_valid] >= ends[positions])
        & ((gap_prefix[last_valid] - gap_prefix[first_valid]) == 0)
    )
    positions = positions[covered]
    if positions.size == 0:
        return outcomes

    first_valid = first[positions]
    last_valid = last[positions]
    before = first_valid - 1

    def range_count(prefix: np.ndarray) -> np.ndarray:
        return prefix[last_valid] - np.where(before >= 0, prefix[before], 0)

    known = range_count(unknown_prefix) == 0
    has_true = range_count(true_prefix) > 0
    has_false = range_count(false_prefix) > 0
    outcomes[positions[known & has_true & ~has_false]] = 1
    outcomes[positions[known & has_false & ~has_true]] = 0
    return outcomes


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()



def _tensor_to_pil(image: torch.Tensor) -> Any:
    """Convert a raw CHW uint8 image to PIL lazily for the VLM processor."""

    from PIL import Image

    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"Expected a CHW image with one or three channels, got {tuple(image.shape)}.")
    array = image.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def _validate_target_offset(
    *,
    subgoal_frame_alignment: SubgoalFrameAlignment,
    target_frame_offset: int | None,
    action_frame_offset: int,
) -> tuple[SubgoalFrameAlignment, int]:
    if subgoal_frame_alignment not in ("window", "action"):
        raise ValueError(f"subgoal_frame_alignment must be 'window' or 'action', got {subgoal_frame_alignment!r}.")
    if action_frame_offset not in (0, 1):
        raise ValueError(f"action_frame_offset must be 0 or 1, got {action_frame_offset}.")
    alignment_offset = 0 if subgoal_frame_alignment == "window" else action_frame_offset
    if target_frame_offset is None:
        return subgoal_frame_alignment, alignment_offset
    if isinstance(target_frame_offset, bool) or not isinstance(target_frame_offset, int):
        raise TypeError(f"target_frame_offset must be an integer 0 or 1, got {target_frame_offset!r}.")
    if target_frame_offset not in (0, 1):
        raise ValueError(f"target_frame_offset must be 0 or 1, got {target_frame_offset}.")
    if subgoal_frame_alignment == "action" and target_frame_offset != action_frame_offset:
        raise ValueError(
            "subgoal_frame_alignment='action' requires target_frame_offset="
            f"{action_frame_offset}; got {target_frame_offset}."
        )
    resolved_alignment: SubgoalFrameAlignment = "action" if subgoal_frame_alignment == "action" else "window"
    return resolved_alignment, target_frame_offset


def _validate_history_offsets(values: Sequence[int]) -> tuple[int, ...]:
    offsets = tuple(values)
    if not offsets:
        raise ValueError("history offsets must contain at least the current-frame offset 0.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
        raise TypeError(f"reasoner_history_offsets must contain integers, got {offsets!r}.")
    if any(value > 0 for value in offsets):
        raise ValueError(
            f"reasoner_history_offsets must be <= 0 so the Reasoner cannot observe future frames; got {offsets!r}."
        )
    if offsets[-1] != 0:
        raise ValueError(f"reasoner_history_offsets must end with the current frame offset 0, got {offsets!r}.")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError(f"history offsets must be unique and chronologically sorted, got {offsets!r}.")
    return offsets


def _validate_history_offset_choices(
    values: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if values is None:
        return None
    choices = tuple(_validate_history_offsets(choice) for choice in values)
    if not choices:
        raise ValueError("vit_frame_offset_choices must contain at least one offset choice.")
    return choices


def _normalize_history_choice_weights(
    weights: Sequence[float] | None,
    *,
    choice_count: int,
) -> tuple[float, ...]:
    """Validate weighted 1-image/3-image Reasoner history choices."""

    resolved = tuple(1.0 for _ in range(choice_count)) if weights is None else tuple(float(value) for value in weights)
    if len(resolved) != choice_count:
        raise ValueError(
            f"reasoner_history_choice_weights length {len(resolved)} does not match the {choice_count} history choices."
        )
    if any(not math.isfinite(value) or value < 0.0 for value in resolved):
        raise ValueError(f"reasoner_history_choice_weights must be finite and non-negative, got {resolved!r}.")
    if sum(resolved) <= 0.0:
        raise ValueError("reasoner_history_choice_weights must contain at least one positive value.")
    return resolved


def _normalize_condition_config(
    *,
    condition_sources: Sequence[str] | None,
    condition_weights: Sequence[float] | None,
) -> tuple[tuple[ConditionSource, ...], tuple[float, ...]]:
    if condition_sources is not None:
        if isinstance(condition_sources, str):
            raise TypeError(
                "condition_sources must be a sequence of source names, not one string; "
                "use condition_sources=('task',) for a single source."
            )
        raw_sources = tuple(str(value).strip().lower() for value in condition_sources)
    else:
        raw_sources = ("task",)

    if not raw_sources:
        raise ValueError("condition_sources must not be empty.")
    allowed_sources = {"task", "gold_subgoal", "gold_subgoal_lap"}
    unknown = sorted(set(raw_sources) - allowed_sources)
    if unknown:
        raise ValueError(
            f"condition sources must be chosen from {sorted(allowed_sources)}, got unknown values {unknown}."
        )
    sources = cast(tuple[ConditionSource, ...], tuple(raw_sources))

    if condition_weights is None:
        weights = tuple(1.0 for _ in sources)
    else:
        weights = tuple(float(value) for value in condition_weights)
        if len(weights) != len(sources):
            raise ValueError(
                f"condition_weights length {len(weights)} does not match condition_sources length {len(sources)}."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError(f"condition_weights must be finite and non-negative, got {weights!r}.")
        if sum(weights) <= 0.0:
            raise ValueError("condition_weights must contain at least one positive value.")
    return sources, weights


class DROIDLeRobotSubgoalDataset(DROIDLeRobotDataset):
    """Isolated DROID dataset for subgoal-conditioned training modes.

    The defaults intentionally match the currently verified DROID WAM recipe:
    ``policy`` mode, 32-step horizon, raw ``joint_pos`` actions, and a prepended
    state row.  All base-DROID constructor arguments remain overrideable.  New
    arguments only affect this class:

    Args:
        training_mode: ``mid_soft`` for the existing P/subtask path, ``wam``
            for standalone WAM action/video training, or ``reasoner_wam`` for
            one of the two homogeneous streams used by the joint
            Reasoner+WAM experiment.
        reasoner_wam_stream: Required for ``training_mode='reasoner_wam'``;
            choose ``reasoner`` or ``wam``.  This is a record-schema choice,
            not a third training mode.
        target_frame_offset: Explicitly override the target frame offset. When
            omitted, ``subgoal_frame_alignment='action'`` follows the commanded
            action timeline (offset 1 for the legacy delayed local convention,
            offset 0 when ``use_same_timestep_joint_action=True``). It is always
            emitted in sample provenance.
        reasoner_history_offsets: One or more non-positive offsets relative to
            the current frame. ``(-1, 0)`` gives a causal two-image view. At an
            episode boundary, ``repeat_first`` duplicates frame zero instead of
            crossing into the previous episode.
        reasoner_history_choices: Optional weighted choices for the Reasoner
            image count. For example ``((0,), (-16, -8, 0))`` mixes one image
            and three images while keeping the action window unchanged.
        reasoner_history_choice_weights: Non-negative weights for
            ``reasoner_history_choices``; the default is equal weight.
        vit_frame_offsets: One or more non-positive offsets used for the
            optional Qwen3-VL image context.  The last offset must be ``0``;
            for example ``(-8, -4, 0)`` gives frames 24, 28, and 32 when the
            action window starts at frame 32.  This context never changes the
            action target window.
        condition_sources: WAM-only condition sources.  WAM streams default to
            task:gold-subgoal weights of 2:8.  The mid-soft and Reasoner streams
            always use the task instruction as their input.
            ``gold_subgoal_lap`` is available only for WAM and adds the compact
            LAP-style EEF cue described above.
        lap_control_include_translation: Include qualitative forward/back,
            left/right, and up/down directions in ``gold_subgoal_lap``.
        lap_control_include_rotation: Include roll/pitch/yaw direction and
            rounded angle in ``gold_subgoal_lap``.
        lap_control_rotation_precision: Rotation rounding in degrees, normally
            10 to match LAP ``verbose_eef_with_rotation``.
        lap_control_min_translation_cm: Per-axis displacement threshold below
            which a qualitative LAP translation cue is omitted.
        lap_control_min_rotation_degrees: Per-axis angular threshold below
            which a LAP rotation cue is omitted. If every cue is below its
            threshold, ``gold_subgoal_lap`` stays identical to ``gold_subgoal``.
        lap_force_condition_rotation_degrees: When set and any absolute
            roll/pitch/yaw displacement is strictly greater than this value,
            force ``gold_subgoal_lap`` instead of using the ordinary weighted
            condition draw. This has no effect unless that source is configured.
        rotation_aware_subgoal_degrees: When set, Reasoner targets and WAM gold
            conditions share one deterministic label. Only roll/pitch/yaw axes
            strictly above this threshold are described, without translation,
            and the rotation is expressed as the means to the effective
            fine-first subgoal. Incompatible with ``gold_subgoal_lap``.
        max_subtasks_per_window: Maximum number of effective semantic subtask
            labels allowed in one target window after the five-frame boundary
            trim and adjacent-equivalent-label collapse. Windows above the
            limit are removed from the dataset index.
        object_position_augment_prob: Probability of adding an annotation-backed
            object-relative-position clause to a WAM gold-subgoal condition.
            The default is zero for exact legacy behavior.
        object_position_augment_style: ``random`` chooses one of the stable
            natural-language templates; concrete values are ``toward``,
            ``lead``, and ``structured``.
        object_position_augment_seed: Seed used with the stable sample-id hash
            for probability and template selection.
        open_gripper_move_augment: When true, rewrite every generated Reasoner
            target phrase ``Move the open gripper`` as ``Open the gripper and
            move the gripper``. The default is false.
        open_gripper_single_subtask_augment: When true, apply the alternative
            ``Open the gripper, next move the open gripper`` rewrite only when
            the resolved window contains exactly one subtask. The default is
            false and this rule takes precedence over the general rewrite.
        apply_subtask_episode_filter_to_windows: Keep the legacy filtered
            window index when true (default).  Set false for label-only WAM:
            all parent windows remain available, while filtered or incomplete
            keep-range windows receive no gold subgoal and fall back to task.
        include_idle_stationary_subtasks: Keep otherwise valid ``Idle`` and
            ``Remain stationary`` annotation ranges. The default is false so
            Reasoner/VLM and legacy datasets continue dropping them.
        remove_post_completion_windows: When true, remove only windows whose
            complete observation/video frame range intersects a group after
            the first task_completed group. Other six-rule exclusions remain
            in the parent index and fall back to the task instruction. Requires
            ``subtask_episode_filter_path``.
        include_post_completion_move_away_windows: Opt-in exception that adds
            only accepted ``Move away ...`` groups after task completion back
            to the filtered training index.
    """

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        action_space: str = "joint_pos",
        use_state: bool = True,
        use_same_timestep_joint_action: bool = False,
        action_normalization: str | None = None,
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
        precise_subtask_annotations_path: str | None = None,
        subtask_episode_filter_path: str | None = None,
        training_mode: TrainingMode = "mid_soft",
        reasoner_wam_stream: ReasonerWAMStream | None = None,
        subgoal_frame_alignment: SubgoalFrameAlignment = "action",
        target_frame_offset: int | None = None,
        reasoner_history_offsets: Sequence[int] = (0,),
        reasoner_history_boundary: HistoryBoundary = "repeat_first",
        reasoner_history_choices: Sequence[Sequence[int]] | None = None,
        reasoner_history_choice_weights: Sequence[float] | None = None,
        reasoner_history_choice_seed: int = 0,
        reasoner_view_mode: Literal["concat", "separate"] = "concat",
        reasoner_image_long_edge: int | None = None,
        vit_frame_offsets: Sequence[int] = (0,),
        vit_frame_offset_choices: Sequence[Sequence[int]] | None = None,
        vit_frame_choice_seed: int = 0,
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
        subtask_label_source: Literal["effective", "coarse"] = "effective",
        max_subtasks_per_window: int = 100,
        max_subtasks_count_segments: bool = False,
        subtask_boundary_trim_frames: int = 5,
        pregrasp_boundary_trim_frames: int = 0,
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
        task_variant_policy: Literal["random", "first", "deterministic", "object_match"] = "first",
        include_idle_stationary_subtasks: bool = False,
        apply_subtask_episode_filter_to_windows: bool = True,
        remove_post_completion_windows: bool = False,
        include_post_completion_move_away_windows: bool = False,
        include_execution_success: bool = False,
        include_task_instruction_context: bool = False,
        execution_success_only: bool = False,
        episode_split: Literal["all", "train", "val"] = "all",
        validation_fraction: float = 0.02,
        validation_split_seed: int = 43,
        episode_indices: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if training_mode not in ("mid_soft", "reasoner_wam", "wam"):
            raise ValueError(f"training_mode must be 'mid_soft', 'wam', or 'reasoner_wam', got {training_mode!r}.")
        if reasoner_wam_stream not in (None, "reasoner", "wam"):
            raise ValueError(f"reasoner_wam_stream must be None, 'reasoner', or 'wam', got {reasoner_wam_stream!r}.")
        if training_mode in ("mid_soft", "wam") and reasoner_wam_stream is not None:
            raise ValueError("reasoner_wam_stream is only valid with training_mode='reasoner_wam'.")
        if training_mode == "reasoner_wam" and reasoner_wam_stream is None:
            raise ValueError("training_mode='reasoner_wam' requires reasoner_wam_stream='reasoner' or 'wam'.")
        if reasoner_view_mode not in ("concat", "separate"):
            raise ValueError(f"reasoner_view_mode must be 'concat' or 'separate', got {reasoner_view_mode!r}.")
        if reasoner_image_long_edge is not None and (
            isinstance(reasoner_image_long_edge, bool) or int(reasoner_image_long_edge) < 1
        ):
            raise ValueError("reasoner_image_long_edge must be positive when set.")
        is_reasoner_stream = training_mode == "reasoner_wam" and reasoner_wam_stream == "reasoner"
        # Standalone ``wam`` has the same action/video record contract as the
        # old joint WAM stream, but deliberately does not require a stream
        # selector and never emits a Reasoner conversation.
        is_wam_stream = training_mode == "wam" or (training_mode == "reasoner_wam" and reasoner_wam_stream == "wam")
        if training_mode not in ("reasoner_wam", "wam") and condition_sources is not None:
            raise ValueError("condition_sources is only valid for a WAM training mode.")
        if is_reasoner_stream and condition_sources is not None:
            normalized_requested_sources = tuple(str(value).strip().lower() for value in condition_sources)
            if normalized_requested_sources != ("task",):
                raise ValueError("The Reasoner stream always uses the task instruction as its input.")
        normalized_same_timestep = _as_bool(use_same_timestep_joint_action, default=False)
        action_frame_offset = (
            0 if normalized_same_timestep and action_space == "joint_pos" else (1 if action_space == "joint_pos" else 0)
        )
        resolved_alignment, resolved_target_offset = _validate_target_offset(
            subgoal_frame_alignment=subgoal_frame_alignment,
            target_frame_offset=target_frame_offset,
            action_frame_offset=action_frame_offset,
        )
        history_offsets = _validate_history_offsets(reasoner_history_offsets)
        if isinstance(reasoner_history_choice_seed, bool) or not isinstance(reasoner_history_choice_seed, int):
            raise TypeError(f"reasoner_history_choice_seed must be an integer, got {reasoner_history_choice_seed!r}.")
        if reasoner_history_choices is None:
            history_choices = None
            history_choice_weights = None
        else:
            history_choices = _validate_history_offset_choices(reasoner_history_choices)
            assert history_choices is not None
            history_choice_weights = _normalize_history_choice_weights(
                reasoner_history_choice_weights,
                choice_count=len(history_choices),
            )
        vit_frame_offsets = _validate_history_offsets(vit_frame_offsets)
        vit_frame_offset_choices = _validate_history_offset_choices(vit_frame_offset_choices)
        if vit_frame_offset_choices is not None and vit_frame_offsets != (0,):
            raise ValueError("Set either vit_frame_offsets or vit_frame_offset_choices, not both.")
        vit_offset_candidates = vit_frame_offset_choices or (vit_frame_offsets,)
        if reasoner_history_boundary not in ("repeat_first", "error"):
            raise ValueError(
                f"reasoner_history_boundary must be 'repeat_first' or 'error', got {reasoner_history_boundary!r}."
            )
        if (
            any(offset < 0 for offset in history_offsets)
            or any(offset < 0 for choice in vit_offset_candidates for offset in choice)
        ) and use_image_augmentation:
            raise ValueError(
                "Causal image history and use_image_augmentation=True would sample separate random image "
                "augmentation parameters. Disable CPU image augmentation; the CUDA augmentation callback "
                "shares one transform across the action video and ViT history."
            )
        if isinstance(condition_seed, bool) or not isinstance(condition_seed, int):
            raise TypeError(f"condition_seed must be an integer, got {condition_seed!r}.")
        if isinstance(object_position_augment_prob, bool) or not isinstance(object_position_augment_prob, (float, int)):
            raise TypeError(
                f"object_position_augment_prob must be a finite number in [0, 1], got {object_position_augment_prob!r}."
            )
        object_position_augment_prob = float(object_position_augment_prob)
        if not math.isfinite(object_position_augment_prob) or not 0.0 <= object_position_augment_prob <= 1.0:
            raise ValueError(
                f"object_position_augment_prob must be a finite number in [0, 1], got {object_position_augment_prob!r}."
            )
        normalized_object_position_style = str(object_position_augment_style).strip().lower()
        if normalized_object_position_style not in {
            "random",
            "toward",
            "lead",
            "structured",
            "embedded",
        }:
            raise ValueError(
                "object_position_augment_style must be 'random', 'toward', 'lead', 'structured', or 'embedded', "
                f"got {object_position_augment_style!r}."
            )
        if isinstance(object_position_augment_seed, bool) or not isinstance(object_position_augment_seed, int):
            raise TypeError(f"object_position_augment_seed must be an integer, got {object_position_augment_seed!r}.")
        if isinstance(vit_frame_choice_seed, bool) or not isinstance(vit_frame_choice_seed, int):
            raise TypeError(f"vit_frame_choice_seed must be an integer, got {vit_frame_choice_seed!r}.")
        if task_variant_policy not in ("random", "first", "deterministic", "object_match"):
            raise ValueError(
                "task_variant_policy must be 'random', 'first', 'deterministic', or "
                f"'object_match', got {task_variant_policy!r}."
            )
        if (
            reasoner_history_boundary == "error"
            and any(offset < 0 for offset in history_offsets)
            and not is_reasoner_stream
        ):
            raise ValueError(
                "reasoner_history_boundary='error' with negative history offsets is only valid for the Reasoner stream."
            )

        default_sources = ("task", "gold_subgoal") if is_wam_stream else ("task",)
        default_weights = (2.0, 8.0) if is_wam_stream else None
        resolved_sources, resolved_weights = _normalize_condition_config(
            condition_sources=default_sources if condition_sources is None else condition_sources,
            condition_weights=default_weights if condition_weights is None else condition_weights,
        )
        if rotation_aware_subgoal_degrees is not None and "gold_subgoal_lap" in resolved_sources:
            raise ValueError(
                "rotation_aware_subgoal_degrees cannot be combined with gold_subgoal_lap; "
                "use gold_subgoal so the rotation is added exactly once."
            )

        self._training_mode = training_mode
        self._reasoner_wam_stream = reasoner_wam_stream
        self._is_reasoner_stream = is_reasoner_stream
        self._is_wam_stream = is_wam_stream
        self._subgoal_frame_alignment = resolved_alignment
        self._target_frame_offset = resolved_target_offset
        self._window_config_id = f"fps{float(fps):g}-L{int(chunk_length)}-offset{resolved_target_offset}"
        self._reasoner_history_offsets = history_offsets
        self._reasoner_history_boundary = reasoner_history_boundary
        self._reasoner_history_choices = history_choices
        self._reasoner_history_choice_weights = history_choice_weights
        self._reasoner_history_choice_seed = reasoner_history_choice_seed
        self._reasoner_view_mode = str(reasoner_view_mode)
        self._reasoner_image_long_edge = None if reasoner_image_long_edge is None else int(reasoner_image_long_edge)
        self._vit_frame_offsets = vit_frame_offsets
        self._vit_frame_offset_choices = vit_frame_offset_choices
        self._vit_frame_choice_seed = vit_frame_choice_seed
        self._condition_sources = resolved_sources
        self._condition_weights = resolved_weights
        self._condition_seed = condition_seed
        self._lap_control_include_translation = bool(lap_control_include_translation)
        self._lap_control_include_rotation = bool(lap_control_include_rotation)
        if isinstance(lap_control_rotation_precision, bool) or not isinstance(lap_control_rotation_precision, int):
            raise TypeError("lap_control_rotation_precision must be an integer")
        if lap_control_rotation_precision <= 0:
            raise ValueError("lap_control_rotation_precision must be positive")
        self._lap_control_rotation_precision = int(lap_control_rotation_precision)
        if (
            isinstance(lap_control_min_translation_cm, bool)
            or not isinstance(lap_control_min_translation_cm, (float, int))
            or not math.isfinite(float(lap_control_min_translation_cm))
            or float(lap_control_min_translation_cm) < 0
        ):
            raise ValueError("lap_control_min_translation_cm must be a finite non-negative number")
        if (
            isinstance(lap_control_min_rotation_degrees, bool)
            or not isinstance(lap_control_min_rotation_degrees, (float, int))
            or not math.isfinite(float(lap_control_min_rotation_degrees))
            or float(lap_control_min_rotation_degrees) < 0
        ):
            raise ValueError("lap_control_min_rotation_degrees must be a finite non-negative number")
        self._lap_control_min_translation_cm = float(lap_control_min_translation_cm)
        self._lap_control_min_rotation_degrees = float(lap_control_min_rotation_degrees)
        if lap_force_condition_rotation_degrees is not None and (
            isinstance(lap_force_condition_rotation_degrees, bool)
            or not isinstance(lap_force_condition_rotation_degrees, (float, int))
            or not math.isfinite(float(lap_force_condition_rotation_degrees))
            or float(lap_force_condition_rotation_degrees) < 0
        ):
            raise ValueError("lap_force_condition_rotation_degrees must be None or a finite non-negative number")
        self._lap_force_condition_rotation_degrees = (
            None if lap_force_condition_rotation_degrees is None else float(lap_force_condition_rotation_degrees)
        )
        if rotation_aware_subgoal_degrees is not None and (
            isinstance(rotation_aware_subgoal_degrees, bool)
            or not isinstance(rotation_aware_subgoal_degrees, (float, int))
            or not math.isfinite(float(rotation_aware_subgoal_degrees))
            or float(rotation_aware_subgoal_degrees) < 0
        ):
            raise ValueError("rotation_aware_subgoal_degrees must be None or a finite non-negative number")
        self._rotation_aware_subgoal_degrees = (
            None if rotation_aware_subgoal_degrees is None else float(rotation_aware_subgoal_degrees)
        )
        if subtask_label_source not in ("effective", "coarse"):
            raise ValueError(
                "subtask_label_source must be 'effective' or 'coarse', "
                f"got {subtask_label_source!r}."
            )
        self._subtask_label_source = subtask_label_source
        if (
            isinstance(max_subtasks_per_window, bool)
            or not isinstance(max_subtasks_per_window, int)
            or max_subtasks_per_window < 1
        ):
            raise ValueError("max_subtasks_per_window must be a positive integer")
        self._max_subtasks_per_window = int(max_subtasks_per_window)
        self._max_subtasks_count_segments = _as_bool(max_subtasks_count_segments, default=False)
        if isinstance(subtask_boundary_trim_frames, bool) or not isinstance(subtask_boundary_trim_frames, int) or subtask_boundary_trim_frames < 0:
            raise ValueError("subtask_boundary_trim_frames must be a non-negative integer")
        if isinstance(contact_label_delay_frames, bool) or not isinstance(contact_label_delay_frames, int) or contact_label_delay_frames < 0:
            raise ValueError("contact_label_delay_frames must be a non-negative integer")
        if isinstance(release_label_advance_frames, bool) or not isinstance(release_label_advance_frames, int) or release_label_advance_frames < 0:
            raise ValueError("release_label_advance_frames must be a non-negative integer")
        if (
            isinstance(pregrasp_boundary_trim_frames, bool)
            or not isinstance(pregrasp_boundary_trim_frames, int)
            or pregrasp_boundary_trim_frames < 0
        ):
            raise ValueError("pregrasp_boundary_trim_frames must be a non-negative integer")
        self._subtask_boundary_trim_frames = int(subtask_boundary_trim_frames)
        self._pregrasp_boundary_trim_frames = int(pregrasp_boundary_trim_frames)
        self._contact_label_delay_frames = int(contact_label_delay_frames)
        self._release_label_advance_frames = int(release_label_advance_frames)
        self._force_move_away_followup = bool(force_move_away_followup)
        self._drop_standalone_move_away = bool(drop_standalone_move_away)
        self._include_preview_action_in_prompt = bool(include_preview_action_in_prompt)
        self._include_execution_success = _as_bool(
            include_execution_success, default=False
        )
        # Keep the complete episode instruction as an optional JSON-prompt
        # field for WAM.  The default is deliberately off for exact legacy
        # prompt compatibility.
        self._include_task_instruction_context = _as_bool(
            include_task_instruction_context, default=False
        )
        self._execution_success_only = _as_bool(execution_success_only, default=False)
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
        self._object_position_augment_prob = object_position_augment_prob
        self._object_position_augment_style = normalized_object_position_style
        self._object_position_augment_seed = object_position_augment_seed
        self._open_gripper_move_augment = _as_bool(open_gripper_move_augment, default=False)
        self._open_gripper_single_subtask_augment = _as_bool(open_gripper_single_subtask_augment, default=False)
        self._task_variant_policy = task_variant_policy
        normalized_precise_path = (
            str(precise_subtask_annotations_path).strip()
            if precise_subtask_annotations_path is not None
            else ""
        )
        if normalized_precise_path and training_mode not in ("wam", "reasoner_wam"):
            raise ValueError("precise_subtask_annotations_path is only valid for WAM or Reasoner training modes.")
        self._precise_subtask_points: dict[int, PreciseSubtaskEpisode] = {}
        self._precise_subtask_anchor_cache: dict[tuple[int, int], tuple[str, str] | None] = {}
        self._precise_subtask_override_count = 0
        super().__init__(
            root=root,
            episode_indices=episode_indices,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_space=action_space,
            use_state=use_state,
            use_same_timestep_joint_action=normalized_same_timestep,
            action_normalization=action_normalization,
            use_image_augmentation=use_image_augmentation,
            use_filter_dict=use_filter_dict,
            filter_dict_path=filter_dict_path,
            filter_dict_key_mode=filter_dict_key_mode,
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
            episode_split=self._episode_split,
            validation_fraction=self._validation_fraction,
            validation_split_seed=self._validation_split_seed,
            include_cartesian_state_for_language=(
                (is_wam_stream and "gold_subgoal_lap" in resolved_sources)
                or self._rotation_aware_subgoal_degrees is not None
            ),
        )
        if normalized_precise_path:
            if self._subtask_annotations is None:
                raise ValueError("precise_subtask_annotations_path requires subtask_annotations_path as base labels.")
            self._precise_subtask_points = load_precise_subtask_points(normalized_precise_path)
            if _should_log_dataset_summary():
                point_count = sum(len(points.starts) for points in self._precise_subtask_points.values())
                print(
                    "[droid_lerobot_subgoal_dataset] precise subgoal overrides: "
                    f"episodes={len(self._precise_subtask_points)}, points={point_count}, "
                    f"chunk_length={chunk_length}, boundary_trim={self._subtask_boundary_trim_frames}, "
                    f"pregrasp_boundary_trim={self._pregrasp_boundary_trim_frames}, "
                    f"contact_delay={self._contact_label_delay_frames}, "
                    f"release_advance={self._release_label_advance_frames}"
                )

        # Keep provenance and annotation lookup on the same command timeline as
        # the base dataset's actual joint action rows.
        self._action_frame_offset = (
            0
            if self._use_same_timestep_joint_action and action_space == "joint_pos"
            else (1 if action_space == "joint_pos" else 0)
        )
        # Keep the physical annotation map for execution-outcome filtering;
        # semantic Reasoner labels may use a delayed contact boundary.
        self._execution_subtask_annotations = self._subtask_annotations
        if self._subtask_label_source == "coarse":
            # Refined annotation rows keep the original group wording in
            # ``coarse_subtask``.  Relabel only the semantic map used for
            # prompt composition/counting; physical outcomes remain attached
            # to the untouched annotation map above.
            self._subtask_annotations = relabel_subtask_annotations(
                self._subtask_annotations or {},
                label_source="coarse",
            )
        if (is_reasoner_stream or is_wam_stream) and self._contact_label_delay_frames:
            self._subtask_annotations = delay_contact_subtask_boundaries(
                self._subtask_annotations or {},
                delay_frames=self._contact_label_delay_frames,
            )
        if (is_reasoner_stream or is_wam_stream) and self._release_label_advance_frames:
            self._subtask_annotations = advance_release_subtask_boundaries(
                self._subtask_annotations or {},
                advance_frames=self._release_label_advance_frames,
            )
        # A WAM configuration without the task source must never silently fall
        # back to an instruction. In that mode, also remove incomplete/gap
        # windows while applying the requested semantic-subtask count limit.
        require_subgoal = is_wam_stream and "task" not in resolved_sources
        if (
            require_subgoal
            or self._max_subtasks_per_window < int(chunk_length)
            or self._include_execution_success
            or self._execution_success_only
            or self._episode_split != "all"
        ):
            self._apply_subtask_count_window_filter(require_subgoal=require_subgoal)

    def _apply_subtask_count_window_filter(self, *, require_subgoal: bool) -> None:
        """Compact the parent index to runs satisfying the subtask-count rule."""

        outcome_stats = {
            "candidate_success": 0,
            "candidate_failure": 0,
            "candidate_invalid": 0,
            "kept_success": 0,
            "kept_failure": 0,
            "filtered_outcome": 0,
            "filtered_subtask_limit": 0,
        }

        if getattr(self, "_use_segment_window_filter", False):
            segment_ends = np.asarray(self._seg_cum, dtype=np.int64)
            segment_bases = np.concatenate((np.zeros(1, dtype=np.int64), segment_ends[:-1]))
            source_segments = zip(
                np.asarray(self._seg_ep_pos, dtype=np.int64),
                np.asarray(self._seg_win_start, dtype=np.int64),
                segment_ends - segment_bases,
                strict=True,
            )
        else:
            valid_ends = np.asarray(self._valid_cum, dtype=np.int64)
            valid_bases = np.concatenate((np.zeros(1, dtype=np.int64), valid_ends[:-1]))
            source_segments = (
                (episode_position, 0, length) for episode_position, length in enumerate(valid_ends - valid_bases)
            )

        annotations_by_episode = getattr(self, "_subtask_annotations", {})
        execution_annotations_by_episode = getattr(
            self, "_execution_subtask_annotations", annotations_by_episode
        )
        kept_episode_positions: list[int] = []
        kept_window_starts: list[int] = []
        kept_lengths: list[int] = []
        for episode_position_value, window_start_value, length_value in source_segments:
            episode_position = int(episode_position_value)
            window_start = int(window_start_value)
            length = int(length_value)
            if length <= 0:
                continue
            episode_index = int(self._ep_vals[episode_position])
            is_validation = _episode_is_validation(
                episode_index=episode_index,
                validation_fraction=getattr(self, "_validation_fraction", 0.02),
                seed=getattr(self, "_validation_split_seed", 43),
            )
            episode_split = getattr(self, "_episode_split", "all")
            if episode_split == "train" and is_validation:
                continue
            if episode_split == "val" and not is_validation:
                continue
            local_starts = np.arange(window_start, window_start + length, dtype=np.int64)
            target_starts = local_starts + int(self._target_frame_offset)
            counts = _effective_subtask_counts_for_windows(
                annotations_by_episode.get(episode_index),
                target_starts=target_starts,
                target_ends=target_starts + int(self._chunk_length),
                boundary_trim_frames=getattr(
                    self,
                    "_subtask_boundary_trim_frames",
                    5,
                ),
                pregrasp_boundary_trim_frames=getattr(
                    self,
                    "_pregrasp_boundary_trim_frames",
                    0,
                ),
                count_annotation_segments=getattr(self, "_max_subtasks_count_segments", False),
            )
            keep = counts <= self._max_subtasks_per_window
            if require_subgoal:
                keep &= counts > 0
            include_execution_success = getattr(self, "_include_execution_success", False)
            execution_success_only = getattr(self, "_execution_success_only", False)
            if include_execution_success or execution_success_only:
                outcomes = _uniform_execution_success_for_windows(
                    execution_annotations_by_episode.get(episode_index),
                    target_starts=target_starts,
                    target_ends=target_starts + int(self._chunk_length),
                )
                outcome_stats["candidate_success"] += int(np.count_nonzero(outcomes == 1))
                outcome_stats["candidate_failure"] += int(np.count_nonzero(outcomes == 0))
                outcome_stats["candidate_invalid"] += int(np.count_nonzero(outcomes < 0))
                keep &= outcomes == 1 if execution_success_only else outcomes >= 0
                kept_outcomes = outcomes[keep]
                outcome_stats["kept_success"] += int(np.count_nonzero(kept_outcomes == 1))
                outcome_stats["kept_failure"] += int(np.count_nonzero(kept_outcomes == 0))
                outcome_stats["filtered_outcome"] += int(np.count_nonzero(outcomes < 0))
                outcome_stats["filtered_subtask_limit"] += int(
                    np.count_nonzero((outcomes >= 0) & ~keep)
                )
            padded = np.concatenate((np.zeros(1, dtype=bool), keep, np.zeros(1, dtype=bool)))
            transitions = np.flatnonzero(padded[1:] != padded[:-1])
            for run_start, run_end in transitions.reshape(-1, 2):
                kept_episode_positions.append(episode_position)
                kept_window_starts.append(window_start + int(run_start))
                kept_lengths.append(int(run_end - run_start))

        self._seg_ep_pos = np.asarray(kept_episode_positions, dtype=np.int64)
        self._seg_win_start = np.asarray(kept_window_starts, dtype=np.int64)
        self._seg_cum = np.cumsum(kept_lengths, dtype=np.int64) if kept_lengths else np.zeros(0, dtype=np.int64)
        self._use_segment_window_filter = True
        self._execution_window_stats = outcome_stats
        if (
            getattr(self, "_include_execution_success", False)
            or getattr(self, "_execution_success_only", False)
        ) and _should_log_dataset_summary():
            print(
                "[droid_lerobot_subgoal_dataset] execution outcome windows: "
                f"success_only={getattr(self, '_execution_success_only', False)}, "
                f"kept_success={outcome_stats['kept_success']}, "
                f"kept_failure={outcome_stats['kept_failure']}, "
                f"candidate_success={outcome_stats['candidate_success']}, "
                f"candidate_failure={outcome_stats['candidate_failure']}, "
                f"filtered_mixed_uncovered_unknown={outcome_stats['filtered_outcome']}, "
                f"filtered_subtask_limit={outcome_stats['filtered_subtask_limit']}"
            )

    @property
    def training_mode(self) -> TrainingMode:
        return self._training_mode

    @property
    def reasoner_wam_stream(self) -> ReasonerWAMStream | None:
        return self._reasoner_wam_stream

    @property
    def target_frame_offset(self) -> int:
        return self._target_frame_offset

    @property
    def subgoal_frame_alignment(self) -> SubgoalFrameAlignment:
        return self._subgoal_frame_alignment

    @property
    def apply_subtask_episode_filter_to_windows(self) -> bool:
        """Whether the subtask filter trims this dataset's window index."""

        return bool(getattr(self, "_apply_subtask_episode_filter_to_windows", True))

    def _raw_length(self) -> int:
        return DROIDLeRobotDataset.__len__(self)

    def _raw_index(self, idx: int) -> int:
        public_length = len(self)
        idx = int(idx)
        if idx < 0:
            idx += public_length
        if idx < 0 or idx >= public_length:
            raise IndexError(f"DROID subgoal dataset index {idx} is out of range for length {public_length}.")
        return idx

    def _resolve_raw_window(self, raw_idx: int) -> DROIDSubgoalWindow:
        raw_length = self._raw_length()
        if raw_idx < 0 or raw_idx >= raw_length:
            raise IndexError(f"Raw DROID index {raw_idx} is out of range for length {raw_length}.")

        use_segment_window_filter = getattr(
            self,
            "_use_segment_window_filter",
            getattr(self, "_use_filter_dict", False),
        )
        if use_segment_window_filter:
            segment_position = int(np.searchsorted(self._seg_cum, raw_idx, side="right"))
            segment_base = int(self._seg_cum[segment_position - 1]) if segment_position > 0 else 0
            episode_position = int(self._seg_ep_pos[segment_position])
            global_start = (
                int(self._ep_starts[episode_position])
                + int(self._seg_win_start[segment_position])
                + (raw_idx - segment_base)
            )
        else:
            episode_position = int(np.searchsorted(self._valid_cum, raw_idx, side="right"))
            episode_base = int(self._valid_cum[episode_position - 1]) if episode_position > 0 else 0
            global_start = int(self._ep_starts[episode_position]) + (raw_idx - episode_base)

        episode_index = int(self._ep_vals[episode_position])
        episode_global_start = int(self._ep_starts[episode_position])
        episode_global_end = (
            int(self._ep_starts[episode_position + 1])
            if episode_position + 1 < len(self._ep_starts)
            else len(self._row_episode)
        )
        episode_frame_count = episode_global_end - episode_global_start
        window_start = global_start - episode_global_start
        window_end = window_start + self._chunk_length + 1
        action_frame_offset = getattr(
            self,
            "_action_frame_offset",
            1 if getattr(self, "_action_space", "joint_pos") == "joint_pos" else 0,
        )
        action_start = window_start + action_frame_offset
        action_end = action_start + self._chunk_length
        target_frame_offset = getattr(self, "_target_frame_offset", 0)
        target_start = window_start + target_frame_offset
        target_end = target_start + self._chunk_length
        if target_start < 0 or target_end > episode_frame_count:
            raise ValueError(
                f"Resolved target [{target_start}, {target_end}) is outside episode_index={episode_index} "
                f"with {episode_frame_count} frames."
            )
        return DROIDSubgoalWindow(
            raw_index=raw_idx,
            episode_position=episode_position,
            episode_index=episode_index,
            global_start=global_start,
            episode_global_start=episode_global_start,
            episode_frame_count=episode_frame_count,
            window_start_frame=window_start,
            window_end_frame=window_end,
            action_start_frame=action_start,
            action_end_frame=action_end,
            target_start_frame=target_start,
            target_end_frame=target_end,
        )

    def _lookup_subgoal(self, window: DROIDSubgoalWindow) -> _SubgoalLookup:
        # Label-only WAM keeps every baseline window, but filtered clips,
        # annotation gaps, and episodes absent from the six-rule filter must
        # use the task instruction. Validate the same action-aligned interval
        # that supplies the gold text. In the legacy delayed joint_pos
        # convention this is
        # ``[s+1, s+33)``; with same-timestep actions it is ``[s, s+32)``.
        # The preview observation at ``s`` is not itself an executable action
        # and must not make a boundary window fall back to instruction.
        if not self._subtask_window_is_label_eligible(
            episode_index=window.episode_index,
            window_start_frame=window.target_start_frame,
            window_end_frame=window.target_end_frame,
        ):
            return _SubgoalLookup(text="", status="subtask_filter_excluded", segments=())
        annotations: dict[int, EpisodeSubtaskAnnotations] | None = getattr(self, "_subtask_annotations", None)
        if annotations is None:
            return _SubgoalLookup(text="", status="annotations_disabled", segments=())
        episode_annotations = annotations.get(window.episode_index)
        if episode_annotations is None:
            return _SubgoalLookup(text="", status="missing_episode", segments=())

        event_mode = any(bool(segment.source_subtask) for segment in episode_annotations.segments)
        contributing = resolve_subtask_segments(
            annotations,
            episode_index=window.episode_index,
            action_start_frame=window.target_start_frame,
            action_end_frame=window.target_end_frame,
            # A few metadata-only tools construct this object with
            # ``__new__`` and only populate annotation state. Preserve the
            # historical five-frame rule for those callers.
            boundary_trim_frames=(0 if event_mode else getattr(self, "_subtask_boundary_trim_frames", 5)),
            pregrasp_boundary_trim_frames=(0 if event_mode else getattr(self, "_pregrasp_boundary_trim_frames", 0)),
        )
        # Event-enriched annotations explicitly use null event_caption rows
        # as gaps.  Never fall back to a task instruction for those windows.
        if any(segment.source_subtask and segment.event_caption is None for segment in contributing):
            return _SubgoalLookup(text="", status="event_caption_missing", segments=())
        if event_mode:
            if not contributing:
                return _SubgoalLookup(text="", status="event_caption_missing", segments=())
            active = contributing[0]
            if active.event_caption is None or active.event_id is None:
                return _SubgoalLookup(text="", status="event_caption_missing", segments=())
            active_index = int(
                np.searchsorted(
                    np.asarray(episode_annotations.starts, dtype=np.int64),
                    window.target_start_frame,
                    side="right",
                )
                - 1
            )
            event_end_frame = _contiguous_event_end_frames(episode_annotations.segments)[active_index]
            event_coverage = min(event_end_frame, window.target_end_frame) - window.target_start_frame
            if event_coverage < _EVENT_MIN_START_COVERAGE_FRAMES:
                return _SubgoalLookup(text="", status="event_tail_too_short", segments=())
            if str(active.source_subtask or "").strip().casefold() in {"idle", "remain stationary", "stationary"}:
                return _SubgoalLookup(text="", status="idle_stationary", segments=())
            return _SubgoalLookup(text=active.event_caption, status="available", segments=(active,))
        # Idle/stationary source labels are kept in the event-enriched file so
        # they retain event provenance, but their windows are not executable
        # event supervision.
        if any(
            str(segment.source_subtask or "").strip().casefold()
            in {"idle", "remain stationary", "stationary"}
            for segment in contributing
        ):
            return _SubgoalLookup(text="", status="idle_stationary", segments=())
        if getattr(self, "_force_move_away_followup", False) and contributing:
            first_label = " ".join(contributing[0].subtask.split()).casefold()
            if first_label.startswith("move away") and len(contributing) == 1:
                episode_segments = annotations.get(window.episode_index)
                next_segment: SubtaskAnnotation | None = None
                if episode_segments is not None:
                    last_end = contributing[-1].end_frame
                    for candidate in episode_segments.segments:
                        if candidate.start_frame < last_end:
                            continue
                        if candidate.start_frame != last_end:
                            break
                        candidate_label = " ".join(candidate.subtask.split()).casefold()
                        if candidate_label in {"idle", "remain stationary", "unknown"}:
                            last_end = candidate.end_frame
                            continue
                        if candidate_label != first_label:
                            next_segment = candidate
                            break
                if next_segment is not None:
                    next_label = " ".join(next_segment.subtask.split()).casefold()
                    # Move-away windows immediately before grasp are useful
                    # single-step avoidance examples; do not expose the
                    # contact action as a forced future label.
                    if not next_label.startswith("grasp"):
                        contributing = (*contributing, next_segment)
                elif getattr(self, "_drop_standalone_move_away", False):
                    return _SubgoalLookup(text="", status="standalone_move_away", segments=())
        elif getattr(self, "_drop_standalone_move_away", False) and contributing:
            first_label = " ".join(contributing[0].subtask.split()).casefold()
            if first_label.startswith("move away") and len(contributing) == 1:
                return _SubgoalLookup(text="", status="standalone_move_away", segments=())
        # Contact-boundary delay and forced Move-away look-ahead can expose a
        # third distinct label even when the original index passed the limit.
        # Never emit a three-step Reasoner target.
        distinct_labels: list[str] = []
        for segment in contributing:
            key = " ".join(segment.subtask.split()).casefold()
            if not distinct_labels or distinct_labels[-1] != key:
                distinct_labels.append(key)
        if len(distinct_labels) > getattr(self, "_max_subtasks_per_window", 100):
            return _SubgoalLookup(text="", status="subtask_limit", segments=())
        text = compose_subtask_text(contributing)
        if not text:
            return _SubgoalLookup(text="", status="coverage_gap", segments=())
        return _SubgoalLookup(text=text, status="available", segments=contributing)

    @staticmethod
    def _normalize_override_text(value: str) -> str:
        return " ".join(str(value).split()).strip().rstrip(".!?。！？").casefold()

    def _precise_anchor_base(self, episode_index: int, start_frame: int) -> tuple[str, str] | None:
        """Resolve one precise point using the exact WAM boundary rules."""

        key = (int(episode_index), int(start_frame))
        if key in self._precise_subtask_anchor_cache:
            return self._precise_subtask_anchor_cache[key]
        if not self._precise_subtask_points or self._subtask_annotations is None:
            self._precise_subtask_anchor_cache[key] = None
            return None
        points = self._precise_subtask_points.get(int(episode_index))
        if points is None:
            self._precise_subtask_anchor_cache[key] = None
            return None
        try:
            precise_index = points.starts.index(int(start_frame))
        except ValueError:
            self._precise_subtask_anchor_cache[key] = None
            return None
        contributing = resolve_subtask_segments(
            self._subtask_annotations,
            episode_index=int(episode_index),
            action_start_frame=int(start_frame),
            action_end_frame=int(start_frame) + int(self._chunk_length),
            boundary_trim_frames=int(self._subtask_boundary_trim_frames),
            pregrasp_boundary_trim_frames=int(self._pregrasp_boundary_trim_frames),
        )
        # If the precise point is in a leading/trailing fragment removed by
        # WAM's boundary rules, it cannot serve as an override anchor.
        if not contributing or not any(
            segment.start_frame <= int(start_frame) < segment.end_frame for segment in contributing
        ):
            self._precise_subtask_anchor_cache[key] = None
            return None
        base_text = compose_subtask_text(contributing)
        precise_text = points.subtasks[precise_index]
        result = (base_text, precise_text) if base_text and precise_text else None
        self._precise_subtask_anchor_cache[key] = result
        return result

    def _precise_subgoal_at_window_start(self, window: DROIDSubgoalWindow) -> str | None:
        """Return the precise label anchored exactly at this input frame."""

        if not self._precise_subtask_points:
            return None
        anchor = self._precise_anchor_base(window.episode_index, window.window_start_frame)
        return anchor[1] if anchor is not None else None

    def _precise_subgoal_for_window(
        self,
        window: DROIDSubgoalWindow,
        subgoal: _SubgoalLookup,
        *,
        count_override: bool = True,
    ) -> str | None:
        """Return precise text for overlapping windows with the same base label."""

        if not self._precise_subtask_points:
            return None
        points = self._precise_subtask_points.get(int(window.episode_index))
        if points is None:
            return None
        # An exact precise point is authoritative, even when the base lookup
        # is unavailable or uses different wording. The neighboring +/-31
        # rule remains conservative and requires an identical base text.
        exact_text = self._precise_subgoal_at_window_start(window)
        if exact_text is not None:
            if count_override:
                self._precise_subtask_override_count += 1
            return exact_text
        if not subgoal.available or not subgoal.text:
            return None
        chunk = int(self._chunk_length)
        left = int(window.target_start_frame) - chunk + 1
        right = int(window.target_start_frame) + chunk - 1
        base_key = self._normalize_override_text(subgoal.text)
        candidates: list[tuple[int, int, str]] = []
        for precise_start, _ in points.in_range(left, right):
            anchor = self._precise_anchor_base(window.episode_index, precise_start)
            if anchor is None:
                continue
            anchor_base, precise_text = anchor
            if self._normalize_override_text(anchor_base) == base_key:
                candidates.append(
                    (abs(int(precise_start) - int(window.target_start_frame)), int(precise_start), precise_text)
                )
        if not candidates:
            return None
        candidates.sort(key=lambda value: (value[0], value[1]))
        nearest_distance = candidates[0][0]
        nearest = [value for value in candidates if value[0] == nearest_distance]
        if len({self._normalize_override_text(value[2]) for value in nearest}) != 1:
            return None
        if count_override:
            self._precise_subtask_override_count += 1
        return nearest[0][2]

    def reasoner_target_for_metadata(self, window: DROIDSubgoalWindow, subgoal: _SubgoalLookup) -> str:
        """Return the final Reasoner target without decoding images."""

        precise_text = self._precise_subgoal_for_window(window, subgoal, count_override=False)
        target = precise_text if precise_text is not None else subgoal.text
        if getattr(self, "_open_gripper_single_subtask_augment", False) and len(subgoal.segments) == 1:
            target = _rewrite_single_move_open_gripper_text(target)
        elif getattr(self, "_open_gripper_move_augment", False):
            target = _rewrite_move_open_gripper_text(target)
        return target

    def precise_override_raw_indices(self) -> tuple[int, ...]:
        """Return valid raw windows anchored by precise labels."""

        if not self._precise_subtask_points:
            return ()
        indices: list[int] = []
        for start, length in self.get_shuffle_blocks():
            for raw_index in range(int(start), int(start) + int(length)):
                window, _subgoal = self._metadata_for_raw_index(int(raw_index))
                # Exact precise points are authoritative; unlike +/-31
                # propagation they do not require the coarse/fine base text
                # to match. Boundary-invalid points still remain excluded.
                if self._precise_subgoal_at_window_start(window) is None:
                    continue
                indices.append(int(raw_index))
        return tuple(indices)

    def _execution_success_for_window(
        self, window: DROIDSubgoalWindow
    ) -> bool | None:
        if not getattr(self, "_include_execution_success", False):
            return None
        annotations = getattr(self, "_execution_subtask_annotations", getattr(self, "_subtask_annotations", {})).get(
            window.episode_index
        )
        value = _uniform_execution_success_for_windows(
            annotations,
            target_starts=np.asarray([window.target_start_frame], dtype=np.int64),
            target_ends=np.asarray([window.target_end_frame], dtype=np.int64),
        )[0]
        if int(value) < 0:
            raise RuntimeError(
                f"Execution-outcome filtered window became mixed or uncovered: {window.sample_id}"
            )
        return bool(value)

    def primary_subgoal(self, idx: int) -> str:
        """Return the first (boundary-trimmed) subgoal label for a window.

        This lightweight helper is intended for deterministic WAM sampling;
        it performs no video decode and returns an empty string whenever the
        window has no eligible gold subgoal.
        """

        raw_idx = self._raw_index(idx)
        _, subgoal = self._metadata_for_raw_index(raw_idx)
        return subgoal.segments[0].subtask if subgoal.available and subgoal.segments else ""

    def is_primary_grip_window(self, idx: int) -> bool:
        """Whether a window's first usable subgoal is a grip/grasp action."""

        label = _normalize_text(self.primary_subgoal(idx)).casefold()
        return label.startswith(("grip", "grasp", "pick up", "pick-up"))

    def _metadata_for_raw_index(self, raw_idx: int) -> tuple[DROIDSubgoalWindow, _SubgoalLookup]:
        window = self._resolve_raw_window(raw_idx)
        return window, self._lookup_subgoal(window)

    def window_metadata(self, idx: int) -> dict[str, Any]:
        """Return Python-scalar provenance without decoding video."""

        raw_idx = self._raw_index(idx)
        window, subgoal = self._metadata_for_raw_index(raw_idx)
        return {
            "schema_version": _SCHEMA_VERSION,
            "sample_id": window.sample_id,
            "raw_index": window.raw_index,
            "episode_index": window.episode_index,
            "episode_frame_count": window.episode_frame_count,
            "window_start_frame": window.window_start_frame,
            "window_end_frame": window.window_end_frame,
            "observation_frame": window.window_start_frame,
            "action_start_frame": window.action_start_frame,
            "action_end_frame": window.action_end_frame,
            "target_start_frame": window.target_start_frame,
            "target_end_frame": window.target_end_frame,
            "target_frame_offset": self._target_frame_offset,
            "window_config_id": getattr(self, "_window_config_id", ""),
            "subgoal_alignment": self._subgoal_frame_alignment,
            "subgoal": subgoal.text,
            "subgoal_available": subgoal.available,
            "subgoal_status": subgoal.status,
            "subgoal_segment_count": len(subgoal.segments),
            "subgoal_annotation_start_frame": subgoal.segments[0].start_frame if subgoal.segments else -1,
            "subgoal_annotation_end_frame": subgoal.segments[-1].end_frame if subgoal.segments else -1,
        }

    def _condition_draw(self, window: DROIDSubgoalWindow) -> ConditionSource:
        if len(self._condition_sources) == 1:
            return self._condition_sources[0]
        digest = hashlib.blake2b(
            f"{self._condition_seed}:{window.sample_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        draw = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
        total = sum(self._condition_weights)
        threshold = draw * total
        cumulative = 0.0
        for source, weight in zip(self._condition_sources, self._condition_weights, strict=True):
            cumulative += weight
            if threshold < cumulative:
                return source
        return self._condition_sources[-1]

    def _vit_offsets_for_window(self, window: DROIDSubgoalWindow) -> tuple[int, ...]:
        choices = self._vit_frame_offset_choices
        if choices is None:
            return self._vit_frame_offsets
        digest = hashlib.blake2b(
            f"{self._vit_frame_choice_seed}:{window.sample_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        choice_index = int.from_bytes(digest, byteorder="big", signed=False) % len(choices)
        return choices[choice_index]

    def _select_wam_condition(
        self,
        *,
        window: DROIDSubgoalWindow,
        task_text: str,
        subgoal: _SubgoalLookup,
    ) -> str:
        """Return the WAM text input: task or gold subgoal, with task fallback.

        Object-position augmentation is deliberately applied only after the
        source draw selects an available gold subgoal.  The stored
        ``subgoal.text`` (and therefore Reasoner CE supervision) is never
        modified; only this WAM conditioning string can receive the optional
        annotation-backed context.
        """

        if not subgoal.available:
            if "task" in self._condition_sources:
                return task_text
            raise RuntimeError(
                f"Window {window.sample_id} has no valid subgoal, but the configured WAM condition sources "
                "contain no task/instruction fallback."
            )
        precise_text = self._precise_subgoal_for_window(window, subgoal)
        precise_override = precise_text is not None
        if precise_text is not None:
            # WAM-only precise text; retain base segments for boundary,
            # execution, and locator metadata.
            subgoal = _SubgoalLookup(
                text=precise_text,
                status="precise_override",
                segments=subgoal.segments,
            )
        source = self._condition_draw(window)
        if precise_override:
            # Precise annotations are the final labels.  Do not add any
            # position, LAP, or rotation-aware language on top of them.
            return precise_text if source in {"gold_subgoal", "gold_subgoal_lap"} else task_text
        source, lap_action = self._condition_source_and_lap_action(window)
        if source == "gold_subgoal" and subgoal.available:
            return self._rotation_aware_subgoal_text(
                window=window,
                subgoal=subgoal,
            )
        if source == "gold_subgoal_lap" and subgoal.available:
            return self._maybe_append_lap_control(
                window=window,
                subgoal=subgoal,
                action=lap_action,
            )
        return task_text

    def _eef_action_for_window(self, window: DROIDSubgoalWindow) -> np.ndarray:
        feature = "observation.state.cartesian_position"
        states = getattr(self, "_feat", {}).get(feature)
        if states is None:
            raise RuntimeError(
                "rotation/LAP language requires observation.state.cartesian_position; "
                "the dataset was not initialized with include_cartesian_state_for_language."
            )
        local_states = np.asarray(
            states[window.episode_global_start : window.episode_global_start + window.episode_frame_count],
            dtype=np.float64,
        )
        return eef_window_action(
            local_states,
            (window.target_start_frame, window.target_end_frame),
        )

    def _condition_source_and_lap_action(
        self,
        window: DROIDSubgoalWindow,
    ) -> tuple[ConditionSource, np.ndarray | None]:
        """Resolve the weighted source, forcing LAP for configured large rotations."""

        source = self._condition_draw(window)
        force_threshold = getattr(self, "_lap_force_condition_rotation_degrees", None)
        if force_threshold is None or "gold_subgoal_lap" not in self._condition_sources:
            return source, None
        action = self._eef_action_for_window(window)
        rotation_degrees = np.abs(np.rad2deg(action[3:6]))
        if bool(np.any(rotation_degrees > force_threshold)):
            source = "gold_subgoal_lap"
        return source, action

    def _maybe_append_lap_control(
        self,
        *,
        window: DROIDSubgoalWindow,
        subgoal: _SubgoalLookup,
        action: np.ndarray | None = None,
        apply_object_position_augment: bool = True,
    ) -> str:
        """Prefix qualitative EEF directions and rotation before the gold subgoal."""

        base_text = self._maybe_augment_wam_subgoal(
            window=window,
            subgoal=subgoal,
            enabled=apply_object_position_augment,
        )
        control = format_lap_control(
            self._eef_action_for_window(window) if action is None else action,
            include_translation=self._lap_control_include_translation,
            include_rotation=self._lap_control_include_rotation,
            rotation_precision=self._lap_control_rotation_precision,
            min_translation_cm=self._lap_control_min_translation_cm,
            min_rotation_degrees=self._lap_control_min_rotation_degrees,
        )
        return append_lap_control(base_text, control)

    def _rotation_aware_subgoal_text(
        self,
        *,
        window: DROIDSubgoalWindow,
        subgoal: _SubgoalLookup,
        apply_object_position_augment: bool = True,
    ) -> str:
        """Return one shared fine-first target for Reasoner and WAM."""

        base_text = self._maybe_augment_wam_subgoal(
            window=window,
            subgoal=subgoal,
            enabled=apply_object_position_augment,
        )
        threshold = getattr(self, "_rotation_aware_subgoal_degrees", None)
        if threshold is None:
            return base_text
        control = format_rotation_control(
            self._eef_action_for_window(window),
            min_rotation_degrees=threshold,
            rotation_precision=10,
        )
        return append_rotation_purpose(base_text, control)

    def _maybe_augment_wam_subgoal(
        self,
        *,
        window: DROIDSubgoalWindow,
        subgoal: _SubgoalLookup,
        enabled: bool = True,
    ) -> str:
        """Apply stable, opt-in object-locator augmentation to a WAM label."""

        if not enabled:
            return subgoal.text
        probability = float(getattr(self, "_object_position_augment_prob", 0.0))
        if probability <= 0.0 or not subgoal.available or not subgoal.segments:
            return subgoal.text

        seed = int(getattr(self, "_object_position_augment_seed", 42))
        digest = hashlib.blake2b(
            f"object-position-prob:{seed}:{window.sample_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        draw = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
        if draw >= probability:
            return subgoal.text

        configured_style = str(getattr(self, "_object_position_augment_style", "random")).strip().lower()
        if configured_style == "random":
            style_digest = hashlib.blake2b(
                f"object-position-style:{seed}:{window.sample_id}".encode("utf-8"),
                digest_size=8,
            ).digest()
            # Random training mixtures use prose-only variants.  The structured
            # template remains available as an explicit experiment setting.
            configured_style = ("toward", "lead", "embedded")[
                int.from_bytes(style_digest, byteorder="big", signed=False) % 3
            ]
        # A window can contain a leading avoidance step followed by the actual
        # approach/move step.  Keep the leading action verbatim and augment the
        # first later segment that satisfies the same object/action guard used
        # by the standalone augmentation helper.
        augment_index = next(
            (
                index
                for index, segment in enumerate(subgoal.segments)
                if object_position_augmentation_parts(segment) is not None
            ),
            None,
        )
        if augment_index is None:
            return subgoal.text
        if augment_index == 0 and configured_style != "embedded":
            return augment_subtask_with_object_position(
                subgoal.text,
                segment=subgoal.segments[0],
                style=configured_style,
            )

        augmented_segment = augment_subtask_with_object_position(
            subgoal.segments[augment_index].subtask,
            segment=subgoal.segments[augment_index],
            style=configured_style,
        )

        def continuation(text: str) -> str:
            normalized = " ".join(str(text).split()).strip()
            if len(normalized) > 1 and normalized[0].isupper() and normalized[1].islower():
                return normalized[0].lower() + normalized[1:]
            return normalized

        parts: list[str] = []
        before = compose_subtask_text(subgoal.segments[:augment_index]).strip().rstrip(".!?")
        if before:
            parts.append(before)
        # Keep the window-opening action capitalized; only lowercase genuine
        # continuations after a leading segment.
        parts.append(augmented_segment if not parts else continuation(augmented_segment))
        after = compose_subtask_text(subgoal.segments[augment_index + 1 :]).strip()
        if after:
            parts.append(continuation(after))
        return ", then ".join(parts)

    def _task_text(self, *, window: DROIDSubgoalWindow, base_sample: dict[str, Any]) -> str:
        """Choose the task instruction used as the actual model input.

        The legacy parent samples a variant with process-global ``random``.  A
        new sample id must not silently receive a different Reasoner prompt on
        every access, so this wrapper defaults to the first non-empty variant.
        ``object_match`` selects the first variant naming the current annotation
        object and falls back to the first variant. ``random`` remains available
        when augmentation parity with the parent is desired, and
        ``deterministic`` hashes the sample id.
        """

        task_index = int(self._row_task[window.global_start])
        full_task = str(self._tasks.get(task_index, ""))
        variants = [_normalize_text(part) for part in full_task.split(" | ")]
        variants = [variant for variant in variants if variant]
        if not variants:
            return _normalize_text(base_sample.get("ai_caption", ""))
        task_variant_policy = getattr(self, "_task_variant_policy", "first")
        if task_variant_policy == "first":
            return variants[0]
        if task_variant_policy == "object_match":
            # DROID task paraphrases occasionally disagree on a color/object
            # noun (for example, "orange plushy" vs "yellow plush toy").
            # Prefer the first variant that names the current annotation
            # object so rationale supervision does not use a contradictory
            # instruction. Fall back to the historical first variant when no
            # reliable object match exists.
            annotations = getattr(self, "_subtask_annotations", {}).get(window.episode_index)
            if annotations is not None:
                target_frame = int(window.target_start_frame)
                for segment in annotations.segments:
                    if segment.start_frame <= target_frame < segment.end_frame:
                        object_name = _normalize_text(segment.object_name).casefold()
                        if object_name and object_name not in {"unknown", "object"}:
                            matching = [variant for variant in variants if object_name in variant.casefold()]
                            if matching:
                                return matching[0]
                        break
            return variants[0]
        if task_variant_policy == "random":
            sampled = _normalize_text(base_sample.get("ai_caption", ""))
            return sampled if sampled in variants else variants[0]
        digest = hashlib.blake2b(window.sample_id.encode("utf-8"), digest_size=8).digest()
        variant_index = int.from_bytes(digest, byteorder="big", signed=False) % len(variants)
        return variants[variant_index]

    def _preview_action_text(self, window: DROIDSubgoalWindow) -> str:
        """Return the last completed annotation before the preview frame.

        This is causal history, unlike the future target.  At inference time
        the caller can replace it with the previously executed/predicted
        action text; keeping the field optional preserves the legacy prompt.
        """

        annotations = getattr(self, "_execution_subtask_annotations", None)
        episode = annotations.get(window.episode_index) if annotations is not None else None
        if episode is None:
            return ""
        previous = [segment for segment in episode.segments if segment.end_frame <= window.target_start_frame]
        return previous[-1].subtask if previous else ""

    def _build_reasoner_images(
        self,
        *,
        window: DROIDSubgoalWindow,
        base_sample: dict[str, Any],
        requested_offsets: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return self._build_frame_context(
            window=window,
            base_sample=base_sample,
            requested_offsets=(
                self._reasoner_offsets_for_window(window) if requested_offsets is None else requested_offsets
            ),
        )

    def _reasoner_offsets_for_window(self, window: DROIDSubgoalWindow) -> tuple[int, ...]:
        """Choose one Reasoner history shape without changing any target index.

        The choice is a deterministic hash of the sample id, so worker/rank
        ordering cannot change the 1-image/3-image assignment.  Only this
        choice is randomized; frame-boundary resolution below is deterministic.
        """

        choices = getattr(self, "_reasoner_history_choices", None)
        if choices is None:
            return self._reasoner_history_offsets
        weights = getattr(self, "_reasoner_history_choice_weights", None)
        if weights is None:
            weights = tuple(1.0 for _ in choices)
        digest = hashlib.blake2b(
            f"{getattr(self, '_reasoner_history_choice_seed', 0)}:{window.sample_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        draw = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
        threshold = draw * sum(weights)
        cumulative = 0.0
        for offsets, weight in zip(choices, weights, strict=True):
            cumulative += weight
            if threshold < cumulative:
                return offsets
        return choices[-1]

    @staticmethod
    def _resolve_history_positions(
        *,
        window: DROIDSubgoalWindow,
        requested_offsets: Sequence[int],
    ) -> tuple[list[int], bool]:
        """Resolve causal history frames, smoothly shrinking a 3-frame span.

        For the default ``(-16, -8, 0)`` choice, a current frame ``6`` cannot
        provide a full sixteen-frame look-back.  We therefore sample the same
        number of points over the available prefix: ``[0, 3, 6]``.  The
        integer ceiling gives the requested ``[0, 1, 1]`` and ``[0, 0, 0]``
        behavior for current frames 1 and 0.  Once the full span is available,
        the requested offsets are used verbatim.
        """

        current = int(window.window_start_frame)
        offsets = tuple(int(offset) for offset in requested_offsets)
        if not offsets:
            raise ValueError("Reasoner history offsets must not be empty.")
        max_lookback = max(-offset for offset in offsets)
        if max_lookback <= current:
            return [current + offset for offset in offsets], False
        if len(offsets) == 1:
            return [0], current != 0

        # Preserve the relative positions of arbitrary causal choices while
        # fitting their span into [0, current].  The default 16/8/0 choices
        # reduce exactly to the examples in the user-facing launcher notes.
        positions: list[int] = []
        for offset in offsets:
            relative = max_lookback + offset  # oldest offset -> 0
            position = math.ceil(current * relative / max_lookback)
            positions.append(min(current, max(0, position)))
        return positions, True

    def _build_frame_context(
        self,
        *,
        window: DROIDSubgoalWindow,
        base_sample: dict[str, Any],
        requested_offsets: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Decode causal context frames without changing the action window.

        The base sample already contains the preview frame at ``window.start``.
        Historical ViT/Reasoner frames are decoded separately so WAM still sees
        exactly its original ``[start, start + chunk_length]`` video and action
        supervision remains anchored at ``action_start_frame``.
        """
        if requested_offsets == (0,):
            video = base_sample.get("video")
            if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[1] < 1:
                raise ValueError("Reasoner view requires base sample video with shape [C,T,H,W] and T >= 1.")
            if getattr(self, "_reasoner_view_mode", "concat") == "separate":
                rows = [self._window_rows(window.episode_global_start + window.window_start_frame,
                                          window.episode_global_start + window.window_start_frame + 1,
                                          window.episode_index)[0]]
                images = self._load_separate_reasoner_views(
                    self._episodes[window.episode_index], rows, long_edge=getattr(self, "_reasoner_image_long_edge", None)
                )
                images = (images * 255.0).clamp(0.0, 255.0).to(torch.uint8).contiguous()
            else:
                images = video[:, :1].permute(1, 0, 2, 3).contiguous()
            frames = torch.tensor([window.window_start_frame], dtype=torch.long)
            return images, frames, False

        actual_local_frames, clamped = self._resolve_history_positions(
            window=window,
            requested_offsets=requested_offsets,
        )
        if clamped and self._reasoner_history_boundary == "error":
            raise IndexError(
                "Reasoner history reaches before the episode start: "
                f"episode_index={window.episode_index}, window_start_frame={window.window_start_frame}, "
                f"requested_offsets={tuple(requested_offsets)!r}."
            )
        actual_global_positions = [window.episode_global_start + position for position in actual_local_frames]

        rows = [
            self._window_rows(position, position + 1, window.episode_index)[0] for position in actual_global_positions
        ]
        if getattr(self, "_reasoner_view_mode", "concat") == "separate":
            images = self._load_separate_reasoner_views(
                self._episodes[window.episode_index], rows, long_edge=getattr(self, "_reasoner_image_long_edge", None)
            )
            frames = torch.tensor(actual_local_frames, dtype=torch.long)
            return (images * 255.0).clamp(0.0, 255.0).to(torch.uint8).contiguous(), frames, clamped
        raw_images = self._load_concat_video(self._episodes[window.episode_index], rows)
        if raw_images.ndim != 4 or raw_images.shape[0] != len(rows):
            raise ValueError(
                "Expected causal Reasoner history decode with shape [N,C,H,W], "
                f"got {tuple(raw_images.shape)} for N={len(rows)}."
            )
        images = (raw_images * 255.0).clamp(0.0, 255.0).to(torch.uint8).contiguous()
        frames = torch.tensor(actual_local_frames, dtype=torch.long)
        return images, frames, clamped

    def _build_reasoner_record(
        self,
        *,
        window: DROIDSubgoalWindow,
        subgoal: _SubgoalLookup,
        sample: dict[str, Any],
        task_text: str,
        rationale_reason: str | None = None,
        rationale_subtask: str | None = None,
        requested_history_offsets: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Build one ordinary or start-frame rationale Reasoner record."""

        precise_text = self._precise_subgoal_for_window(window, subgoal)
        if precise_text is not None:
            # Precise annotations are final labels.  They intentionally bypass
            # the synthetic rotation-aware suffix and object-position
            # augmentation used by the legacy Reasoner target.
            reasoner_target = precise_text
        else:
            reasoner_target = (
                self._rotation_aware_subgoal_text(window=window, subgoal=subgoal)
                if getattr(self, "_rotation_aware_subgoal_degrees", None) is not None and subgoal.available
                else subgoal.text
            )
        if rationale_reason is not None or rationale_subtask is not None:
            if rationale_reason is None or rationale_subtask is None:
                raise ValueError("rationale_reason and rationale_subtask must be provided together.")
            reasoner_target = format_reasoner_rationale_target(
                reason=rationale_reason,
                subtask=rationale_subtask,
            )
        if getattr(self, "_open_gripper_single_subtask_augment", False) and len(subgoal.segments) == 1:
            reasoner_target = _rewrite_single_move_open_gripper_text(reasoner_target)
        elif getattr(self, "_open_gripper_move_augment", False):
            reasoner_target = _rewrite_move_open_gripper_text(reasoner_target)
        precise_target_available = precise_text is not None
        reasoner_images, reasoner_frames, _ = self._build_reasoner_images(
            window=window,
            base_sample=sample,
            requested_offsets=requested_history_offsets,
        )
        pil_images = [_tensor_to_pil(image) for image in reasoner_images]
        image_count = len(pil_images)
        # Preserve the legacy one-frame prompt. Preview context is only
        # meaningful when the model receives the multi-frame history.
        preview_action = (
            self._preview_action_text(window=window)
            if getattr(self, "_include_preview_action_in_prompt", False) and image_count > 1
            else None
        )
        reasoner_input_text = format_action_prediction_prompt(
            task_text,
            preview_action=preview_action,
            separate_views=getattr(self, "_reasoner_view_mode", "concat") == "separate",
            require_think=rationale_reason is not None,
        )
        if getattr(self, "_reasoner_view_mode", "concat") == "separate":
            # Make temporal/view semantics explicit in the multimodal prompt.
            # The image order itself remains LEFT, RIGHT, WRIST within each
            # group.  Label each temporal group once, rather than adding a
            # label before every individual image.
            group_count = len(reasoner_frames)
            group_names = ["CURRENT"] if group_count == 1 else ["PREV"] * (group_count - 1) + ["CURRENT"]
            image_placeholders = ""
            for group_name in group_names:
                image_placeholders += f"{group_name}:\n"
                image_placeholders += "<image>\n" * 3
        else:
            image_placeholders = "".join("<image>\n" for _ in range(image_count))
        record = {
            "conversations": [
                {"from": "human", "value": image_placeholders + reasoner_input_text},
                {"from": "gpt", "value": reasoner_target},
            ],
            # This is consumed by DROIDReasonerSupervisionProcessor to mask
            # missing/gap annotations or an empty task without changing
            # dataset length. There is no fallback target.
            "reasoner_ce_available": torch.tensor(
                bool(task_text.strip())
                and (subgoal.available or precise_target_available or rationale_subtask is not None),
                dtype=torch.bool,
            ),
        }
        # Keep the legacy one-image key for old processors and use an explicit
        # list only when a sample is selected for multi-image VLM supervision.
        record["image" if image_count == 1 else "images"] = pil_images[0] if image_count == 1 else pil_images
        return record

    def reasoner_rationale_item(
        self,
        idx: int,
        *,
        reason: str,
        subtask: str,
        history_offsets: Sequence[int],
    ) -> dict[str, Any]:
        """Build rationale SFT at one existing raw window start frame.

        ``history_offsets`` comes from the annotation sidecar so training sees
        exactly the same causal images that were available to the rationale
        writer. It changes only the image context, never the start frame.
        """

        if not self._is_reasoner_stream:
            raise RuntimeError("reasoner_rationale_item is only valid for the Reasoner stream.")
        requested_offsets = _validate_history_offsets(history_offsets)
        raw_idx = self._raw_index(idx)
        window, subgoal = self._metadata_for_raw_index(raw_idx)
        sample = DROIDLeRobotDataset.__getitem__(self, raw_idx)
        task_text = self._task_text(window=window, base_sample=sample)
        return self._build_reasoner_record(
            window=window,
            subgoal=subgoal,
            sample=sample,
            task_text=task_text,
            rationale_reason=reason,
            rationale_subtask=subtask,
            requested_history_offsets=requested_offsets,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw_idx = self._raw_index(idx)
        window, subgoal = self._metadata_for_raw_index(raw_idx)
        sample = DROIDLeRobotDataset.__getitem__(self, raw_idx)
        task_text = self._task_text(window=window, base_sample=sample)

        if self._is_reasoner_stream:
            return self._build_reasoner_record(
                window=window,
                subgoal=subgoal,
                sample=sample,
                task_text=task_text,
            )

        if self._is_wam_stream:
            sample["ai_caption"] = self._select_wam_condition(
                window=window,
                task_text=task_text,
                subgoal=subgoal,
            )
            if getattr(self, "_include_task_instruction_context", False):
                sample["task_instruction"] = task_text
            # The WAM stream has flow targets only; it must not carry the
            # subtask decoder target that the parent exposes when annotations
            # are loaded.
            sample.pop("subtask", None)
        else:
            sample["ai_caption"] = task_text
            sample["subtask"] = subgoal.text

        if getattr(self, "_include_execution_success", False):
            sample["execution_success"] = self._execution_success_for_window(window)

        # Standalone WAM intentionally exposes only the established video and
        # action tensors plus its selected caption.  ViT history belongs to the
        # optional joint Reasoner/WAM path; ignoring it here prevents an
        # accidental Qwen/VLM processor payload when ``training_mode='wam'``.
        vit_frame_offsets = self._vit_offsets_for_window(window)
        if self._training_mode != "wam" and (self._vit_frame_offset_choices is not None or vit_frame_offsets != (0,)):
            vit_images, _, _ = self._build_frame_context(
                window=window,
                base_sample=sample,
                requested_offsets=vit_frame_offsets,
            )
            sample.update(
                {
                    "vit_images": vit_images,
                    "vit_frame_offsets": torch.tensor(vit_frame_offsets, dtype=torch.long),
                }
            )

        return sample

    def __len__(self) -> int:
        return self._raw_length()

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return DROIDLeRobotDataset.get_shuffle_blocks(self)


__all__ = [
    "ConditionSource",
    "DROIDLeRobotSubgoalDataset",
    "DROIDSubgoalWindow",
    "HistoryBoundary",
    "ReasonerWAMStream",
    "SubgoalFrameAlignment",
    "TrainingMode",
]
