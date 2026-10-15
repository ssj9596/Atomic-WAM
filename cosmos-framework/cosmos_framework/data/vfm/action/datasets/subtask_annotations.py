# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load VLM2 episode subtasks and derive supervision for action windows."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MINIMAL_ANNOTATION_KEYS = {"episode_index", "start_frame", "end_frame", "subtask"}
_VLM2_ANNOTATION_FILENAME = "annotation.json"
_VLM2_DONE_FILENAME = "EPISODE_PASS2_DONE.json"
_TRAILING_SENTENCE_PUNCTUATION = ".!?。！？"
_SUBTASK_FILTER_DROP_LABELS = frozenset(("idle", "remain stationary"))
_SUBTASK_FILTER_MAX_GROUP_DURATION_FRAMES = 150
# A segment that contributes only a handful of frames at the edge of a
# prediction window is not useful as an executable plan step.  Keep this as a
# named, code-level default rather than a dataset-index filter: the window must
# remain in the dataset and only its text label is trimmed.
_SUBTASK_BOUNDARY_MAX_OVERLAP_FRAMES = 5
@dataclass(frozen=True, slots=True)
class SubtaskAnnotation:
    """One episode-local, half-open subtask segment."""

    episode_index: int
    start_frame: int
    end_frame: int
    subtask: str
    # Enriched VLM2 group metadata.  These fields deliberately have defaults
    # so the minimal four-column JSONL format and existing positional test
    # fixtures remain source-compatible.
    object_name: str = ""
    object_locator: str = ""
    target: str = ""
    # ``subtask`` is the effective training label.  Refined v3 rows retain the
    # original group label here for audit while replacing ``subtask`` with the
    # more precise ``fine_subtask`` text.
    coarse_subtask: str = field(default="", compare=False)
    fine_subtask: str = field(default="", compare=False)
    # Optional physical execution outcome produced by the offline checker.
    # ``None`` preserves compatibility with all historical annotation files.
    execution_success: bool | None = field(default=None, compare=False)
    # Optional goal-level event metadata.  Event annotations retain the
    # original refined-group text in ``source_subtask`` while ``subtask`` is
    # the executable event caption consumed by WAM.
    event_caption: str | None = field(default=None, compare=False)
    event_id: int | None = field(default=None, compare=False)
    source_subtask: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class EpisodeSubtaskAnnotations:
    """Sorted, non-overlapping subtask segments for one episode."""

    starts: tuple[int, ...]
    segments: tuple[SubtaskAnnotation, ...]


@dataclass(frozen=True, slots=True)
class EpisodeSubtaskFilter:
    """Episode-local clip ranges derived from flattened group annotations.

    ``keep_start_frame``/``keep_end_frame`` preserve compatibility with the
    legacy one-interval filter. Enriched annotations populate ``keep_ranges``
    so removed groups or clips remain real gaps in the training index.
    ``post_completion_ranges`` is a separate selective-removal view used by
    label-only WAM; it does not make Idle/long/object-unrelated clips vanish.
    """

    keep_start_frame: int
    keep_end_frame: int
    keep_ranges: tuple[tuple[int, int], ...] = ()
    # Ranges belonging to groups after the first task_completed group.  They
    # are kept separate from ``keep_ranges`` so a label-only WAM run can drop
    # post-completion windows while retaining Idle/long/object-unrelated
    # windows as task-instruction samples.
    post_completion_ranges: tuple[tuple[int, int], ...] = ()
    # Safe post-completion exception: object-related, short groups whose coarse
    # label starts with ``Move away``.  Dataset recipes can opt into only these
    # ranges without reopening every action after task completion.
    post_completion_move_away_ranges: tuple[tuple[int, int], ...] = ()


def _is_contact_subtask(segment: SubtaskAnnotation) -> bool:
    """Whether a segment is a grasp/close contact transition.

    Refined rows keep both the coarse and fine wording.  Checking both makes
    the delay robust to either annotation level while limiting it to the
    explicit contact verbs requested for Reasoner supervision.
    """

    for value in (segment.coarse_subtask, segment.fine_subtask, segment.subtask):
        text = " ".join(str(value).strip().split()).casefold()
        if text.startswith(("grasp", "close")):
            return True
    return False


def delay_contact_subtask_boundaries(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    delay_frames: int,
) -> dict[int, EpisodeSubtaskAnnotations]:
    """Delay the start of Grasp/Close segments while preserving coverage.

    A boundary is moved later by up to ``delay_frames`` and the preceding
    segment is extended by the same amount.  This keeps the episode intervals
    contiguous and leaves execution-outcome metadata untouched in the caller's
    original annotation map.  Very short contact segments are shifted by at
    most ``duration - 1`` so no segment becomes empty.
    """

    if isinstance(delay_frames, bool) or not isinstance(delay_frames, int) or delay_frames < 0:
        raise ValueError(f"delay_frames must be a non-negative integer, got {delay_frames!r}.")
    if delay_frames == 0:
        return annotations

    delayed: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index, episode in annotations.items():
        source = episode.segments
        if len(source) < 2:
            delayed[int(episode_index)] = episode
            continue
        starts = [segment.start_frame for segment in source]
        ends = [segment.end_frame for segment in source]
        for position in range(1, len(source)):
            segment = source[position]
            if not _is_contact_subtask(segment):
                continue
            duration = segment.end_frame - segment.start_frame
            shift = min(delay_frames, max(0, duration - 1))
            boundary = segment.start_frame + shift
            starts[position] = boundary
            ends[position - 1] = boundary
        shifted_segments = tuple(
            SubtaskAnnotation(
                episode_index=segment.episode_index,
                start_frame=starts[position],
                end_frame=ends[position],
                subtask=segment.subtask,
                object_name=segment.object_name,
                object_locator=segment.object_locator,
                target=segment.target,
                coarse_subtask=segment.coarse_subtask,
                fine_subtask=segment.fine_subtask,
                execution_success=segment.execution_success,
                event_caption=segment.event_caption,
                event_id=segment.event_id,
                source_subtask=segment.source_subtask,
            )
            for position, segment in enumerate(source)
        )
        delayed[int(episode_index)] = EpisodeSubtaskAnnotations(
            starts=tuple(starts),
            segments=shifted_segments,
        )
    return delayed


def _is_release_subtask(segment: SubtaskAnnotation) -> bool:
    """Whether a segment is an explicit release action."""

    for value in (segment.coarse_subtask, segment.fine_subtask, segment.subtask):
        text = " ".join(str(value).strip().split()).casefold()
        if text.startswith("release"):
            return True
    return False


def advance_release_subtask_boundaries(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    advance_frames: int,
) -> dict[int, EpisodeSubtaskAnnotations]:
    """Move explicit Release boundaries earlier while preserving coverage.

    For every Release segment that has a preceding segment, its start boundary
    is moved earlier by up to ``advance_frames``.  The preceding segment is
    shortened by the same amount, while at least one frame remains in that
    segment.  The returned map is semantic-only; callers keep the original
    annotation map for physical execution-outcome filtering.
    """

    if isinstance(advance_frames, bool) or not isinstance(advance_frames, int) or advance_frames < 0:
        raise ValueError(f"advance_frames must be a non-negative integer, got {advance_frames!r}.")
    if advance_frames == 0:
        return annotations

    advanced: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index, episode in annotations.items():
        source = episode.segments
        if len(source) < 2:
            advanced[int(episode_index)] = episode
            continue
        starts = [segment.start_frame for segment in source]
        ends = [segment.end_frame for segment in source]
        changed = False
        for position in range(1, len(source)):
            segment = source[position]
            if not _is_release_subtask(segment):
                continue
            # Use the current preceding start so consecutive Release segments
            # remain contiguous after each boundary is advanced.
            available_previous = segment.start_frame - starts[position - 1]
            shift = min(advance_frames, max(0, available_previous - 1))
            if shift <= 0:
                continue
            boundary = segment.start_frame - shift
            starts[position] = boundary
            ends[position - 1] = boundary
            changed = True
        if not changed:
            advanced[int(episode_index)] = episode
            continue
        shifted_segments = tuple(
            SubtaskAnnotation(
                episode_index=segment.episode_index,
                start_frame=starts[position],
                end_frame=ends[position],
                subtask=segment.subtask,
                object_name=segment.object_name,
                object_locator=segment.object_locator,
                target=segment.target,
                coarse_subtask=segment.coarse_subtask,
                fine_subtask=segment.fine_subtask,
                execution_success=segment.execution_success,
                event_caption=segment.event_caption,
                event_id=segment.event_id,
                source_subtask=segment.source_subtask,
            )
            for position, segment in enumerate(source)
        )
        advanced[int(episode_index)] = EpisodeSubtaskAnnotations(
            starts=tuple(starts),
            segments=shifted_segments,
        )
    return advanced


def advance_pregrasp_subtask_boundaries(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    max_move_frames: int,
) -> dict[int, EpisodeSubtaskAnnotations]:
    """Swallow short coarse Move/Approach -> Grasp boundaries.

    Only transitions whose preceding coarse segment lasts at most
    ``max_move_frames`` are changed.  The complete short Move/Approach segment
    is merged into the following Grasp/Close segment.  Longer Move/Approach
    segments are left untouched.  This is independent of the generic boundary
    trim and does not modify action targets or the physical execution-outcome
    annotation map.
    """

    if isinstance(max_move_frames, bool) or not isinstance(max_move_frames, int) or max_move_frames < 0:
        raise ValueError(f"max_move_frames must be a non-negative integer, got {max_move_frames!r}.")
    if max_move_frames == 0:
        return annotations

    def coarse_starts_with(segment: SubtaskAnnotation, prefixes: tuple[str, ...]) -> bool:
        text = " ".join(str(segment.coarse_subtask or segment.subtask).strip().split()).casefold()
        return text.startswith(prefixes)

    advanced: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index, episode in annotations.items():
        # Collapse contiguous rows with the same coarse label first.  Refined
        # exports can split one coarse Move segment into several fine rows.
        groups: list[SubtaskAnnotation] = []
        for segment in episode.segments:
            coarse = segment.coarse_subtask or segment.subtask
            previous_coarse = groups[-1].coarse_subtask or groups[-1].subtask if groups else ""
            if (
                groups
                and " ".join(str(previous_coarse).split()).casefold()
                == " ".join(str(coarse).split()).casefold()
                and groups[-1].end_frame == segment.start_frame
            ):
                previous = groups[-1]
                groups[-1] = SubtaskAnnotation(
                    episode_index=previous.episode_index,
                    start_frame=previous.start_frame,
                    end_frame=segment.end_frame,
                    subtask=coarse,
                    object_name=previous.object_name,
                    object_locator=previous.object_locator,
                    target=previous.target,
                    coarse_subtask=coarse,
                    fine_subtask="",
                    execution_success=previous.execution_success,
                    event_caption=previous.event_caption,
                    event_id=previous.event_id,
                    source_subtask=previous.source_subtask,
                )
            else:
                groups.append(
                    SubtaskAnnotation(
                        episode_index=segment.episode_index,
                        start_frame=segment.start_frame,
                        end_frame=segment.end_frame,
                        subtask=coarse,
                        object_name=segment.object_name,
                        object_locator=segment.object_locator,
                        target=segment.target,
                        coarse_subtask=coarse,
                        fine_subtask="",
                        execution_success=segment.execution_success,
                        event_caption=segment.event_caption,
                        event_id=segment.event_id,
                        source_subtask=segment.source_subtask,
                    )
                )

        shifted: list[SubtaskAnnotation] = []
        i = 0
        while i < len(groups):
            segment = groups[i]
            if i + 1 < len(groups):
                next_segment = groups[i + 1]
                coarse = " ".join(str(segment.coarse_subtask or segment.subtask).split()).casefold()
                duration = segment.end_frame - segment.start_frame
                # ``move away`` is deliberately excluded: it is not a
                # pre-grasp approach even though it begins with "move".
                is_move_or_approach = (coarse.startswith("approach") or coarse.startswith("move")) and not coarse.startswith("move away")
                is_grasp = coarse_starts_with(next_segment, ("close", "grasp"))
                if is_move_or_approach and is_grasp and duration <= max_move_frames:
                    shifted.append(
                        SubtaskAnnotation(
                            episode_index=next_segment.episode_index,
                            start_frame=segment.start_frame,
                            end_frame=next_segment.end_frame,
                            subtask=next_segment.subtask,
                            object_name=next_segment.object_name,
                            object_locator=next_segment.object_locator,
                            target=next_segment.target,
                            coarse_subtask=next_segment.coarse_subtask,
                            fine_subtask="",
                            execution_success=next_segment.execution_success,
                            event_caption=next_segment.event_caption,
                            event_id=next_segment.event_id,
                            source_subtask=next_segment.source_subtask,
                        )
                    )
                    i += 2
                    continue
            shifted.append(segment)
            i += 1
        advanced[int(episode_index)] = EpisodeSubtaskAnnotations(
            starts=tuple(segment.start_frame for segment in shifted),
            segments=tuple(shifted),
        )
    return advanced


def relabel_subtask_annotations(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    label_source: str,
) -> dict[int, EpisodeSubtaskAnnotations]:
    """Select the text label while preserving fine/coarse metadata."""

    normalized = str(label_source).strip().lower()
    if normalized not in {"effective", "coarse"}:
        raise ValueError(f"label_source must be 'effective' or 'coarse', got {label_source!r}.")
    if normalized == "effective":
        return annotations
    relabeled: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index, episode in annotations.items():
        segments = tuple(
            SubtaskAnnotation(
                episode_index=segment.episode_index,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                subtask=segment.coarse_subtask or segment.subtask,
                object_name=segment.object_name,
                object_locator=segment.object_locator,
                target=segment.target,
                coarse_subtask=segment.coarse_subtask,
                fine_subtask=segment.fine_subtask,
                execution_success=segment.execution_success,
                event_caption=segment.event_caption,
                event_id=segment.event_id,
                source_subtask=segment.source_subtask,
            )
            for segment in episode.segments
        )
        relabeled[int(episode_index)] = EpisodeSubtaskAnnotations(
            starts=tuple(segment.start_frame for segment in segments),
            segments=segments,
        )
    return relabeled


def _parse_nonnegative_json_int(value: Any, *, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: {field} must be a non-negative JSON integer, got {value!r}.")
    return value


def _canonicalize_subtask(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: subtask must be a non-empty string, got {value!r}.")
    subtask = " ".join(value.split()).rstrip(_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    if not subtask:
        raise ValueError(f"{location}: subtask must be a non-empty string, got {value!r}.")
    return subtask


def _canonicalize_optional_annotation_text(value: Any, *, field: str, location: str) -> str:
    """Normalize optional enriched annotation text without changing legacy rows.

    The merged VLM2 export always writes strings, but accepting a missing or
    JSON ``null`` value is useful for partially-produced episodes.  Unexpected
    optional types are treated as unavailable metadata: these fields are only
    an opt-in WAM augmentation hint and must never make an otherwise valid
    legacy annotation fail to load.
    """

    if value is None or not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip().rstrip(_TRAILING_SENTENCE_PUNCTUATION).rstrip()


def _parse_segment(
    row: Any,
    *,
    episode_index: int | None,
    location: str,
    require_exact_keys: bool,
) -> SubtaskAnnotation:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: expected a JSON object, got {type(row).__name__}.")
    keys = set(row)
    if require_exact_keys and keys != _MINIMAL_ANNOTATION_KEYS:
        missing = sorted(_MINIMAL_ANNOTATION_KEYS - keys)
        extra = sorted(keys - _MINIMAL_ANNOTATION_KEYS)
        raise ValueError(
            f"{location}: expected exactly {sorted(_MINIMAL_ANNOTATION_KEYS)}; missing={missing}, extra={extra}."
        )
    required_keys = _MINIMAL_ANNOTATION_KEYS if episode_index is None else _MINIMAL_ANNOTATION_KEYS - {"episode_index"}
    missing = sorted(required_keys - keys)
    if missing:
        raise ValueError(f"{location}: missing required subtask fields: {missing}.")

    row_episode_index = _parse_nonnegative_json_int(
        row["episode_index"] if episode_index is None else episode_index,
        field="episode_index",
        location=location,
    )
    start_frame = _parse_nonnegative_json_int(row["start_frame"], field="start_frame", location=location)
    end_frame = _parse_nonnegative_json_int(row["end_frame"], field="end_frame", location=location)
    if end_frame <= start_frame:
        raise ValueError(
            f"{location}: end_frame must be greater than start_frame for a half-open interval, "
            f"got [{start_frame}, {end_frame})."
        )
    source_subtask = _canonicalize_subtask(row["subtask"], location=location)
    is_event_enriched = "event_caption" in row
    event_caption_raw = row.get("event_caption")
    event_caption = None
    if event_caption_raw is not None:
        if not isinstance(event_caption_raw, str):
            raise ValueError(f"{location}: event_caption must be a string or null, got {event_caption_raw!r}.")
        normalized_caption = " ".join(event_caption_raw.split()).strip()
        if normalized_caption and normalized_caption.casefold() not in {"none", "null", "unknown"}:
            event_caption = _canonicalize_subtask(normalized_caption, location=location)
    # Event-enriched rows use the goal-level caption as the effective training
    # label while retaining the refined source label for audit/debugging.
    coarse_subtask = source_subtask
    fine_subtask = _canonicalize_optional_annotation_text(
        row.get("fine_subtask"),
        field="fine_subtask",
        location=location,
    )
    execution_success = row.get("execution_success")
    if execution_success is not None and not isinstance(execution_success, bool):
        raise ValueError(
            f"{location}: execution_success must be a JSON boolean when present, "
            f"got {execution_success!r}."
        )
    return SubtaskAnnotation(
        episode_index=row_episode_index,
        start_frame=start_frame,
        end_frame=end_frame,
        subtask=event_caption or fine_subtask or coarse_subtask,
        object_name=_canonicalize_optional_annotation_text(
            row.get("object", row.get("object_name")),
            field="object",
            location=location,
        ),
        object_locator=_canonicalize_optional_annotation_text(
            row.get("object_locator"),
            field="object_locator",
            location=location,
        ),
        target=_canonicalize_optional_annotation_text(
            row.get("target"),
            field="target",
            location=location,
        ),
        coarse_subtask=coarse_subtask,
        fine_subtask=fine_subtask,
        execution_success=execution_success,
        event_caption=event_caption,
        event_id=(int(row["event_id"]) if row.get("event_id") is not None else None),
        source_subtask=source_subtask if is_event_enriched else "",
    )


def _episode_annotations(
    segments: tuple[SubtaskAnnotation, ...],
    *,
    source: Path,
    require_contiguous: bool,
) -> EpisodeSubtaskAnnotations:
    if not segments:
        raise ValueError(f"{source}: completed VLM2 annotation must contain at least one subtask group.")
    episode_index = segments[0].episode_index
    if require_contiguous and segments[0].start_frame != 0:
        raise ValueError(
            f"{source}: completed VLM2 groups for episode_index={episode_index} must start at frame 0, "
            f"got {segments[0].start_frame}."
        )
    for previous, current in zip(segments, segments[1:]):
        if current.start_frame < previous.start_frame:
            raise ValueError(
                f"{source}: VLM2 groups for episode_index={episode_index} are not in chronological order: "
                f"[{previous.start_frame}, {previous.end_frame}) before "
                f"[{current.start_frame}, {current.end_frame})."
            )
        if current.start_frame < previous.end_frame:
            raise ValueError(
                f"{source}: overlapping subtask annotations for episode_index={episode_index}: "
                f"[{previous.start_frame}, {previous.end_frame}) and "
                f"[{current.start_frame}, {current.end_frame})."
            )
        if require_contiguous and current.start_frame != previous.end_frame:
            raise ValueError(
                f"{source}: completed VLM2 groups for episode_index={episode_index} contain a gap between "
                f"[{previous.start_frame}, {previous.end_frame}) and "
                f"[{current.start_frame}, {current.end_frame})."
            )
    return EpisodeSubtaskAnnotations(
        starts=tuple(segment.start_frame for segment in segments),
        segments=segments,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(value).__name__}.")
    return value


def _load_json_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}.") from exc
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(value).__name__}.")
    return value


def _load_json_array_annotations(annotation_path: Path) -> dict[int, EpisodeSubtaskAnnotations]:
    grouped: dict[int, list[SubtaskAnnotation]] = {}
    for row_index, row in enumerate(_load_json_array(annotation_path)):
        location = f"{annotation_path}[{row_index}]"
        segment = _parse_segment(
            row,
            episode_index=None,
            location=location,
            require_exact_keys=False,
        )
        grouped.setdefault(segment.episode_index, []).append(segment)

    annotations: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index in sorted(grouped):
        segments = tuple(sorted(grouped[episode_index], key=lambda item: (item.start_frame, item.end_frame)))
        annotations[episode_index] = _episode_annotations(
            segments,
            source=annotation_path,
            require_contiguous=False,
        )
    return annotations


def _load_minimal_jsonl(annotation_path: Path) -> dict[int, EpisodeSubtaskAnnotations]:
    grouped: dict[int, list[SubtaskAnnotation]] = {}
    with annotation_path.open(encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            location = f"{annotation_path}:{line_number}"
            if not raw_line.strip():
                raise ValueError(f"{location}: blank lines are not allowed in subtask annotation JSONL.")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc.msg}.") from exc
            segment = _parse_segment(
                row,
                episode_index=None,
                location=location,
                require_exact_keys=True,
            )
            grouped.setdefault(segment.episode_index, []).append(segment)

    annotations: dict[int, EpisodeSubtaskAnnotations] = {}
    for episode_index in sorted(grouped):
        segments = tuple(sorted(grouped[episode_index], key=lambda item: (item.start_frame, item.end_frame)))
        annotations[episode_index] = _episode_annotations(
            segments,
            source=annotation_path,
            require_contiguous=False,
        )
    return annotations


def _load_vlm2_episodes(annotation_dir: Path) -> dict[int, EpisodeSubtaskAnnotations]:
    episode_dirs = sorted(
        child for child in annotation_dir.iterdir() if child.is_dir() and child.name.startswith("episode_")
    )
    annotations: dict[int, EpisodeSubtaskAnnotations] = {}
    completed_count = 0
    for episode_dir in episode_dirs:
        done_path = episode_dir / _VLM2_DONE_FILENAME
        if not done_path.is_file():
            # annotation.json and rolling checkpoints may exist before the atomic DONE marker.
            continue
        completed_count += 1
        annotation_path = episode_dir / _VLM2_ANNOTATION_FILENAME
        if not annotation_path.is_file():
            raise ValueError(f"{episode_dir}: DONE marker exists but {_VLM2_ANNOTATION_FILENAME} is missing.")

        episode_suffix = episode_dir.name.removeprefix("episode_")
        if not episode_suffix.isdigit():
            raise ValueError(f"{episode_dir}: completed episode directory must end in a non-negative integer ID.")
        directory_episode_index = int(episode_suffix)

        marker = _load_json_object(done_path)
        if marker.get("status") != "complete":
            raise ValueError(f"{done_path}: DONE marker status must be 'complete', got {marker.get('status')!r}.")
        marker_episode_index = _parse_nonnegative_json_int(
            marker.get("episode_index"),
            field="episode_index",
            location=str(done_path),
        )
        if marker_episode_index != directory_episode_index:
            raise ValueError(
                f"{done_path}: marker episode_index={marker_episode_index} does not match directory "
                f"episode_index={directory_episode_index}."
            )

        annotation_bytes = annotation_path.read_bytes()
        expected_sha256 = marker.get("annotation_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise ValueError(f"{done_path}: annotation_sha256 must be a non-empty string.")
        actual_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{annotation_path}: SHA256 does not match the completed marker: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        try:
            annotation = json.loads(annotation_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{annotation_path}: invalid JSON: {exc}.") from exc
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path}: expected a JSON object, got {type(annotation).__name__}.")
        episode_index = _parse_nonnegative_json_int(
            annotation.get("episode_index"),
            field="episode_index",
            location=str(annotation_path),
        )
        if episode_index != marker_episode_index:
            raise ValueError(
                f"{annotation_path}: annotation episode_index={episode_index} does not match completed marker "
                f"episode_index={marker_episode_index}."
            )
        if episode_index in annotations:
            raise ValueError(f"{annotation_dir}: duplicate completed episode_index={episode_index}.")

        groups = annotation.get("groups")
        if not isinstance(groups, list):
            raise ValueError(f"{annotation_path}: groups must be a JSON list, got {type(groups).__name__}.")
        marker_group_count = _parse_nonnegative_json_int(
            marker.get("group_count"),
            field="group_count",
            location=str(done_path),
        )
        if marker_group_count != len(groups):
            raise ValueError(
                f"{annotation_path}: groups count {len(groups)} does not match completed marker "
                f"group_count={marker_group_count}."
            )
        segments = tuple(
            _parse_segment(
                group,
                episode_index=episode_index,
                location=f"{annotation_path}:groups[{group_index}]",
                require_exact_keys=False,
            )
            for group_index, group in enumerate(groups)
        )
        annotations[episode_index] = _episode_annotations(
            segments,
            source=annotation_path,
            require_contiguous=True,
        )

    if completed_count == 0:
        raise ValueError(f"{annotation_dir}: no completed VLM2 episodes with {_VLM2_DONE_FILENAME} were found.")
    return dict(sorted(annotations.items()))


def _merge_subtask_filter_ranges(
    ranges: list[tuple[int, int]],
    *,
    location: str,
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start < merged[-1][1]:
            raise ValueError(f"{location}: overlapping kept clip ranges {merged[-1]!r} and {[start, end]!r}.")
        if merged and start == merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def load_subtask_episode_filters(
    path: str | Path,
    *,
    include_idle_stationary_subtasks: bool = False,
) -> dict[int, EpisodeSubtaskFilter]:
    """Load flattened group annotations and derive clip-level keep ranges.

    Expected input: a JSON array whose rows include at least ``episode_index``,
    ``start_frame``, ``end_frame``, ``subtask``, and ``task_completed``. Extra
    keys are ignored. Enriched v2 rows contain ``source_clips`` and apply the
    rules per clip. Enriched v3 rows contain group-level
    ``object_related_to_task`` and apply the rules to the complete row. Event
    enriched rows carry an ``event_caption`` on each source group; only a
    non-null event caption is retained for event WAM windows, while the source
    group label remains available to reject Idle/Stationary windows. Legacy
    rows retain the historical one-interval behavior for old launchers.

    Enriched rules:
    - episodes absent from the file are not kept;
    - drop every clip in groups labelled ``unknown``;
    - drop ``Idle`` and ``Remain stationary`` unless
      ``include_idle_stationary_subtasks`` is enabled;
    - keep the first ``task_completed=true`` group and drop clips in later groups;
    - for v2, drop individual source clips whose ``object_related_to_task`` is
      false; for v3, drop the complete row when its conservative group-level
      value is false;
    - drop every clip in a coarse group spanning more than 150 frames (150 is
      kept, 151 is dropped). A conflict group replaced by refine intervals uses
      each replacement interval as its new coarse group.
    """

    annotation_path = Path(path)
    rows = _load_json_array(annotation_path)
    v2_flags = [isinstance(row, dict) and "source_clips" in row for row in rows]
    v3_flags = [isinstance(row, dict) and "object_related_to_task" in row for row in rows]
    event_flags = [isinstance(row, dict) and "event_caption" in row for row in rows]
    if any(v2_flags) and not all(v2_flags):
        raise ValueError(f"{annotation_path}: cannot mix legacy rows with enriched source_clips rows.")
    if any(v3_flags) and not all(v3_flags):
        raise ValueError(f"{annotation_path}: cannot mix rows with and without object_related_to_task.")
    use_v2_filter = bool(rows) and all(v2_flags)
    use_v3_filter = bool(rows) and all(v3_flags) and not use_v2_filter
    # Event-enriched refined rows retain ``object_related_to_task`` from the
    # source JSON.  The event filter deliberately takes precedence over the
    # historical v3 filter: caption=null and event boundaries are authoritative
    # for this experiment.
    use_event_filter = bool(rows) and all(event_flags)
    grouped: dict[int, list[tuple[SubtaskAnnotation, bool, dict[str, Any], str]]] = {}
    for row_index, row in enumerate(rows):
        location = f"{annotation_path}[{row_index}]"
        segment = _parse_segment(
            row,
            episode_index=None,
            location=location,
            require_exact_keys=False,
        )
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a JSON object, got {type(row).__name__}.")
        task_completed = row.get("task_completed")
        if not isinstance(task_completed, bool):
            raise ValueError(f"{location}: task_completed must be a JSON boolean, got {task_completed!r}.")
        # Keep only the clip-level fields needed by the filter.  The complete
        # merged rows can be large; dropping the unrelated group metadata here
        # avoids retaining a second full copy while the dataset is initialized.
        filter_row = (
            {"event_caption": row.get("event_caption"), "event_id": row.get("event_id")}
            if use_event_filter
            else
            (
            {
                "source_clips": row.get("source_clips"),
                "group_duration_frames": row.get("group_duration_frames"),
            }
            if use_v2_filter
            else {
                "object_related_to_task": row.get("object_related_to_task"),
                "group_duration_frames": row.get("group_duration_frames"),
                "coarse_group_duration_frames": row.get("coarse_group_duration_frames"),
            }
            if use_v3_filter
            else {}
            )
        )
        grouped.setdefault(segment.episode_index, []).append((segment, task_completed, filter_row, location))
    del rows

    filters: dict[int, EpisodeSubtaskFilter] = {}
    for episode_index in sorted(grouped):
        entries = sorted(grouped[episode_index], key=lambda item: (item[0].start_frame, item[0].end_frame))
        segments = tuple(segment for segment, _, _, _ in entries)
        _episode_annotations(
            segments,
            source=annotation_path,
            require_contiguous=False,
        )

        if use_event_filter:
            first_completed_position = next(
                (position for position, (_, task_completed, _, _) in enumerate(entries) if task_completed),
                None,
            )
            kept_clip_ranges: list[tuple[int, int]] = []
            for position, (segment, _, row, row_location) in enumerate(entries):
                # Preserve the first completing group itself, but never train
                # motion after task completion even if it inherited a caption.
                if first_completed_position is not None and position > first_completed_position:
                    continue
                caption = row.get("event_caption")
                if caption is None or (isinstance(caption, str) and caption.strip().casefold() in {"", "none", "null", "unknown"}):
                    continue
                event_id = row.get("event_id")
                if isinstance(event_id, bool) or not isinstance(event_id, int):
                    raise ValueError(f"{row_location}: event_id must be an integer for event-enriched rows.")
                kept_clip_ranges.append((segment.start_frame, segment.end_frame))
            keep_ranges = _merge_subtask_filter_ranges(
                kept_clip_ranges,
                location=f"{annotation_path}:episode_index={episode_index}",
            )
            if keep_ranges:
                filters[episode_index] = EpisodeSubtaskFilter(
                    keep_start_frame=keep_ranges[0][0],
                    keep_end_frame=keep_ranges[-1][1],
                    keep_ranges=keep_ranges,
                )
            else:
                filters[episode_index] = EpisodeSubtaskFilter(keep_start_frame=0, keep_end_frame=0)
            continue

        if use_v2_filter or use_v3_filter:
            first_completed_position = next(
                (position for position, (_, task_completed, _, _) in enumerate(entries) if task_completed),
                None,
            )
            kept_clip_ranges: list[tuple[int, int]] = []
            post_completion_clip_ranges: list[tuple[int, int]] = []
            post_completion_move_away_clip_ranges: list[tuple[int, int]] = []
            for position, (segment, _, row, row_location) in enumerate(entries):
                canonical_subtask = _canonicalize_subtask(
                    segment.subtask,
                    location=f"{annotation_path}:episode_index={episode_index}",
                ).casefold()
                group_duration = segment.end_frame - segment.start_frame
                reported_duration = row.get("group_duration_frames")
                if reported_duration is not None:
                    parsed_duration = _parse_nonnegative_json_int(
                        reported_duration,
                        field="group_duration_frames",
                        location=row_location,
                    )
                    if parsed_duration != group_duration:
                        raise ValueError(
                            f"{row_location}: group_duration_frames={parsed_duration} does not match "
                            f"[{segment.start_frame}, {segment.end_frame}) span {group_duration}."
                        )
                coarse_group_duration = group_duration
                reported_coarse_duration = row.get("coarse_group_duration_frames")
                if reported_coarse_duration is not None:
                    coarse_group_duration = _parse_nonnegative_json_int(
                        reported_coarse_duration,
                        field="coarse_group_duration_frames",
                        location=row_location,
                    )
                    if coarse_group_duration < group_duration:
                        raise ValueError(
                            f"{row_location}: coarse_group_duration_frames={coarse_group_duration} is smaller than "
                            f"the split group span {group_duration}."
                        )
                drop_idle_stationary = (
                    canonical_subtask in _SUBTASK_FILTER_DROP_LABELS
                    and not include_idle_stationary_subtasks
                )
                drop_group = (
                    canonical_subtask == "unknown"
                    or drop_idle_stationary
                    or (first_completed_position is not None and position > first_completed_position)
                    or coarse_group_duration > _SUBTASK_FILTER_MAX_GROUP_DURATION_FRAMES
                )
                is_post_completion = first_completed_position is not None and position > first_completed_position
                coarse_label = (segment.coarse_subtask or segment.subtask).casefold()
                is_move_away = coarse_label.startswith("move away")

                if use_v3_filter:
                    if is_post_completion:
                        post_completion_clip_ranges.append((segment.start_frame, segment.end_frame))
                    object_related = row.get("object_related_to_task")
                    if not isinstance(object_related, bool):
                        raise ValueError(
                            f"{row_location}: object_related_to_task must be a JSON boolean, got {object_related!r}."
                        )
                    if (
                        is_post_completion
                        and is_move_away
                        and object_related
                        and canonical_subtask not in _SUBTASK_FILTER_DROP_LABELS
                        and canonical_subtask != "unknown"
                        and coarse_group_duration <= _SUBTASK_FILTER_MAX_GROUP_DURATION_FRAMES
                    ):
                        post_completion_move_away_clip_ranges.append(
                            (segment.start_frame, segment.end_frame)
                        )
                    if not drop_group and object_related:
                        kept_clip_ranges.append((segment.start_frame, segment.end_frame))
                    continue

                raw_clips = row.get("source_clips")
                if not isinstance(raw_clips, list) or not raw_clips:
                    raise ValueError(f"{row_location}: source_clips must be a non-empty JSON list.")
                previous_clip_end: int | None = None
                for clip_position, clip in enumerate(raw_clips):
                    clip_location = f"{row_location}.source_clips[{clip_position}]"
                    if not isinstance(clip, dict):
                        raise ValueError(f"{clip_location}: expected a JSON object.")
                    clip_start = _parse_nonnegative_json_int(
                        clip.get("start_frame"),
                        field="start_frame",
                        location=clip_location,
                    )
                    clip_end = _parse_nonnegative_json_int(
                        clip.get("end_frame"),
                        field="end_frame",
                        location=clip_location,
                    )
                    if clip_end <= clip_start:
                        raise ValueError(f"{clip_location}: expected a non-empty half-open frame range.")
                    if clip_start < segment.start_frame or clip_end > segment.end_frame:
                        raise ValueError(
                            f"{clip_location}: clip [{clip_start}, {clip_end}) lies outside group "
                            f"[{segment.start_frame}, {segment.end_frame})."
                        )
                    if previous_clip_end is not None and clip_start < previous_clip_end:
                        raise ValueError(f"{clip_location}: source clips overlap or are out of order.")
                    previous_clip_end = clip_end
                    if is_post_completion:
                        post_completion_clip_ranges.append((clip_start, clip_end))
                    object_related = clip.get("object_related_to_task")
                    if not isinstance(object_related, bool):
                        raise ValueError(
                            f"{clip_location}: object_related_to_task must be a JSON boolean, got {object_related!r}."
                        )
                    if (
                        is_post_completion
                        and is_move_away
                        and object_related
                        and canonical_subtask not in _SUBTASK_FILTER_DROP_LABELS
                        and canonical_subtask != "unknown"
                        and coarse_group_duration <= _SUBTASK_FILTER_MAX_GROUP_DURATION_FRAMES
                    ):
                        post_completion_move_away_clip_ranges.append((clip_start, clip_end))
                    if not drop_group and object_related:
                        kept_clip_ranges.append((clip_start, clip_end))

            keep_ranges = _merge_subtask_filter_ranges(
                kept_clip_ranges,
                location=f"{annotation_path}:episode_index={episode_index}",
            )
            if not keep_ranges:
                filters[episode_index] = EpisodeSubtaskFilter(
                    keep_start_frame=0,
                    keep_end_frame=0,
                    post_completion_ranges=_merge_subtask_filter_ranges(
                        post_completion_clip_ranges,
                        location=f"{annotation_path}:episode_index={episode_index}:post_completion",
                    ),
                    post_completion_move_away_ranges=_merge_subtask_filter_ranges(
                        post_completion_move_away_clip_ranges,
                        location=f"{annotation_path}:episode_index={episode_index}:post_completion_move_away",
                    ),
                )
                continue
            filters[episode_index] = EpisodeSubtaskFilter(
                keep_start_frame=keep_ranges[0][0],
                keep_end_frame=keep_ranges[-1][1],
                keep_ranges=keep_ranges,
                post_completion_ranges=_merge_subtask_filter_ranges(
                    post_completion_clip_ranges,
                    location=f"{annotation_path}:episode_index={episode_index}:post_completion",
                ),
                post_completion_move_away_ranges=_merge_subtask_filter_ranges(
                    post_completion_move_away_clip_ranges,
                    location=f"{annotation_path}:episode_index={episode_index}:post_completion_move_away",
                ),
            )
            continue

        # Backward compatibility for the pre-v2 flattened annotation format.
        keep_start_frame = 0
        keep_end_frame: int | None = None
        post_completion_ranges: list[tuple[int, int]] = []
        first_completed_position: int | None = None
        for position, (segment, task_completed, _, _) in enumerate(entries):
            if task_completed and first_completed_position is None:
                first_completed_position = position
            elif first_completed_position is not None and position > first_completed_position:
                post_completion_ranges.append((segment.start_frame, segment.end_frame))
        for segment, task_completed, _, _ in entries:
            if (
                _canonicalize_subtask(
                    segment.subtask, location=f"{annotation_path}:episode_index={episode_index}"
                ).casefold()
                == "idle"
            ):
                keep_start_frame = max(keep_start_frame, segment.end_frame)
            if task_completed and keep_end_frame is None:
                keep_end_frame = segment.end_frame

        if keep_end_frame is None or keep_end_frame <= keep_start_frame:
            continue
        filters[episode_index] = EpisodeSubtaskFilter(
            keep_start_frame=keep_start_frame,
            keep_end_frame=keep_end_frame,
            post_completion_ranges=_merge_subtask_filter_ranges(
                post_completion_ranges,
                location=f"{annotation_path}:episode_index={episode_index}:post_completion",
            ),
        )
    return filters


def load_subtask_annotations(path: str | Path) -> dict[int, EpisodeSubtaskAnnotations]:
    """Load subtask annotations from minimal JSONL or a VLM2 ``episodes/`` directory.

    Minimal JSONL rows contain exactly ``episode_index``, ``start_frame``,
    ``end_frame``, and ``subtask``. A directory input consumes only episode
    folders atomically published with ``EPISODE_PASS2_DONE.json`` and validates
    the marker, annotation SHA256, group count, and contiguous group intervals.
    All frame ranges are episode-local half-open intervals.
    """
    annotation_path = Path(path)
    if annotation_path.is_dir():
        return _load_vlm2_episodes(annotation_path)
    if annotation_path.is_file():
        content = annotation_path.read_text(encoding="utf-8")
        if content.lstrip().startswith("["):
            return _load_json_array_annotations(annotation_path)
        return _load_minimal_jsonl(annotation_path)
    raise FileNotFoundError(f"Subtask annotations path does not exist: {annotation_path}")


def resolve_subtask_segments(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    episode_index: int,
    action_start_frame: int,
    action_end_frame: int,
    boundary_trim_frames: int = _SUBTASK_BOUNDARY_MAX_OVERLAP_FRAMES,
    pregrasp_boundary_trim_frames: int = 0,
) -> tuple[SubtaskAnnotation, ...]:
    """Resolve the annotation segments used to label one action window.

    The full half-open action window must be covered without a gap. Consecutive
    segments that differ only in whitespace, case, or terminal sentence
    punctuation are still returned separately; text composition collapses those
    adjacent equivalent labels later.

    A leading segment contributing at most five frames is omitted when another
    segment covers the window.  Likewise, a trailing segment contributing at
    most five frames is omitted when an earlier segment remains.  This handles
    the common ``5 frames of approach then 27 frames of grasp`` boundary
    without preserving a stale leading step or appending a barely entered future
    step. Internal short segments are retained. Coverage validation happens
    before trimming, so annotation gaps still return no supervision and the
    parent dataset's window count is unchanged.
    """
    if isinstance(boundary_trim_frames, bool) or not isinstance(boundary_trim_frames, int) or boundary_trim_frames < 0:
        raise ValueError(f"boundary_trim_frames must be a non-negative integer, got {boundary_trim_frames!r}.")
    if (
        isinstance(pregrasp_boundary_trim_frames, bool)
        or not isinstance(pregrasp_boundary_trim_frames, int)
        or pregrasp_boundary_trim_frames < 0
    ):
        raise ValueError(
            "pregrasp_boundary_trim_frames must be a non-negative integer, "
            f"got {pregrasp_boundary_trim_frames!r}."
        )
    if action_start_frame < 0 or action_end_frame <= action_start_frame:
        raise ValueError(
            "action window must be a non-empty, non-negative half-open interval, "
            f"got [{action_start_frame}, {action_end_frame})."
        )
    episode_annotations = annotations.get(int(episode_index))
    if episode_annotations is None:
        return ()

    segment_position = bisect_right(episode_annotations.starts, action_start_frame) - 1
    if segment_position < 0:
        return ()
    cursor = action_start_frame
    covered: list[SubtaskAnnotation] = []
    while cursor < action_end_frame:
        if segment_position >= len(episode_annotations.segments):
            return ()
        segment = episode_annotations.segments[segment_position]
        if segment.start_frame > cursor or segment.end_frame <= cursor:
            return ()

        covered.append(segment)
        cursor = min(action_end_frame, segment.end_frame)
        if cursor < action_end_frame:
            segment_position += 1
            if (
                segment_position >= len(episode_annotations.segments)
                or episode_annotations.segments[segment_position].start_frame != cursor
            ):
                return ()

    # Dedicated coarse pre-grasp rule: when the first distinct coarse label in
    # the window is Move/Approach and the next distinct coarse label is Grasp,
    # drop the leading Move/Approach portion if its overlap is short enough.
    # This is measured in the current window, not by the total segment
    # duration.  Fine rows inside one coarse segment are ignored here.
    selected = covered
    if pregrasp_boundary_trim_frames > 0 and len(covered) > 1:
        def coarse_label(segment: SubtaskAnnotation) -> str:
            return " ".join(str(segment.coarse_subtask or segment.subtask).split()).casefold()

        first_label = coarse_label(covered[0])
        next_distinct = next(
            (position for position, segment in enumerate(covered[1:], start=1)
             if coarse_label(segment) != first_label),
            None,
        )
        if next_distinct is not None:
            next_label = coarse_label(covered[next_distinct])
            leading_overlap = covered[next_distinct].start_frame - action_start_frame
            is_approach = (first_label.startswith("approach") or first_label.startswith("move")) and not first_label.startswith("move away")
            if is_approach and next_label.startswith("grasp") and leading_overlap <= pregrasp_boundary_trim_frames:
                selected = covered[next_distinct:]

    # Only trim the two boundaries.  A short middle segment can still be a
    # meaningful transition (for example, a brief contact/grip event).
    if len(selected) > 1:
        leading_overlap = min(action_end_frame, selected[0].end_frame) - action_start_frame
        if leading_overlap <= boundary_trim_frames:
            selected = selected[1:]
        if len(selected) > 1:
            trailing_overlap = action_end_frame - max(action_start_frame, selected[-1].start_frame)
            if trailing_overlap <= boundary_trim_frames:
                selected = selected[:-1]

    return tuple(selected)


def compose_subtask_text(segments: tuple[SubtaskAnnotation, ...]) -> str:
    labels: list[str] = []
    previous_label_key: str | None = None
    for segment in segments:
        label = _canonicalize_subtask(segment.subtask, location="loaded subtask annotation")
        label_key = label.casefold()
        if label_key != previous_label_key:
            labels.append(label)
            previous_label_key = label_key
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]

    def continuation(label: str) -> str:
        # Lowercase ordinary sentence starts after a connector while preserving
        # acronyms and proper labels such as USB or LEGO.
        if len(label) > 1 and label[0].isupper() and label[1].islower():
            return label[0].lower() + label[1:]
        return label

    text = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        digest = hashlib.blake2b(
            f"subtask-connector:{index}:{labels[index - 1]}:{label}".encode("utf-8"),
            digest_size=1,
        ).digest()[0]
        connector = (", then ", "; next, ", ". After that, ")[digest % 3]
        text += connector + continuation(label)
    return text


# These are deliberately scene-relative relations.  ``toward``/``away`` are
# action directions and therefore do not count as an already-described object
# position.  Keeping the vocabulary here (rather than in the dataset class)
# makes the same guard available to offline annotation inspection tools.
_OBJECT_POSITION_RE = re.compile(
    r"\b(?:"
    r"left\s+of|right\s+of|left\s+side(?:\s+of)?|right\s+side(?:\s+of)?|"
    r"on\s+the\s+(?:left|right)|to\s+the\s+(?:left|right)\b|"
    r"behind|in\s+front\s+of|front\s+of|next\s+to|beside|alongside|"
    # ``in the bowl`` is common in the labels and is already an object
    # location; keep it distinct from the action-direction word ``in``.
    r"between|inside|within|into|near|close\s+to|above|below|under|over|"
    r"on\s+top\s+of|in\s+(?:the|a|an)\s+"
    r")\b",
    re.IGNORECASE,
)

_SUPPORT_NOUN_RE = (
    r"(?:"
    r"(?:[a-z]+\s+)?table|"
    r"(?:[a-z]+\s+)?counter(?:top)?|"
    r"(?:[a-z]+\s+)?work\s+surface|"
    r"(?:[a-z]+\s+)?surface|"
    r"(?:[a-z]+\s+)?desk|"
    r"worktop"
    r")"
)
_SUPPORT_ARTICLE_RE = r"(?:the|a|an)\s+"
_SUPPORT_ONLY_RE = re.compile(rf"^(?:{_SUPPORT_ARTICLE_RE})?{_SUPPORT_NOUN_RE}$", re.IGNORECASE)
_SUPPORT_PREFIX_RE = re.compile(
    rf"^(?:on|upon|at|by)\s+(?:{_SUPPORT_ARTICLE_RE})?{_SUPPORT_NOUN_RE}(?:\s*(?:[,;:]\s*|\s+))",
    re.IGNORECASE,
)
_SUPPORT_SUFFIX_RE = re.compile(
    rf"\s+(?:on|upon|at)\s+(?:{_SUPPORT_ARTICLE_RE})?{_SUPPORT_NOUN_RE}$",
    re.IGNORECASE,
)
_DYNAMIC_LOCATOR_RE = re.compile(
    r"\b(?:gripper|robot\s+gripper|robot\s+arm|robot\s+hand|held\s+by|in\s+hand)\b",
    re.IGNORECASE,
)
_UNKNOWN_OBJECT_RE = re.compile(r"^(?:unknown|none|null|n/?a|)$", re.IGNORECASE)
_DYNAMIC_OBJECT_RE = re.compile(
    r"^(?:the\s+)?(?:gripper|robot\s+gripper|robot\s+arm|robot\s+hand|hand)$",
    re.IGNORECASE,
)


def _normalize_annotation_phrase(value: str) -> str:
    """Collapse whitespace and harmless terminal punctuation in metadata."""

    return " ".join(value.replace("\n", " ").split()).strip().rstrip(_TRAILING_SENTENCE_PUNCTUATION).rstrip()


def _is_generic_support_reference(value: str) -> bool:
    return bool(_SUPPORT_ONLY_RE.fullmatch(value.strip()))


def extract_object_position_relation(object_locator: str) -> str:
    """Extract a useful scene-relative phrase from ``object_locator``.

    VLM2 locators commonly look like ``on the table behind the pink bowl``.
    The support-surface prefix is removed, yielding ``behind the pink bowl``.
    Bare support locations (``on the table``/``on the counter``), gripper
    states, and references to an anonymous support surface return ``""``.
    The returned phrase is suitable for inserting after ``located`` or ``is``.
    """

    locator = _normalize_annotation_phrase(str(object_locator or ""))
    if not locator or _UNKNOWN_OBJECT_RE.fullmatch(locator) or _DYNAMIC_LOCATOR_RE.search(locator):
        return ""

    # Strip only a generic support prefix.  Do not strip ``inside the orange
    # cup`` or other object-containing locators.
    while True:
        stripped = _SUPPORT_PREFIX_RE.sub("", locator, count=1).strip()
        if stripped == locator:
            break
        locator = stripped
    if not locator or _UNKNOWN_OBJECT_RE.fullmatch(locator):
        return ""

    # A relation may be followed by the support surface (``right of the mouse
    # on the table``).  Removing it leaves the actual reference object.
    locator = _SUPPORT_SUFFIX_RE.sub("", locator).strip()

    relation_patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^to\s+(?:the\s+)?left\s+of\s+(.+)$", re.IGNORECASE), "to the left of"),
        (re.compile(r"^to\s+(?:the\s+)?right\s+of\s+(.+)$", re.IGNORECASE), "to the right of"),
        (re.compile(r"^left\s+of\s+(.+)$", re.IGNORECASE), "to the left of"),
        (re.compile(r"^right\s+of\s+(.+)$", re.IGNORECASE), "to the right of"),
        (re.compile(r"^left\s+side\s+of\s+(.+)$", re.IGNORECASE), "on the left side of"),
        (re.compile(r"^right\s+side\s+of\s+(.+)$", re.IGNORECASE), "on the right side of"),
        (re.compile(r"^in\s+front\s+of\s+(.+)$", re.IGNORECASE), "in front of"),
        (re.compile(r"^front\s+of\s+(.+)$", re.IGNORECASE), "in front of"),
        (re.compile(r"^behind\s+(.+)$", re.IGNORECASE), "behind"),
        (re.compile(r"^(?:next\s+to|beside|alongside)\s+(.+)$", re.IGNORECASE), "next to"),
        (re.compile(r"^between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE), "between"),
        (re.compile(r"^(?:inside|within|in)\s+(.+)$", re.IGNORECASE), "inside"),
    )
    for pattern, canonical_prefix in relation_patterns:
        match = pattern.fullmatch(locator)
        if match is None:
            continue
        if canonical_prefix == "between":
            reference_parts = tuple(_normalize_annotation_phrase(part) for part in (match.group(1), match.group(2)))
            if any(
                not part
                or _is_generic_support_reference(part)
                or _DYNAMIC_LOCATOR_RE.search(part)
                or _UNKNOWN_OBJECT_RE.fullmatch(part)
                for part in reference_parts
            ):
                return ""
            reference = " and ".join(reference_parts)
        else:
            reference = _normalize_annotation_phrase(match.group(1))
        if not reference or _is_generic_support_reference(reference) or _DYNAMIC_LOCATOR_RE.search(reference):
            return ""
        # ``unknown`` should never become a generated target description.
        if _UNKNOWN_OBJECT_RE.fullmatch(reference):
            return ""
        return f"{canonical_prefix} {reference}"
    return ""


def subtask_contains_object_position(subtask: str) -> bool:
    """Whether a subtask already states a scene/object relative position."""

    normalized = _normalize_annotation_phrase(str(subtask or ""))
    if not normalized:
        return False
    return bool(_OBJECT_POSITION_RE.search(normalized))


def is_object_position_augmentation_action(subtask: str) -> bool:
    """Restrict augmentation to approach/toward actions.

    This whitelist prevents a positional locator from changing the meaning of
    ``Move ... away``, ``Move ... into``, or object-transport actions.
    """

    normalized = _normalize_annotation_phrase(str(subtask or "")).casefold()
    if re.match(r"^approach\b", normalized):
        return True
    return bool(
        re.match(r"^move\s+(?:to|towards?)\b", normalized)
        or re.match(r"^move\s+(?:the\s+)?(?:robot\s+)?gripper\s+(?:to|towards?)\b", normalized)
    )


def _usable_annotation_object(segment: SubtaskAnnotation) -> str:
    object_name = _normalize_annotation_phrase(segment.object_name)
    # An explicit ``object=unknown/gripper`` is a deliberate negative signal;
    # do not silently replace it with ``target`` and generate a misleading
    # ``Move ... toward`` condition.  ``target`` is a fallback only when the
    # enriched row omitted ``object`` altogether.
    if object_name:
        return (
            ""
            if _UNKNOWN_OBJECT_RE.fullmatch(object_name) or _DYNAMIC_OBJECT_RE.fullmatch(object_name)
            else object_name
        )
    target = _normalize_annotation_phrase(segment.target)
    if target and not _UNKNOWN_OBJECT_RE.fullmatch(target) and not _DYNAMIC_OBJECT_RE.fullmatch(target):
        return target
    return ""


def object_position_augmentation_parts(segment: SubtaskAnnotation) -> tuple[str, str] | None:
    """Return ``(display_object, relation)`` when a segment is augmentable."""

    if not is_object_position_augmentation_action(segment.subtask):
        return None
    if subtask_contains_object_position(segment.subtask):
        return None
    object_name = _usable_annotation_object(segment)
    relation = extract_object_position_relation(segment.object_locator)
    if not object_name or not relation:
        return None
    return object_name, relation


def augment_subtask_with_object_position(
    subtask: str,
    *,
    segment: SubtaskAnnotation,
    style: str = "toward",
) -> str:
    """Add annotation-backed object position context to one WAM condition.

    ``style`` accepts ``toward`` (the explicit target-location sentence),
    ``lead`` (location sentence first), ``structured`` (three-line
    target/relative-position/action form), and ``embedded`` (keeps the
    original action clause and appends ``<relation>`` directly as a
    post-nominal modifier, e.g. ``Approach the pen next to the keyboard``).
    ``random`` is intentionally not resolved here: the dataset
    resolves it with a stable sample-id hash and calls this function with one
    concrete style.
    """

    parts = object_position_augmentation_parts(segment)
    if parts is None:
        return _normalize_annotation_phrase(subtask)
    object_name, relation = parts
    full_subtask = _normalize_annotation_phrase(subtask)
    # A window can contain multiple labels.  If a later label already states a
    # relation, leave the complete condition untouched rather than prepending
    # a second, potentially contradictory location description.
    if subtask_contains_object_position(full_subtask):
        return full_subtask
    first_subtask = _normalize_annotation_phrase(segment.subtask)
    suffix = ""
    if full_subtask.casefold().startswith(first_subtask.casefold()):
        suffix = full_subtask[len(first_subtask) :].strip()
    # Continuations always use a lowercase ``, then`` connector regardless of
    # how the source multi-label text spelled the boundary.
    suffix = re.sub(r"^(?:then|,\s*then|;\s*next,?|\.\s*after\s+that,?)\s+", "", suffix, flags=re.IGNORECASE).strip()
    if suffix and suffix[0].isupper():
        suffix = suffix[0].lower() + suffix[1:]
    suffix = f", then {suffix}" if suffix else ""

    article_object = object_name if re.match(r"^(?:the|a|an)\s+", object_name, re.IGNORECASE) else f"the {object_name}"
    normalized_style = str(style).strip().lower()
    if normalized_style == "embedded":
        # Position augmentation only: append the relation to the augmented
        # segment.  Multi-segment joining stays in the caller.
        return f"{first_subtask} {relation}"
    if normalized_style == "toward":
        return f"Move the gripper toward {article_object}. The target object is located {relation}{suffix}"
    if normalized_style == "lead":
        article_object_sentence = article_object[:1].upper() + article_object[1:]
        return f"{article_object_sentence} is {relation}. Move the gripper toward {article_object}{suffix}"
    if normalized_style == "structured":
        return f"Target: {object_name}\nRelative position: {relation}\nAction: {full_subtask}"
    raise ValueError(
        "object position augmentation style must be one of 'toward', 'lead', 'structured', or 'embedded', "
        f"got {style!r}."
    )


def lookup_subtask(
    annotations: dict[int, EpisodeSubtaskAnnotations],
    *,
    episode_index: int,
    action_start_frame: int,
    action_end_frame: int,
    boundary_trim_frames: int = _SUBTASK_BOUNDARY_MAX_OVERLAP_FRAMES,
) -> str:
    """Return text supervision for one action window, joining segments with ``then``."""

    return compose_subtask_text(
        resolve_subtask_segments(
            annotations,
            episode_index=episode_index,
            action_start_frame=action_start_frame,
            action_end_frame=action_end_frame,
            boundary_trim_frames=boundary_trim_frames,
        )
    )


__all__ = [
    "EpisodeSubtaskFilter",
    "EpisodeSubtaskAnnotations",
    "SubtaskAnnotation",
    "load_subtask_episode_filters",
    "load_subtask_annotations",
    "compose_subtask_text",
    "extract_object_position_relation",
    "subtask_contains_object_position",
    "is_object_position_augmentation_action",
    "object_position_augmentation_parts",
    "augment_subtask_with_object_position",
    "resolve_subtask_segments",
    "delay_contact_subtask_boundaries",
    "advance_release_subtask_boundaries",
    "advance_pregrasp_subtask_boundaries",
    "relabel_subtask_annotations",
    "lookup_subtask",
]
