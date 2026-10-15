# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Start-frame rationale sidecar view for the DROID Reasoner dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import _should_log_dataset_summary
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_subgoal_dataset import (
    DROIDLeRobotSubgoalDataset,
    DROIDSubgoalWindow,
)
from cosmos_framework.data.vfm.action.reasoner_targets import format_reasoner_rationale_target
from cosmos_framework.utils import log

DROIDStartFrameBlockIndex = dict[int, tuple[tuple[int, int, int], ...]]


@dataclass(frozen=True, slots=True)
class DROIDReasonerRationaleAnnotation:
    """One verified rationale attached to an episode-local decision frame."""

    sample_id: str
    episode_index: int
    start_frame: int
    history_offsets: tuple[int, ...]
    reason: str
    subtask: str
    bucket: str = ""


@dataclass(frozen=True, slots=True)
class _MappedRationale:
    annotation: DROIDReasonerRationaleAnnotation
    source_index: int


@dataclass(frozen=True, slots=True)
class _RationaleLookup:
    text: str
    available: bool = True


def _nonnegative_int(value: Any, *, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: {field} must be a non-negative JSON integer, got {value!r}.")
    return int(value)


def _text(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string.")
    return value.strip()


def _history_offsets(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location}: history_offsets must be a JSON integer array.")
    offsets = tuple(value)
    if not offsets or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets):
        raise ValueError(f"{location}: history_offsets must contain one or more integers.")
    if any(item > 0 for item in offsets) or offsets[-1] != 0 or tuple(sorted(set(offsets))) != offsets:
        raise ValueError(
            f"{location}: history_offsets must be unique, sorted, causal (<= 0), and end in 0; got {offsets!r}."
        )
    return offsets


def _read_rows(path: Path) -> list[Any]:
    content = path.read_text(encoding="utf-8")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if value is not None:
        raise ValueError(f"{path}: expected a JSON object, JSON array, or JSONL records.")

    rows: list[Any] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL lines are not allowed.")
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}.") from exc
    return rows


def _row_target_fields(row: dict[str, Any], *, location: str) -> tuple[str, str]:
    """Read the canonical flat sidecar, with demo-result compatibility."""

    parsed = row.get("parsed")
    parsed = parsed if isinstance(parsed, dict) else {}
    verdict = row.get("action_verdict", parsed.get("action_verdict"))
    if verdict is not None and verdict != "correct":
        raise ValueError(f"{location}: only action_verdict='correct' rows are trainable, got {verdict!r}.")
    status = row.get("status")
    if status is not None and status != "complete":
        raise ValueError(f"{location}: only status='complete' rows are trainable, got {status!r}.")

    reason = row.get("reason", parsed.get("reason"))
    subtask = row.get("subtask", row.get("original_annotation", parsed.get("original_annotation")))
    if subtask is None:
        annotation = row.get("annotation")
        if isinstance(annotation, dict):
            subtask = annotation.get("fine_subtask") or annotation.get("subtask")
    clean_reason = _text(reason, field="reason", location=location)
    clean_subtask = _text(subtask, field="subtask", location=location)
    # Central formatting validation rejects nested think/chat boundary tokens.
    format_reasoner_rationale_target(reason=clean_reason, subtask=clean_subtask)
    return clean_reason, clean_subtask


def load_droid_reasoner_rationale_annotations(
    path: str | Path,
) -> tuple[DROIDReasonerRationaleAnnotation, ...]:
    """Load strict JSON/JSONL start-frame rationale annotations.

    Canonical rows contain ``sample_id``, ``episode_index``, ``start_frame``,
    ``history_offsets``, ``reason``, and ``subtask``. Completed two-pass demo
    records are also accepted by reading ``parsed.reason`` and
    ``parsed.original_annotation``.
    """

    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Reasoner rationale sidecar does not exist: {annotation_path}")
    output: list[DROIDReasonerRationaleAnnotation] = []
    seen_sample_ids: set[str] = set()
    seen_frames: set[tuple[int, int]] = set()
    for index, value in enumerate(_read_rows(annotation_path)):
        location = f"{annotation_path}[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected a JSON object, got {type(value).__name__}.")
        episode_index = _nonnegative_int(value.get("episode_index"), field="episode_index", location=location)
        start_frame = _nonnegative_int(value.get("start_frame"), field="start_frame", location=location)
        sample_id = _text(
            value.get("sample_id", f"ep{episode_index:06d}_f{start_frame:06d}"),
            field="sample_id",
            location=location,
        )
        if sample_id in seen_sample_ids:
            raise ValueError(f"{location}: duplicate sample_id {sample_id!r}.")
        frame_key = (episode_index, start_frame)
        if frame_key in seen_frames:
            raise ValueError(f"{location}: duplicate rationale decision frame {frame_key!r}.")
        reason, subtask = _row_target_fields(value, location=location)
        offsets = _history_offsets(value.get("history_offsets", [0]), location=location)
        bucket_value = value.get("family", value.get("bucket", ""))
        bucket = "" if bucket_value is None else str(bucket_value).strip()
        output.append(
            DROIDReasonerRationaleAnnotation(
                sample_id=sample_id,
                episode_index=episode_index,
                start_frame=start_frame,
                history_offsets=offsets,
                reason=reason,
                subtask=subtask,
                bucket=bucket,
            )
        )
        seen_sample_ids.add(sample_id)
        seen_frames.add(frame_key)
    if not output:
        raise ValueError(f"{annotation_path}: rationale sidecar contains no samples.")
    return tuple(output)


def build_droid_start_frame_block_index(
    dataset: DROIDLeRobotSubgoalDataset,
) -> DROIDStartFrameBlockIndex:
    """Index filtered dataset blocks by episode-local start frame.

    Each stored tuple is ``(local_start, local_end, source_start)``. The source
    index remains in the wrapped dataset's public flat-index space, so the same
    mapping works with episode splits and kept-range segment filtering.
    """

    blocks_by_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for source_start, length in dataset.get_shuffle_blocks():
        source_start = int(source_start)
        length = int(length)
        if length <= 0:
            continue
        first_window, _ = dataset._metadata_for_raw_index(source_start)
        last_window, _ = dataset._metadata_for_raw_index(source_start + length - 1)
        if (
            first_window.episode_index != last_window.episode_index
            or last_window.window_start_frame != first_window.window_start_frame + length - 1
        ):
            raise RuntimeError("DROID shuffle block is not one contiguous episode-local window run.")
        blocks_by_episode[first_window.episode_index].append(
            (first_window.window_start_frame, first_window.window_start_frame + length, source_start)
        )
    return {
        episode_index: tuple(sorted(blocks))
        for episode_index, blocks in blocks_by_episode.items()
    }


def find_droid_start_frame_source_index(
    block_index: DROIDStartFrameBlockIndex,
    *,
    episode_index: int,
    start_frame: int,
) -> int | None:
    """Return the filtered flat index for one episode-local decision frame."""

    for local_start, local_end, source_start in block_index.get(int(episode_index), ()):
        if local_start <= int(start_frame) < local_end:
            return source_start + int(start_frame) - local_start
    return None


class DROIDStartFrameRationaleDataset(Dataset):
    """Map rationale rows to existing DROID window starts without video scans.

    The wrapped dataset owns the DROID metadata/video implementation. This view
    stores only a compact sidecar-to-raw-index mapping, so the assistant target
    can change while task prompts and causal image decoding stay identical.
    """

    def __init__(
        self,
        dataset: DROIDLeRobotSubgoalDataset,
        *,
        rationale_annotations_path: str,
        strict_window_match: bool = False,
    ) -> None:
        if not getattr(dataset, "_is_reasoner_stream", False):
            raise TypeError("DROIDStartFrameRationaleDataset requires a DROID Reasoner-stream dataset.")
        self._dataset = dataset
        annotations = load_droid_reasoner_rationale_annotations(rationale_annotations_path)

        def normalize_subtask(value: str) -> str:
            return " ".join(value.split()).rstrip(".!?。！？").rstrip().casefold()

        block_index = build_droid_start_frame_block_index(dataset)

        mapped: list[_MappedRationale] = []
        unavailable: list[DROIDReasonerRationaleAnnotation] = []
        for annotation in annotations:
            source_index = find_droid_start_frame_source_index(
                block_index,
                episode_index=annotation.episode_index,
                start_frame=annotation.start_frame,
            )
            if source_index is None:
                unavailable.append(annotation)
                continue
            _, source_lookup = dataset._metadata_for_raw_index(source_index)
            if not source_lookup.available:
                raise ValueError(
                    f"Rationale {annotation.sample_id!r} maps to a source window without a subtask target."
                )
            source_labels = {
                normalize_subtask(segment.subtask) for segment in source_lookup.segments if str(segment.subtask).strip()
            }
            # The sidecar is keyed to a group start, while the ordinary
            # 32-frame target may include a following phase.  Require the
            # verified final subtask to match one contributing source segment;
            # never silently train a reason against a different action.
            if normalize_subtask(annotation.subtask) not in source_labels:
                raise ValueError(
                    f"Rationale {annotation.sample_id!r} subtask {annotation.subtask!r} does not match "
                    f"the mapped source-window labels {sorted(source_labels)!r}."
                )
            mapped.append(_MappedRationale(annotation=annotation, source_index=source_index))

        if strict_window_match and unavailable:
            first = unavailable[0]
            raise ValueError(
                f"{len(unavailable)} rationale rows do not map to the wrapped DROID split/filter index; "
                f"first unavailable sample is {first.sample_id!r} at "
                f"episode={first.episode_index}, start_frame={first.start_frame}."
            )
        if not mapped:
            raise ValueError(
                "No rationale rows map to the wrapped DROID split/filter index. Check the DROID root, "
                "episode split seed, window length, and filtering configuration."
            )
        if _should_log_dataset_summary():
            summary = (
                "DROID rationale sidecar mapping: "
                f"retained={len(mapped)}, total={len(annotations)}, dropped={len(unavailable)}, "
                f"strict_window_match={strict_window_match}"
            )
            if unavailable:
                log.warning(
                    f"{summary}. Dropped rows are outside the active split/filter index; "
                    "use strict_window_match=True to fail instead."
                )
            else:
                log.info(summary)
        self._items = tuple(
            sorted(mapped, key=lambda item: (item.annotation.episode_index, item.annotation.start_frame))
        )
        self._unavailable_count = len(unavailable)

    @property
    def unavailable_count(self) -> int:
        return self._unavailable_count

    def __len__(self) -> int:
        return len(self._items)

    def _raw_index(self, idx: int) -> int:
        """Expose this view's index contract to stratified metadata scans."""

        idx = int(idx)
        if idx < 0:
            idx += len(self._items)
        if idx < 0 or idx >= len(self._items):
            raise IndexError(f"DROID rationale index {idx} is out of range for length {len(self._items)}.")
        return idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._items[self._raw_index(idx)]
        annotation = item.annotation
        return self._dataset.reasoner_rationale_item(
            item.source_index,
            reason=annotation.reason,
            subtask=annotation.subtask,
            history_offsets=annotation.history_offsets,
        )

    def _metadata_for_raw_index(self, idx: int) -> tuple[DROIDSubgoalWindow, _RationaleLookup]:
        item = self._items[self._raw_index(idx)]
        window, _ = self._dataset._metadata_for_raw_index(item.source_index)
        target = format_reasoner_rationale_target(
            reason=item.annotation.reason,
            subtask=item.annotation.subtask,
        )
        return window, _RationaleLookup(text=target)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        while start < len(self._items):
            episode_index = self._items[start].annotation.episode_index
            end = start + 1
            while end < len(self._items) and self._items[end].annotation.episode_index == episode_index:
                end += 1
            blocks.append((start, end - start))
            start = end
        return blocks


__all__ = [
    "DROIDStartFrameBlockIndex",
    "DROIDReasonerRationaleAnnotation",
    "DROIDStartFrameRationaleDataset",
    "build_droid_start_frame_block_index",
    "find_droid_start_frame_source_index",
    "load_droid_reasoner_rationale_annotations",
]
