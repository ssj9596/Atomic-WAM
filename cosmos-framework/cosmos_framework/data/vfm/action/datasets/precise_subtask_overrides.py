# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Point-sampled precise subgoal overrides for WAM/Reasoner windows."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PreciseSubtaskPoint:
    episode_index: int
    start_frame: int
    subtask: str


@dataclass(frozen=True, slots=True)
class PreciseSubtaskEpisode:
    starts: tuple[int, ...]
    subtasks: tuple[str, ...]

    def in_range(self, start: int, end: int) -> tuple[tuple[int, str], ...]:
        left = bisect_left(self.starts, int(start))
        right = bisect_right(self.starts, int(end))
        return tuple(zip(self.starts[left:right], self.subtasks[left:right], strict=True))


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_precise_subtask_points(path: str | Path) -> dict[int, PreciseSubtaskEpisode]:
    """Load point annotations keyed by episode and current-frame start.

    The 40,721-row precise manifest is intentionally not interpreted as a
    contiguous annotation file: it has point samples and no ``end_frame``.
    Duplicate keys with conflicting text are rejected conservatively.
    """

    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"precise subtask annotations path does not exist: {annotation_path}")
    text = annotation_path.read_text(encoding="utf-8")
    raw_rows: list[Any]
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("precise annotations JSON must contain an array")
        raw_rows = value
    else:
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    points: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise ValueError(f"precise annotation row {row_index} is not an object")
        try:
            episode_index = int(row["episode_index"])
            start_frame = int(row["start_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"precise annotation row {row_index} has invalid episode/start frame") from exc
        subtask = _clean(row.get("subtask"))
        if episode_index < 0 or start_frame < 0 or not subtask:
            raise ValueError(f"precise annotation row {row_index} has invalid values")
        key = (episode_index, start_frame)
        previous = points.get(key)
        if previous is not None and previous.casefold() != subtask.casefold():
            raise ValueError(
                "conflicting precise subtask annotations for "
                f"episode_index={episode_index}, start_frame={start_frame}: {previous!r} vs {subtask!r}"
            )
        points[key] = subtask

    grouped: dict[int, list[tuple[int, str]]] = {}
    for (episode_index, start_frame), subtask in points.items():
        grouped.setdefault(episode_index, []).append((start_frame, subtask))
    return {
        episode_index: PreciseSubtaskEpisode(
            starts=tuple(start for start, _ in values),
            subtasks=tuple(subtask for _, subtask in values),
        )
        for episode_index, values in (
            (episode_index, sorted(items)) for episode_index, items in grouped.items()
        )
    }


__all__ = ["PreciseSubtaskEpisode", "PreciseSubtaskPoint", "load_precise_subtask_points"]
