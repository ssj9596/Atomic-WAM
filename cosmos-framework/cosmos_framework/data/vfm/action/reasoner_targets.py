# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Formatting and parsing for optional DROID Reasoner rationale targets."""

from __future__ import annotations

import re
from dataclasses import dataclass

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_TARGET = re.compile(r"\A\s*<think>(?P<reason>.*?)</think>(?P<subtask>.*?)\s*\Z", re.DOTALL)
_MOVE_OPEN_GRIPPER_RE = re.compile(r"\bmove\s+the\s+open\s+gripper\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedReasonerTarget:
    """One assistant target split into optional reasoning and final subtask."""

    reason: str | None
    subtask: str


def _clean_target_part(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}.")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    if _THINK_OPEN in text or _THINK_CLOSE in text:
        raise ValueError(f"{field} must not contain {_THINK_OPEN!r} or {_THINK_CLOSE!r}.")
    if "<|im_start|>" in text or "<|im_end|>" in text:
        raise ValueError(f"{field} must not contain chat-template boundary tokens.")
    return text


def format_reasoner_rationale_target(*, reason: str, subtask: str) -> str:
    """Return the exact assistant target used for rationale SFT."""

    clean_reason = _clean_target_part(reason, field="reason")
    clean_subtask = _clean_target_part(subtask, field="subtask")
    return f"{_THINK_OPEN}{clean_reason}{_THINK_CLOSE}{clean_subtask}"


def parse_reasoner_target(value: str) -> ParsedReasonerTarget | None:
    """Parse a plain subtask or one strict ``<think>...</think>subtask`` target.

    Malformed or nested think tags return ``None``. This fail-closed behavior is
    shared by sampling and deployment so rationale text can never become an
    action-family label or a WAM caption.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    has_think_marker = _THINK_OPEN in text or _THINK_CLOSE in text
    if not has_think_marker:
        return ParsedReasonerTarget(reason=None, subtask=text)

    match = _THINK_TARGET.fullmatch(text)
    if match is None:
        return None
    reason = match.group("reason").strip()
    subtask = match.group("subtask").strip()
    if not reason or not subtask:
        return None
    if any(marker in reason or marker in subtask for marker in (_THINK_OPEN, _THINK_CLOSE)):
        return None
    return ParsedReasonerTarget(reason=reason, subtask=subtask)


def reasoner_final_subtask(value: str) -> str:
    """Return only the deployable final subtask, or ``""`` if malformed."""

    parsed = parse_reasoner_target(value)
    return parsed.subtask if parsed is not None else ""


def rewrite_move_open_gripper_text(text: str) -> str:
    """Make an open-gripper move explicit while preserving surrounding text.

    This is intentionally a text-only, opt-in rewrite for inference-time
    Reasoner captions.  It matches the same phrase used by the DROID subgoal
    data augmentation and leaves all other text and punctuation unchanged.
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


__all__ = [
    "ParsedReasonerTarget",
    "format_reasoner_rationale_target",
    "parse_reasoner_target",
    "reasoner_final_subtask",
    "rewrite_move_open_gripper_text",
]
