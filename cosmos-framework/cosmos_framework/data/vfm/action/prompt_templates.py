# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared text prompts for DROID action prediction and Reasoner supervision."""

from __future__ import annotations

ACTION_PREDICTION_PROMPT_TEMPLATE = (
    "Using the current visual observation and the task as context, predict the executable robot action plan for the "
    "next 2 seconds.\n"
    "The upper half of the image shows the wrist-mounted camera view. The lower half contains two side views: the "
    "left-side view is on the left, and the right-side view is on the right.\n"
    "Task: {task}\n"
    "{preview_action}"
    "{planning_instruction}"
)
SEPARATE_VIEW_ACTION_PREDICTION_PROMPT_TEMPLATE = (
    "Use the camera images and the task as context to predict the executable robot action plan for the next 2 seconds.\n"
    "For each timepoint, the three images are ordered LEFT exterior view, RIGHT exterior view, then WRIST view.\n"
    "Task: {task}\n"
    "{preview_action}"
    "{planning_instruction}"
)
MID_SOFT_TASK_PROMPT_TEMPLATE = ACTION_PREDICTION_PROMPT_TEMPLATE


def format_action_prediction_prompt(
    task_instruction: str,
    preview_action: str | None = None,
    *,
    separate_views: bool = False,
    require_think: bool = False,
) -> str:
    """Wrap a task in the shared visual action-prediction supervision prompt."""

    task = task_instruction.strip()
    if not task:
        return ""
    preview = ""
    planning_instruction = "Next 2-second executable action plan:"
    if preview_action is not None and str(preview_action).strip():
        preview = f"Previous action pred: {str(preview_action).strip()}\n"
        planning_instruction = (
            "Based on the previous action prediction and past execution results, determine the best "
            "next 2-second executable action plan:"
        )
    if require_think:
        planning_instruction = (
            "First, analyze the current state inside <think> and </think>. "
            "After </think>, write the final executable 2-second action plan that is most likely to complete the task:"
        )
    template = SEPARATE_VIEW_ACTION_PREDICTION_PROMPT_TEMPLATE if separate_views else ACTION_PREDICTION_PROMPT_TEMPLATE
    return template.format(
        task=task,
        preview_action=preview,
        planning_instruction=planning_instruction,
    )


def format_mid_soft_task_prompt(task_instruction: str) -> str:
    """Backward-compatible name for :func:`format_action_prediction_prompt`."""

    return format_action_prediction_prompt(task_instruction)


__all__ = [
    "ACTION_PREDICTION_PROMPT_TEMPLATE",
    "SEPARATE_VIEW_ACTION_PREDICTION_PROMPT_TEMPLATE",
    "MID_SOFT_TASK_PROMPT_TEMPLATE",
    "format_action_prediction_prompt",
    "format_mid_soft_task_prompt",
]
