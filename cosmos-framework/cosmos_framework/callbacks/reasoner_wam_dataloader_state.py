# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint state for the paired Reasoner/WAM iterable streams."""

from __future__ import annotations

from typing import Any

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


def _task_name(batch: dict[str, Any]) -> str | None:
    marker = batch.get("dataset_name", batch.get("task_type"))
    if isinstance(marker, str):
        return marker.strip().lower()
    if isinstance(marker, (list, tuple)):
        values = {str(value).strip().lower() for value in marker}
        if len(values) == 1:
            return next(iter(values))
    return None


class ReasonerWAMPairedDataLoaderStateCallback(Callback):
    """Persist per-worker cursors for both child ``CosmosDataLoader`` streams.

    The paired loader itself is resumed by the trainer's logical ``iteration``;
    this callback restores each child's worker ``(epoch, index)`` environment
    before the lazy child iterators are created.
    """

    checkpoint_component: str = "dataloader"

    def __init__(self) -> None:
        super().__init__()
        self._inner = {
            "reasoner": CosmosDataLoaderStateCallback(name="reasoner"),
            "wam": CosmosDataLoaderStateCallback(name="wam"),
        }

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        name = _task_name(data_batch)
        callback = self._inner.get(name or "")
        if callback is not None:
            callback._update_state_from_batch(data_batch)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if self.config and iteration % self.config.trainer.logging_iter == 0:
            log.info(f"Reasoner/WAM paired loader state: logical_step={iteration}")

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[str, Any]:
        return {name: callback.state_dict() for name, callback in self._inner.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        for name, callback in self._inner.items():
            callback.load_state_dict(state.get(name, {}))


__all__ = ["ReasonerWAMPairedDataLoaderStateCallback"]
