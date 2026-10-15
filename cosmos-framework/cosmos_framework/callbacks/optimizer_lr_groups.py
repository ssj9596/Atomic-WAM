# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from typing import Any

import torch
import wandb

from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback


class OptimizerLRGroupsLogger(Callback):
    """Log unique optimizer param-group learning rates to W&B."""

    def __init__(self, every_n: int = 1, log_first: bool = True) -> None:
        super().__init__()
        self.every_n = int(every_n)
        self.log_first = bool(log_first)

    @staticmethod
    def _iter_optimizers(optimizer: Any):
        """Yield raw torch optimizers from arbitrarily nested containers.

        The regular FSDP-aware :class:`OptimizersContainer` exposes a flat
        ``optimizers`` list.  The opt-in Reasoner/WAM trainer wraps two of
        those containers, so its public ``optimizers`` attribute is a
        two-dimensional list.  A shallow iteration would hand an inner list
        to the logging loop and fail with ``AttributeError: 'list' object has
        no attribute 'param_groups'``.  Keep this callback compatible with
        both shapes (and with future composite containers) by recursively
        walking container/list/tuple nodes and yielding only objects that
        actually expose optimizer ``param_groups``.
        """

        if optimizer is None:
            return
        if hasattr(optimizer, "param_groups"):
            yield optimizer
            return
        children = getattr(optimizer, "optimizers", None)
        if children is None and isinstance(optimizer, (list, tuple)):
            children = optimizer
        if children is None:
            raise TypeError(
                "OptimizerLRGroupsLogger expected a torch optimizer or a nested "
                f"optimizer container, got {type(optimizer).__name__}."
            )
        for child in children:
            yield from OptimizerLRGroupsLogger._iter_optimizers(child)

    def on_before_optimizer_step(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model, scheduler, grad_scaler
        if not distributed.is_rank0() or wandb.run is None:
            return
        gate = self.config.trainer.logging_iter * self.every_n
        if not (self.log_first and iteration <= 0) and not (gate > 0 and iteration % gate == 0):
            return

        lrs = []
        for optim in self._iter_optimizers(optimizer):
            for group in optim.param_groups:
                lrs.append(float(group["lr"]))
        if not lrs:
            return

        unique_lrs = sorted(set(round(lr, 12) for lr in lrs))
        info = {
            "optim/lr_group_min": min(lrs),
            "optim/lr_group_max": max(lrs),
            "optim/lr_group_count": len(lrs),
            "optim/lr_unique_count": len(unique_lrs),
        }
        for idx, lr in enumerate(unique_lrs):
            info[f"optim/lr_unique_{idx}"] = lr
        wandb.log(info, step=iteration)
