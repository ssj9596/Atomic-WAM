# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpointable containers for two disjoint training phases.

The ordinary Cosmos trainer accepts one optimizer/scheduler pair.  The
Reasoner/WAM recipe deliberately performs two forwards in one logical step,
so it needs one optimizer for the VLM LoRA parameters and another for the WAM
generation parameters.  These thin containers keep the public checkpoint
contract singular while exposing each phase to the specialized trainer.
"""

from __future__ import annotations

from typing import Any, Iterator

from torch.distributed.checkpoint.stateful import Stateful

from cosmos_framework.utils.vfm.optimizer import LRSchedulersContainer, OptimizersContainer


class DualOptimizersContainer(Stateful):
    """Hold independent Reasoner and WAM optimizer containers.

    ``OptimizersContainer`` may itself contain one optimizer per FSDP device
    mesh.  The phase accessors intentionally return those containers rather
    than a raw ``torch.optim.Optimizer`` so FSDP-aware state handling remains
    unchanged.
    """

    def __init__(self, reasoner: OptimizersContainer, wam: OptimizersContainer) -> None:
        if len(reasoner) == 0 or len(wam) == 0:
            raise ValueError("DualOptimizersContainer requires non-empty Reasoner and WAM optimizers.")
        self.reasoner = reasoner
        self.wam = wam
        # A few callbacks inspect ``.optimizers`` and ``.model_part_names``.
        # Keep the nested shape used by the VLM ``LearningRateLogger``:
        # one list per logical model part, with one raw optimizer per mesh.
        self.optimizers = [reasoner.optimizers, wam.optimizers]
        self.model_part_names = ["reasoner_lora", "wam"]

    def phase(self, name: str) -> OptimizersContainer:
        if name == "reasoner":
            return self.reasoner
        if name == "wam":
            return self.wam
        raise KeyError(f"Unknown dual optimizer phase {name!r}; expected 'reasoner' or 'wam'.")

    def __iter__(self) -> Iterator[Any]:
        return iter([*self.reasoner.optimizers, *self.wam.optimizers])

    def __len__(self) -> int:
        return len(self.reasoner) + len(self.wam)

    def step(self) -> None:
        self.reasoner.step()
        self.wam.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.reasoner.zero_grad(set_to_none=set_to_none)
        self.wam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "reasoner_wam_dual_v1",
            "reasoner": self.reasoner.state_dict(),
            "wam": self.wam.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "reasoner" not in state_dict or "wam" not in state_dict:
            raise ValueError(
                "DualOptimizersContainer expected a checkpoint with 'reasoner' and 'wam' optimizer states."
            )
        self.reasoner.load_state_dict(state_dict["reasoner"])
        self.wam.load_state_dict(state_dict["wam"])


class DualLRSchedulersContainer(Stateful):
    """Scheduler counterpart to :class:`DualOptimizersContainer`."""

    def __init__(self, reasoner: LRSchedulersContainer, wam: LRSchedulersContainer) -> None:
        self.reasoner = reasoner
        self.wam = wam

    def phase(self, name: str) -> LRSchedulersContainer:
        if name == "reasoner":
            return self.reasoner
        if name == "wam":
            return self.wam
        raise KeyError(f"Unknown dual scheduler phase {name!r}; expected 'reasoner' or 'wam'.")

    def __iter__(self) -> Iterator[Any]:
        return iter([*self.reasoner.schedulers, *self.wam.schedulers])

    def __len__(self) -> int:
        return len(self.reasoner) + len(self.wam)

    def step(self) -> None:
        self.reasoner.step()
        self.wam.step()

    def get_last_lr(self) -> list[Any]:
        return [*self.reasoner.get_last_lr(), *self.wam.get_last_lr()]

    def get_lr(self) -> list[Any]:
        return [*self.reasoner.get_lr(), *self.wam.get_lr()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "reasoner_wam_dual_v1",
            "reasoner": self.reasoner.state_dict(),
            "wam": self.wam.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "reasoner" not in state_dict or "wam" not in state_dict:
            raise ValueError(
                "DualLRSchedulersContainer expected a checkpoint with 'reasoner' and 'wam' scheduler states."
            )
        self.reasoner.load_state_dict(state_dict["reasoner"])
        self.wam.load_state_dict(state_dict["wam"])


__all__ = ["DualLRSchedulersContainer", "DualOptimizersContainer"]
