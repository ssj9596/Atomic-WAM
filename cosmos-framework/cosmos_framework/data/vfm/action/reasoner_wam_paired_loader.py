# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""A small, opt-in loader that advances Reasoner and WAM streams in parallel.

The existing heterogeneous loaders deliberately select *one* stream for each
yield.  That is useful for an alternating single-forward recipe, but it cannot
represent a training step that performs one Reasoner forward and one WAM
forward on independently shuffled samples.  :class:`ReasonerWAMPairedDataLoader`
is the intentionally minimal counterpart for that recipe:

* each child is instantiated and iterated independently;
* one yield contains ``reasoner_batch`` and ``wam_batch`` (never a merged
  schema), plus a monotonically increasing ``pair_step``;
* finite children are not silently restarted when the other child continues;
* the logical pair counter and any child loader state are checkpointable.

This module is not wired into an existing experiment by default.  Keeping it
separate is important for legacy launchers, whose loaders and batch schemas are
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from cosmos_framework.utils.lazy_config import instantiate


def _resolve_child_loader(value: Any, *, name: str) -> Any:
    """Resolve a child loader supplied as an object or LazyConfig mapping.

    A ratio-entry mapping (``{"dataloader": ..., "ratio": ...}``) is accepted
    for convenience when reusing an existing Hydra data-loader fragment.  The
    ratio is intentionally ignored: this loader always advances both children.
    """

    candidate = value
    if isinstance(candidate, Mapping) and "_target_" not in candidate and "dataloader" in candidate:
        candidate = candidate["dataloader"]
    candidate = instantiate(candidate)
    if isinstance(candidate, Mapping) and "_target_" not in candidate and "dataloader" in candidate:
        # ``instantiate`` leaves plain mappings untouched; unwrap one more time
        # in case a regular dict held a LazyConfig child.
        candidate = instantiate(candidate["dataloader"])
    if not hasattr(candidate, "__iter__"):
        raise TypeError(
            f"{name} child must be an iterable/DataLoader or a LazyConfig mapping, got {type(candidate).__name__}."
        )
    return candidate


def _call_optional(loader: Any, method_name: str, *args: Any) -> Any:
    """Call an optional child method, returning ``None`` when it is absent."""

    method = getattr(loader, method_name, None)
    if method is None:
        return None
    if not callable(method):
        raise TypeError(f"Child loader attribute {method_name!r} must be callable when present.")
    return method(*args)


class ReasonerWAMPairedDataLoader:
    """Yield one independently sampled Reasoner batch and WAM batch together.

    Args:
        reasoner_loader: An instantiated iterable/DataLoader, a LazyConfig
            mapping, or a ratio-entry mapping containing ``dataloader``.
        wam_loader: The corresponding WAM child.

    Child iterators are created lazily on the first call to :meth:`__iter__`.
    This mirrors ``JointCosmosDataLoader`` and lets checkpoint restore code call
    :meth:`set_start_iteration` before worker processes/iterators are spawned.
    ``StopIteration`` from either finite child ends the paired stream; an
    exhausted child is never silently recycled against a different sample.
    """

    _STATE_VERSION = 1

    def __init__(self, reasoner_loader: Any, wam_loader: Any) -> None:
        self.reasoner_loader = _resolve_child_loader(reasoner_loader, name="reasoner")
        self.wam_loader = _resolve_child_loader(wam_loader, name="wam")
        self._iterators: tuple[Iterator[Any], Iterator[Any]] | None = None
        self._pair_step = 0

    @property
    def pair_step(self) -> int:
        """Number of complete pairs yielded so far (the next pair's id)."""

        return self._pair_step

    def __len__(self) -> int:
        """Return the shorter child length when both lengths are available."""

        lengths: list[int] = []
        for name, loader in (("reasoner", self.reasoner_loader), ("wam", self.wam_loader)):
            try:
                lengths.append(len(loader))
            except TypeError as exc:
                raise TypeError(f"{name} child does not expose a finite length.") from exc
        return min(lengths)

    def set_start_iteration(self, iteration: int) -> None:
        """Set the logical pair counter and forward it to both child loaders.

        Child loaders that implement ``set_start_iteration`` receive the same
        value.  Plain ``torch.utils.data.DataLoader`` instances do not expose
        that method and are simply left unchanged.  Existing iterators are
        discarded so a subsequent ``iter(...)`` observes the restored child
        state.
        """

        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError(f"iteration must be an integer, got {iteration!r}.")
        if iteration < 0:
            raise ValueError(f"iteration must be non-negative, got {iteration}.")
        self._pair_step = iteration
        _call_optional(self.reasoner_loader, "set_start_iteration", iteration)
        _call_optional(self.wam_loader, "set_start_iteration", iteration)
        self._iterators = None

    def __iter__(self):
        if self._iterators is None:
            self._iterators = (iter(self.reasoner_loader), iter(self.wam_loader))
        reasoner_iter, wam_iter = self._iterators
        while True:
            # Fetch both records before mutating the counter.  If either finite
            # child is exhausted, no partially numbered pair is observable.
            try:
                reasoner_batch = next(reasoner_iter)
                wam_batch = next(wam_iter)
            except StopIteration:
                # A bare StopIteration escaping a generator is rewritten as a
                # RuntimeError by PEP 479; return explicitly so callers see the
                # normal iterator exhaustion signal.
                return
            pair_step = self._pair_step
            self._pair_step += 1
            yield {
                "reasoner_batch": reasoner_batch,
                "wam_batch": wam_batch,
                "pair_step": pair_step,
            }

    def state_dict(self) -> dict[str, Any]:
        """Return a lightweight state payload suitable for a checkpoint callback."""

        return {
            "version": self._STATE_VERSION,
            "pair_step": self._pair_step,
            "reasoner": _call_optional(self.reasoner_loader, "state_dict") or {},
            "wam": _call_optional(self.wam_loader, "state_dict") or {},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore pair counter and delegate child state when supported."""

        if not isinstance(state, Mapping):
            raise TypeError(f"paired loader state must be a mapping, got {type(state).__name__}.")
        version = state.get("version", self._STATE_VERSION)
        if version != self._STATE_VERSION:
            raise ValueError(f"Unsupported paired loader state version {version!r}.")
        pair_step = state.get("pair_step", 0)
        if isinstance(pair_step, bool) or not isinstance(pair_step, int) or pair_step < 0:
            raise ValueError(f"state['pair_step'] must be a non-negative integer, got {pair_step!r}.")
        self._pair_step = pair_step
        reasoner_state = state.get("reasoner")
        if reasoner_state is not None:
            _call_optional(self.reasoner_loader, "load_state_dict", reasoner_state)
        wam_state = state.get("wam")
        if wam_state is not None:
            _call_optional(self.wam_loader, "load_state_dict", wam_state)
        self._iterators = None


__all__ = ["ReasonerWAMPairedDataLoader"]
