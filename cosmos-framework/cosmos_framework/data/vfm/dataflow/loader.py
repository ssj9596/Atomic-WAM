# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CosmosDataLoader — slim orchestrator that wires the four dataflow roles
(DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator) inside
each DataLoader worker. The canonical training dataloader.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data

from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)
from cosmos_framework.data.vfm.dataflow.batchers import SequentialPackingBatcher, SimpleBatcher
from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator
from cosmos_framework.utils import log


def _is_memory_heavy_reasoner_distributor(distributor: DataDistributor) -> bool:
    """Return whether a distributor owns the large DROID Reasoner stream.

    Reasoner samples decode several video frames and carry multimodal tensors.
    With forked workers, a large number of prefetched samples causes each
    worker to dirty substantial portions of the inherited dataset/decoder
    state.  The resulting RSS multiplication is much larger than the compact
    sampler index itself.  Keep this guard local to the stratified Reasoner
    stream; ordinary datasets retain their configured worker parallelism.
    """

    cached = getattr(distributor, "_cached", None)
    if cached is None:
        return False
    return cached.__class__.__name__ == "StratifiedReasonerIterableDataset"


def _supports_ordered_epoch_boundary(batcher: SampleBatcher) -> bool:
    """Whether ``batcher`` has no lookahead/reordering state at a boundary."""

    if isinstance(batcher, SimpleBatcher):
        return True
    # Token-budget SequentialPackingBatcher can carry skipped samples across
    # groups.  Restrict the exception to its count-only mode, which is the WAM
    # recipe and has no carry/skip path.
    return isinstance(batcher, SequentialPackingBatcher) and batcher.max_sequence_length is None


def _is_ordered_epoch_boundary(epochs: list[int], positions: list[int]) -> bool:
    """Return whether a batch follows one uninterrupted stream across epochs.

    ``ActionIterableShuffleDataset`` resets ``_dp_stream_pos`` to zero exactly
    when it increments ``_dp_epoch``.  For an order-preserving batcher, a batch
    that happens to straddle that point is still resumable: the cursor is the
    *last* sample in the new epoch.  This helper deliberately requires the
    metadata to be in source order, with contiguous positions within each epoch
    and a zero position at every epoch transition.  A pool/reordering batcher
    therefore cannot use this exception merely by containing the same values.
    """

    if len(epochs) != len(positions) or len(epochs) < 2 or len(set(epochs)) < 2:
        return False

    previous_epoch = int(epochs[0])
    previous_position = int(positions[0])
    for epoch, position in zip(epochs[1:], positions[1:], strict=True):
        epoch = int(epoch)
        position = int(position)
        if epoch == previous_epoch:
            if position != previous_position + 1:
                return False
        elif epoch == previous_epoch + 1 and position == 0:
            # The stream has just rolled to the next shuffled epoch.
            pass
        else:
            return False
        previous_epoch = epoch
        previous_position = position
    return True


class _DataflowIterableDataset(torch.utils.data.IterableDataset):
    """Wires distributor -> processor -> batcher -> collator inside a worker."""

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher,
        collator: BatchCollator,
        dp_rank: int,
        dp_world_size: int,
    ):
        super().__init__()
        self._distributor = distributor
        self._processor = processor
        self._batcher = batcher
        self._collator = collator
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)

        def _processed():
            for item in raw:
                if isinstance(item, dict):
                    meta = {k: item.pop(k) for k in list(item) if k.startswith("_dp_")}
                else:
                    meta = {}
                s = self._processor.process(item)
                if meta and isinstance(s, dict):
                    s.update(meta)
                yield s

        for group in self._batcher.batches(_processed()):
            has_meta = bool(group) and isinstance(group[0], dict) and "_dp_epoch" in group[0]
            if has_meta:
                epochs = [s["_dp_epoch"] for s in group]
                positions = [s["_dp_stream_pos"] for s in group]
                max_epoch = max(epochs)
                max_pos = max(positions)
                cursor_epoch = max_epoch
                cursor_pos = max_pos
                # Resume records a single cursor and fast-forwards to cursor+1.  A
                # contiguous single-epoch batch is safe as before.  An ordered
                # count-only batcher may also straddle an epoch boundary; that
                # case is handled below using the final (epoch, position) pair.
                # Reordering/lookahead batches remain rejected rather than
                # silently dropping samples on resume.
                if len(group) > 1:
                    contiguous = min(epochs) == max_epoch and sorted(positions) == list(
                        range(min(positions), max_pos + 1)
                    )
                    ordered_epoch_boundary = _supports_ordered_epoch_boundary(
                        self._batcher
                    ) and _is_ordered_epoch_boundary(epochs, positions)
                    if not contiguous and not ordered_epoch_boundary:
                        raise ValueError(
                            "Map-style resume cannot safely stamp a multi-sample batch whose "
                            "_dp_stream_pos values are non-contiguous or span epochs (reordering "
                            "batcher + batch_size>1). Use max_batch_size=1 with pool packing, a "
                            "sequential (order-preserving) batcher, or an iterable (non-resumable) "
                            "source."
                        )
                    if ordered_epoch_boundary:
                        # ``max_pos`` belongs to the previous epoch and would
                        # skip almost the entire new epoch on resume.  The
                        # stream's final item is the exact cursor for this
                        # ordered cross-epoch batch.
                        cursor_epoch = int(epochs[-1])
                        cursor_pos = int(positions[-1])
                clean = [{k: v for k, v in s.items() if not k.startswith("_dp_")} for s in group]
                batch = self._collator.collate(clean)
                batch["sample_worker_id"] = torch.tensor([worker_id] * len(group))
                batch["sample_epoch"] = torch.tensor([cursor_epoch] * len(group))
                batch["sample_index"] = torch.tensor([cursor_pos] * len(group))
            else:
                batch = self._collator.collate(group)
            yield batch


class CosmosDataLoader(torch.utils.data.DataLoader):
    """Public entry point: bring any dataset into training via four roles.

    Either pass an explicit ``batcher`` (and optional ``collator``), or pass a
    bare ``batch_size=N`` for stock fixed-size batching — the loader then builds
    ``SimpleBatcher(N)`` + ``DefaultBatchCollator()``. Passing both is an error.

    DP coordinates: ``parallel_dims.dp_coord`` > ``torch.distributed`` > (0, 1).
    """

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher | None = None,
        collator: BatchCollator | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        parallel_dims=None,
    ):
        if batch_size is not None and batcher is not None:
            raise ValueError("Pass either batch_size= (sugar) or an explicit batcher=, not both.")
        if batch_size is None and batcher is None:
            raise ValueError("Provide either a batcher= or a batch_size=.")
        if batch_size is not None:
            batcher = SimpleBatcher(batch_size=batch_size)
        if collator is None:
            collator = DefaultBatchCollator()

        if parallel_dims is not None:
            dp_rank, dp_world_size = parallel_dims.dp_coord
        elif torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()
            if dp_world_size > 1:
                log.info(
                    "CosmosDataLoader: using global rank for DP sharding. "
                    "For FSDP+TP/PP pass parallel_dims= for the correct DP rank.",
                    rank0_only=True,
                )
        else:
            dp_rank, dp_world_size = 0, 1

        # RankPartitionedDistributor may receive lazy dataset configs.  Its
        # dataset must be materialized in this parent/rank process, before the
        # torch DataLoader workers are spawned; otherwise each worker would
        # independently build the same large index.  Other distributors do not
        # implement this optional hook and retain their existing lazy semantics.
        prepare_for_rank = getattr(distributor, "prepare_for_rank", None)
        if prepare_for_rank is not None:
            if not callable(prepare_for_rank):
                raise TypeError("distributor.prepare_for_rank must be callable when present.")
            prepare_for_rank(dp_rank, dp_world_size)

        # Reasoner batches are assembled inside each worker.  With batch_size
        # 16, eight workers and prefetch=4 can retain 512 decoded multimodal
        # samples per rank, unlike the WAM loader whose inner DataLoader emits
        # one sample at a time.  Keep one worker and one prefetched batch: the
        # bounded torchcodec LRU handles decoder lifetime, while this cap
        # bounds in-flight frame tensors without moving decode onto the rank's
        # training thread. This must precede prepare_for_loader(), which sees
        # the final worker geometry.
        if num_workers > 1 and _is_memory_heavy_reasoner_distributor(distributor):
            log.warning(
                "CosmosDataLoader: limiting stratified Reasoner to "
                "num_workers=1, prefetch_factor=1 to bound decoded batch memory "
                f"from the large DROID/video dataset (requested num_workers={num_workers}, "
                f"prefetch_factor={prefetch_factor}).",
                rank0_only=True,
            )
            num_workers = 1
            prefetch_factor = 1

        # Some iterable datasets need the final rank x worker geometry to
        # materialize a compact epoch index.  Run that optional hook in the
        # parent, immediately before worker processes are created; doing it in
        # ``__iter__`` would duplicate the index (and any temporary buffers)
        # in every persistent worker.
        prepare_for_loader = getattr(distributor, "prepare_for_loader", None)
        if prepare_for_loader is not None:
            if not callable(prepare_for_loader):
                raise TypeError("distributor.prepare_for_loader must be callable when present.")
            prepare_for_loader(num_workers=num_workers)

        dataset = _DataflowIterableDataset(
            distributor=distributor,
            processor=processor,
            batcher=batcher,
            collator=collator,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

        from cosmos_framework.data.vfm.dataflow.distributors import MapDistributor

        if isinstance(distributor, MapDistributor) and num_workers > 0 and not persistent_workers:
            log.info(
                "CosmosDataLoader: MapDistributor requires persistent_workers=True for "
                "correct stateful resume; overriding to True.",
                rank0_only=True,
            )
            persistent_workers = True

        if persistent_workers and num_workers == 0:
            log.info(
                "CosmosDataLoader: persistent_workers=True ignored because num_workers=0.",
                rank0_only=True,
            )
            persistent_workers = False

        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(dataset, batch_size=None, **loader_kwargs)


class JointCosmosDataLoader:
    """Wraps multiple ``CosmosDataLoader`` instances with ratio-based seeded selection.

    One output batch = one inner loader, selected deterministically by ratio at each
    step.  Adds a ``"dataset_name"`` key to every yielded batch so downstream callbacks
    can route state updates to the correct inner loader.

    Parameters
    ----------
    dataloaders:
        ``{name: {"dataloader": CosmosDataLoader, "ratio": int}}`` mapping.
        Entries with ``ratio <= 0`` are silently skipped.
    seed:
        Base seed for per-step dataset selection.  Step ``i`` uses
        ``np.random.RandomState(seed + i)`` — fully reproducible on resume via
        ``set_start_iteration``.
    """

    def __init__(
        self,
        dataloaders: dict,
        seed: int = 42,
    ) -> None:
        entries = [
            (name, cfg["dataloader"], cfg["ratio"]) for name, cfg in dataloaders.items() if cfg.get("ratio", 0) > 0
        ]
        if not entries:
            raise ValueError("JointCosmosDataLoader: no dataloaders with ratio > 0")

        self._names: list[str] = [e[0] for e in entries]
        if "global_id" in self._names:
            raise ValueError(
                "JointCosmosDataLoader: dataset name 'global_id' is reserved "
                "by the checkpoint state format; use a different name."
            )
        self._loaders: list[CosmosDataLoader] = [e[1] for e in entries]
        ratios = np.array([e[2] for e in entries], dtype=float)
        self._probs: np.ndarray = ratios / ratios.sum()
        self._seed = seed
        self._global_id = 0
        # Iterators are created lazily on the first __iter__ call so that
        # DataLoaderStateCallback.load_state_dict can install resume env vars
        # before workers are spawned (for num_workers > 0, iter(DataLoader)
        # forks workers immediately; env vars must be set in the parent first).
        self._iterators: list | None = None

        total = ratios.sum()
        lines = [f"JointCosmosDataLoader: {len(self._names)} streams"]
        for name, ratio in zip(self._names, ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

    def set_start_iteration(self, iteration: int) -> None:
        """Restore deterministic selection sequence after checkpoint resume.

        Called by ``JointCosmosDataLoaderStateCallback.load_state_dict`` and by the
        trainer (if present) via ``hasattr`` guard.
        """
        self._global_id = iteration

    def __iter__(self):
        # Lazy init: create iterators here (not in __init__) so that
        # load_state_dict can set resume env vars before workers fork.
        if self._iterators is None:
            self._iterators = [iter(loader) for loader in self._loaders]
        while True:
            rng = np.random.RandomState(self._seed + self._global_id)
            idx = int(rng.choice(len(self._loaders), p=self._probs))
            try:
                batch = next(self._iterators[idx])
            except StopIteration:
                # Inner CosmosDataLoaders are infinite; this guard handles
                # the unlikely case of a finite IterableDataset inner source.
                self._iterators[idx] = iter(self._loaders[idx])
                batch = next(self._iterators[idx])
            batch["dataset_name"] = self._names[idx]
            self._global_id += 1
            yield batch
