# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Distributed checkpoint (DCP) directory structure and storage backends.

The checkpointer saves model state in a sharded format across multiple processes:

self.save_dirname/
├── iter_000000005/                    # Checkpoint at iteration 5
│   ├── model/                         # Model state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── optim/                        # Optimizer state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── scheduler/                    # Learning rate scheduler state
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   └── trainer/                      # Additional training state
│       ├── __0_0.distcp              # Shard 0 from rank 0
│       └── __1_0.distcp              # Shard 1 from rank 1
│   └── dataloader/                   # Optional per-rank dataloader state
│       ├── rank_0.pkl
│       └── rank_1.pkl
└── latest_checkpoint.txt             # Points to most recent checkpoint folder, e.g. iter_000000005

Storage backends:
- Local filesystem:
  self.save_dirname = "{config_job.path_local}/checkpoints"

- S3 object store:
  self.save_dirname = "s3://{bucket}/{config_job.path}/checkpoints"
  where bucket = self.config_checkpoint.save_to_object_store.bucket

The sharded format enables efficient distributed saving/loading by:
1. Parallelizing I/O across processes
2. Reducing memory usage per process
3. Supporting both local and cloud storage backends
"""

import enum
import multiprocessing
import os
import re
import shutil
import time
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
    StorageMeta,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.checkpoint.base import AbstractCheckpointer
from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader, S3StorageWriter
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import callback, distributed, log, misc
from cosmos_framework.utils.config import CheckpointConfig, JobConfig
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.vfm.rand_state import get_rand_state_dict, set_rand_state_dict


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling. Strips away the _orig_mod. prefix
    among other things from the state dict keys.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: dict[str, Any]) -> _IncompatibleKeys:
        return set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


@runtime_checkable
class _DataloaderStateHandler(Protocol):
    """Structural contract for callbacks that participate in dataloader-state checkpointing."""

    checkpoint_component: str

    def has_checkpoint_state(self) -> bool: ...
    def state_dict(self) -> dict[Any, Any]: ...
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None: ...


class _DataloaderWrapper:
    """Adapter that surfaces a dataloader-state callback's checkpoint API.

    Walks the registered callbacks at construction time and binds to the
    first callback that:

    1. Declares ``checkpoint_component == "dataloader"``, AND
    2. Returns ``True`` from ``has_checkpoint_state()``.

    The bound callback's ``state_dict`` / ``load_state_dict`` methods are
    re-exposed via :meth:`state_dict` / :meth:`load_state_dict`.  Callers
    must gate those on :meth:`has_state` — invoking them when nothing was
    bound raises :class:`RuntimeError`.

    Note: only the first callback tagged ``checkpoint_component=="dataloader"``
    is considered; if it does not currently want its state checkpointed,
    no further callbacks are searched.  In practice there is at most one
    such callback (see ``DataLoaderStateCallback``).
    """

    def __init__(self, callbacks: callback.CallBackGroup | None) -> None:
        self._callback: _DataloaderStateHandler | None = None
        if callbacks is None:
            return
        for current_callback in callbacks._callbacks:
            if getattr(current_callback, "checkpoint_component", None) != "dataloader":
                continue
            if current_callback.has_checkpoint_state():
                self._callback = current_callback
            return

    def has_state(self) -> bool:
        return self._callback is not None

    def state_dict(self) -> dict[Any, Any]:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot save dataloader state.")
        return self._callback.state_dict()

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot load dataloader state.")
        self._callback.load_state_dict(state_dict)


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    config_checkpoint: CheckpointConfig,
    config_job: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        config_checkpoint: Configuration settings for checkpoint saving behavior
        config_job: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo")

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(
        config_checkpoint=config_checkpoint,
        config_job=config_job,
        callbacks=None,
        disable_async=True,
    )

    while True:
        log.info(f"Checkpoint background process is ready for next task, waiting for new state_dict")
        received_data = receiver_queue.get()
        log.info(f"Checkpoint background process received new state_dict")

        if isinstance(received_data, Terminate):
            log.info(f"Checkpoint background process received termination signal, closing sender queue")
            break

        assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
        state_dict, checkpoint_path = received_data

        # Save checkpoint and measure time taken.
        start_time = time.monotonic()
        iteration = state_dict["trainer"][0]["iteration"]
        succeeded = False

        try:
            log.info(f"Saving checkpoint to {checkpoint_path}")
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
            succeeded = True
        except Exception as e:
            log.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
            # continue because if the thread exits, the main thread keeps on adding to the queue
        finally:
            elapsed_time = time.monotonic() - start_time
            log.info(
                f"Checkpoint save completed in background process. "
                f"Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}, "
                f"status: {'SUCCESS' if succeeded else 'FAILURE'}"
            )
            sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    log.info("Cleaning up: destroying distributed process group")
    dist.destroy_process_group()


def _replace_keys_with_ema_keys(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """
    Renames model parameters from "net." to "net_ema.".
    """
    if not all(k.startswith("net.") for k in state_dict.keys()):
        raise ValueError("State dict must start with net. keys when load_ema_to_reg is True")
    return {k.replace("net.", "net_ema."): v for k, v in state_dict.items()}


class CustomLoadPlanner(dcp.DefaultLoadPlanner):
    """
    CustomLoadPlanner that supports ignoring keys during checkpoint load.
    This is useful when the checkpoint is saved with a different component
    architecture, e.g. different RoPE embeddings than the current model.
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
        keys_to_skip_loading: List[str] = [],
        load_ema_to_reg: bool = False,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            allow_partial_load=allow_partial_load,
        )
        self.keys_to_skip_loading = keys_to_skip_loading
        self.load_ema_to_reg = load_ema_to_reg
        if len(keys_to_skip_loading) > 0:
            log.info(f"Skipping loading of keys that match the following patterns: {keys_to_skip_loading}")

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        state_dict = self._skip_keys_if_found(state_dict)

        if self.load_ema_to_reg:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

    def _skip_keys_if_found(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Dict[str, Any]:
        """
        While loading the checkpoint, skip the weight loading for the keys
        that contain any element of `self.keys_to_skip_loading` as a substring.
        """
        if len(self.keys_to_skip_loading) == 0:
            return state_dict

        new_state_dict = {}
        for fqn, obj in state_dict.items():
            if any(skip_key in fqn for skip_key in self.keys_to_skip_loading):
                log.warning(f"Skipping loading of key: {fqn}")
                continue
            new_state_dict[fqn] = obj
        return new_state_dict


class CustomSavePlanner(dcp.DefaultSavePlanner):
    """
    Custom save planner that enables an override for cache_plans_key when
    caching of save plans is enabled. Caching of save plans reduces checkpointing
    time by reusing the same save plan across checkpoints. This reduces the
    checkpointing time by ~60% (benchmarked using the 235B-A22B Qwen3-VL model
    on 64 GB200 nodes).
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_save_to_lowest_rank: bool = False,
        save_reg_to_ema: bool = False,
        enable_plan_caching: bool = False,
        cache_plans_key: str | None = None,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
            enable_plan_caching=enable_plan_caching,
        )
        if cache_plans_key is not None:
            self._cached_plans_key = cache_plans_key

        self.save_reg_to_ema = save_reg_to_ema

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.save_reg_to_ema:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            storage_meta=storage_meta,
            is_coordinator=is_coordinator,
        )


class DistributedCheckpointer(AbstractCheckpointer):
    CHECKPOINT_KEYS = ["model", "optim", "scheduler", "trainer", "dataloader"]
    _LOCAL_RESERVE_DIRNAME = ".checkpoint_reserve"
    _LOCAL_RESERVE_HEADROOM = 1.05
    _LOCAL_RESERVE_MIN_FREE_GB = 10.0

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        self._prepared_checkpoint_files: set[str] = set()
        if config_checkpoint.dcp_async_mode_enabled and not disable_async:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self._rename_oldest_before_save_enabled():
            raise ValueError(
                "checkpoint.rename_oldest_before_save=True requires "
                "checkpoint.dcp_async_mode_enabled=False"
            )

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()
            self.checkpoint_in_progress = False

    def _local_checkpoint_directories(self, *, exclude: str | None = None) -> list[tuple[int, str]]:
        if self.save_to_object_store or not os.path.isdir(self.save_dirname):
            return []
        checkpoints: list[tuple[int, str]] = []
        for entry in os.scandir(self.save_dirname):
            if not entry.is_dir(follow_symlinks=False) or entry.name == exclude:
                continue
            match = re.fullmatch(r"iter_(\d{9})", entry.name)
            if match is not None:
                checkpoints.append((int(match.group(1)), entry.path))
        checkpoints.sort()
        return checkpoints

    @staticmethod
    def _local_directory_size_bytes(path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                total += os.stat(os.path.join(root, filename), follow_symlinks=False).st_size
        return total

    @property
    def _local_reserve_dir(self) -> str:
        return os.path.join(self.save_dirname, self._LOCAL_RESERVE_DIRNAME)

    def _local_reserve_files(self) -> list[str]:
        if not os.path.isdir(self._local_reserve_dir):
            return []
        return sorted(
            entry.path
            for entry in os.scandir(self._local_reserve_dir)
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(".reserve")
        )

    @staticmethod
    def _empty_local_directory(path: str) -> None:
        """Remove a directory's contents without replacing the directory itself."""
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)

    @staticmethod
    def _allocate_local_reserve_file(path: str, size_bytes: int) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(path, flags, 0o600)
        try:
            if not hasattr(os, "posix_fallocate"):
                raise OSError("os.posix_fallocate is unavailable on this platform")
            os.posix_fallocate(fd, 0, size_bytes)
        except Exception:
            os.close(fd)
            os.unlink(path)
            raise
        else:
            os.close(fd)

    def _preallocation_enabled(self) -> bool:
        return bool(getattr(self.config_checkpoint, "preallocate_local_slots", False))

    def _rename_oldest_before_save_enabled(self) -> bool:
        return bool(getattr(self.config_checkpoint, "rename_oldest_before_save", False))

    def _run_rank0_local_storage_action(self, action) -> None:  # noqa: ANN001
        local_exception: Exception | None = None
        if distributed.is_rank0():
            try:
                action()
            except Exception as exc:
                local_exception = exc
        any_failed = self._synchronize_checkpoint_save_failure(local_exception is not None)
        if any_failed:
            if local_exception is not None:
                raise local_exception
            raise RuntimeError("Local checkpoint storage preparation failed on rank 0")

    def prepare_local_checkpoint_slots(self) -> None:
        """Preallocate real local disk capacity for the remaining checkpoint slots."""
        if not self._preallocation_enabled():
            return
        if self.save_to_object_store:
            if distributed.is_rank0():
                log.warning("checkpoint.preallocate_local_slots only applies to local checkpoint saves.")
            return

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._wait_for_previous_async_checkpoint()

        max_to_keep = getattr(self.config_checkpoint, "max_to_keep", None)
        if max_to_keep is None or int(max_to_keep) <= 0:
            raise ValueError("checkpoint.preallocate_local_slots=True requires checkpoint.max_to_keep > 0")

        def prepare() -> None:
            os.makedirs(self.save_dirname, exist_ok=True)
            checkpoints = self._local_checkpoint_directories()
            if not checkpoints:
                raise RuntimeError(
                    "Checkpoint preallocation needs one real checkpoint to measure. "
                    "Enable trainer.save_zero_checkpoint or resume a run with an existing checkpoint."
                )

            while len(checkpoints) > int(max_to_keep):
                _, stale_path = checkpoints.pop(0)
                log.info(f"Removing stale checkpoint before slot preallocation: {stale_path}")
                shutil.rmtree(stale_path)

            measured_size = self._local_directory_size_bytes(checkpoints[-1][1])
            if measured_size <= 0:
                raise RuntimeError(f"Cannot preallocate from empty checkpoint directory: {checkpoints[-1][1]}")
            slot_size = max(1, int(measured_size * self._LOCAL_RESERVE_HEADROOM))
            min_free_gb = self._LOCAL_RESERVE_MIN_FREE_GB
            min_free_bytes = int(min_free_gb * 1024**3)

            if os.path.isdir(self._local_reserve_dir):
                shutil.rmtree(self._local_reserve_dir)
            os.makedirs(self._local_reserve_dir, exist_ok=True)
            reserve_count = max(0, int(max_to_keep) - len(checkpoints))
            required_bytes = reserve_count * slot_size
            free_bytes = shutil.disk_usage(self.save_dirname).free
            if free_bytes - required_bytes < min_free_bytes:
                raise OSError(
                    "Insufficient disk space for checkpoint preallocation: "
                    f"need {required_bytes / 1024**3:.1f} GiB for {reserve_count} slots plus "
                    f"{min_free_gb:.1f} GiB free, but only {free_bytes / 1024**3:.1f} GiB is available."
                )

            try:
                for slot_index in range(reserve_count):
                    reserve_path = os.path.join(self._local_reserve_dir, f"slot_{slot_index:04d}.reserve")
                    self._allocate_local_reserve_file(reserve_path, slot_size)
            except Exception:
                shutil.rmtree(self._local_reserve_dir, ignore_errors=True)
                raise
            log.info(
                "Checkpoint disk preallocation complete: "
                f"real_checkpoints={len(checkpoints)}, reserve_slots={reserve_count}, "
                f"slot_size={slot_size / 1024**3:.1f} GiB, min_free={min_free_gb:.1f} GiB"
            )

        self._run_rank0_local_storage_action(prepare)

    def _prepare_local_space_for_checkpoint(self, checkpoint_file: str) -> None:
        """Prepare bounded local storage before any bytes of a checkpoint are written."""
        preallocate = self._preallocation_enabled()
        rename_oldest = self._rename_oldest_before_save_enabled()
        if not preallocate and not rename_oldest:
            return
        if self.save_to_object_store:
            if distributed.is_rank0() and rename_oldest:
                log.warning("checkpoint.rename_oldest_before_save only applies to local checkpoint saves.")
            return
        if checkpoint_file in self._prepared_checkpoint_files:
            return

        max_to_keep = getattr(self.config_checkpoint, "max_to_keep", None)
        if max_to_keep is None or int(max_to_keep) <= 0:
            enabled_options = []
            if preallocate:
                enabled_options.append("checkpoint.preallocate_local_slots")
            if rename_oldest:
                enabled_options.append("checkpoint.rename_oldest_before_save")
            raise ValueError(f"{' and '.join(enabled_options)} require checkpoint.max_to_keep > 0")

        def prepare() -> None:
            target_path = os.path.join(self.save_dirname, checkpoint_file)

            # A retry, or a restart after an interrupted save, may already have
            # the target directory. Reuse that slot instead of consuming a
            # complete checkpoint, but discard partial/stale component files.
            if os.path.isdir(target_path):
                log.info(f"Reusing existing checkpoint target after clearing stale contents: {target_path}")
                self._empty_local_directory(target_path)
                return

            reserve_files = self._local_reserve_files() if preallocate else []
            if reserve_files and len(self._local_checkpoint_directories()) < int(max_to_keep):
                reserve_path = reserve_files[-1]
                os.unlink(reserve_path)
                log.info(f"Released preallocated checkpoint slot before saving {checkpoint_file}: {reserve_path}")
                return

            checkpoints = self._local_checkpoint_directories(exclude=checkpoint_file)
            if len(checkpoints) >= int(max_to_keep):
                latest_checkpoint_file = self._read_latest_checkpoint_file()
                latest_match = (
                    re.fullmatch(r"iter_(\d{9})", latest_checkpoint_file)
                    if latest_checkpoint_file is not None
                    else None
                )
                latest_iteration = int(latest_match.group(1)) if latest_match is not None else None

                # A directory newer than latest_checkpoint.txt is residue from
                # an interrupted save. Reuse it before sacrificing another
                # complete checkpoint. Otherwise recycle the oldest directory
                # that is not the last known-good checkpoint.
                incomplete = [item for item in checkpoints if latest_iteration is not None and item[0] > latest_iteration]
                reusable = incomplete or [
                    item for item in checkpoints if os.path.basename(item[1]) != latest_checkpoint_file
                ]
                _, stale_path = (reusable or checkpoints)[0]
                if rename_oldest:
                    log.info(
                        f"Renaming oldest checkpoint before saving {checkpoint_file} "
                        f"to avoid max_to_keep+1 disk usage: {stale_path} -> {target_path}"
                    )
                    os.rename(stale_path, target_path)
                    self._empty_local_directory(target_path)
                else:
                    log.info(
                        f"Removing oldest checkpoint before saving {checkpoint_file} "
                        f"to avoid max_to_keep+1 disk usage: {stale_path}"
                    )
                    shutil.rmtree(stale_path)

        self._run_rank0_local_storage_action(prepare)
        self._prepared_checkpoint_files.add(checkpoint_file)

    def keys_to_resume_during_load(self) -> tuple[set[str], str | None, bool | None]:
        """
        Determines the keys to resume from the checkpoint and the checkpoint path.
        If the checkpoint is the latest checkpoint of the same model, then it is a
        normal resume. If the checkpoint is a different model's checkpoint, then it is
        a warm start.

        Args:
            None

        Returns:
            resume_keys: The keys to resume from the checkpoint.
            checkpoint_path: The path to the checkpoint. If the checkpoint is a different
            warm_start: Whether to warm start the training from a different model's checkpoint.
                If the checkpoint is a different model's checkpoint, then this is True.
                If the checkpoint is the latest checkpoint of the same model, then this is False.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []
        warm_start = None

        if latest_checkpoint_file is not None:
            # 1. Resume training from the latest checkpoint of the same model.
            warm_start = False
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.CHECKPOINT_KEYS)

        else:
            if self.load_path and not str(self.load_path).endswith(".pt"):
                # 2. Warm Start: Resume training from a different model's checkpoint
                # specified by `load_path`.
                warm_start = True
                checkpoint_path = self.load_path

                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"

                    # If the path doesn't end with specific checkpoint, read the latest
                    # checkpoint file to determine the most recent checkpoint iteration.
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        old_ckpt_path = checkpoint_path
                        latest_ckpt_path = os.path.join(checkpoint_path, "checkpoints/latest_checkpoint.txt")

                        # If the latest checkpoint file exists, use it to determine the
                        # checkpoint path. Otherwise, use the original path.
                        if easy_io.exists(latest_ckpt_path, backend_key=self.load_s3_backend_key):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                        else:
                            log.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, load from {old_ckpt_path}"
                            )
                            checkpoint_path = old_ckpt_path

                if self.load_training_state:
                    resume_keys.extend(self.CHECKPOINT_KEYS)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None

        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.CHECKPOINT_KEYS, f"Invalid key to resume: {key} not in {self.CHECKPOINT_KEYS}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]

        return set(resume_keys), checkpoint_path, warm_start

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)

            for key in resume_keys:
                dist.barrier()

                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {cur_key_ckpt_full_path}")

                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                strict_resume = self.config_checkpoint.strict_resume

                # Note that we only allow skipping loading of keys during warm start. If the checkpoint is
                # the latest checkpoint of the same model, then we don't need to skip any keys.
                keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []

                load_planner = CustomLoadPlanner(
                    allow_partial_load=not strict_resume,
                    keys_to_skip_loading=keys_to_skip_loading,
                )

                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    if self.config_checkpoint.load_ema_to_reg:
                        # The model has both net.* and net_ema.* submodules, so _state_dict
                        # contains both sets of keys after dcp.load(). Copy EMA weights into
                        # regular model weights so we can resume from EMA and reset EMA.
                        for sd_key in list(_state_dict.keys()):
                            if sd_key.startswith("net."):
                                key_ema = "net_ema." + sd_key.removeprefix("net.")
                                assert key_ema in _state_dict, (
                                    f"EMA key {key_ema} not found in state_dict. "
                                    "Ensure the model has net_ema submodule."
                                )
                                _state_dict[sd_key] = _state_dict[key_ema]
                    results = _model_wrapper.load_state_dict(_state_dict)
                    if len(results.missing_keys) > 0:
                        raise ValueError(f"Missing keys (not found in checkpoint): {results.missing_keys}")
                    if len(results.unexpected_keys) > 0:
                        raise ValueError(
                            f"Unexpected keys (found in checkpoint but not in model): {results.unexpected_keys}"
                        )

                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _state_dict = optimizer.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    optimizer.load_state_dict(_state_dict)

                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)

                elif key == "trainer":
                    log.info("- Loading the trainer...")

                    # Use rank-specific key for RNG state to support correct per-rank restoration
                    rng_key = f"rng_state_{dist.get_rank()}"
                    current_rng_state = get_rand_state_dict()
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    # Check if rng_key exists in checkpoint metadata to avoid failure with strict_resume=True
                    metadata = storage_reader.read_metadata()
                    rng_key_exists = any(
                        k.startswith(f"{rng_key}.") or k == rng_key for k in metadata.state_dict_metadata.keys()
                    )
                    if rng_key_exists:
                        _state_dict[rng_key] = current_rng_state

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                    set_rand_state_dict(_state_dict.get(rng_key, current_rng_state))

                elif key == "dataloader":
                    if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading dataloader.",
                            rank0_only=False,
                        )
                        continue

                    rank = dist.get_rank()
                    dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                    if not easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                        log.info(f"No dataloader checkpoint found at {dataloader_pkl_path}", rank0_only=False)
                        continue

                    log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                    _state_dict = easy_io.load(
                        dataloader_pkl_path,
                        file_format="pkl",
                        backend_key=self.load_s3_backend_key,
                    )
                    dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                    if dataloader_wrapper.has_state():
                        dataloader_wrapper.load_state_dict(_state_dict)

                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            if self.callbacks is not None and resume_keys:
                # Note that this callback is never used in the codebase.
                self.callbacks.on_load_checkpoint(model, state_dict={})
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")

        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _checkpoint_async_with_pinned_memory(
        self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]
    ) -> None:
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

        if self.cpu_offload_state_dict is None:
            log.info(f"Preparing the CPU memory for staging")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.info(f"Staging the state_dict in CPU memory")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging_ckpt_file = checkpoint_file

        self.staging_stream.synchronize()
        log.info(f"Staging the state_dict in CPU memory completed")

        self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))
        self.checkpoint_in_progress = True
        log.info(f"Submitted checkpoint to background process")

    def _wait_for_previous_async_checkpoint(self) -> None:
        """
        Gets the results of previously submitted checkpoints.
        Pass them to callbacks if checkpoint succeeded.
        """
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        if not self.checkpoint_in_progress:
            return

        success = False
        try:
            log.info(f"Waiting for checkpoint save result")

            # Note that we set a timeout of 1 hour to avoid blocking the main process
            # indefinitely. Gloo and NCCL timeouts are ~30 minutes, so this timeout
            # should typically be sufficient.
            save_done: SaveDone = self.mp_queue_recv.get(timeout=3600)

            log.info(f"Received checkpoint save result: {save_done}")

            if self.callbacks is not None and save_done.succeeded:
                self.callbacks.on_save_checkpoint_success(
                    iteration=save_done.iteration, elapsed_time=save_done.elapsed_time
                )
            self.checkpoint_in_progress = False
            success = save_done.succeeded

        except Exception as e:
            log.error(f"Error waiting for checkpoint save result: {e}")

        if not success:
            # Terminate training execution upon a failed checkpoint save attempt.
            # A failure at this stage typically indicates a non-recoverable system error.
            # Continuing execution would result in subsequent persistent failures and
            # unnecessary waste of GPU resources.
            raise RuntimeError("Previous checkpoint save failed. Exiting...")

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemReader(checkpoint_path)

    def _save_as_pkl(self, obj: Any, output_dir: str) -> None:
        """Save per-rank Python checkpoint state such as no-replace dataloader progress."""
        rank = dist.get_rank()
        path = os.path.join(output_dir, f"rank_{rank}.pkl")
        easy_io.dump(
            obj,
            path,
            file_format="pkl",
            backend_key=self.save_s3_backend_key,
        )
        log.info(f"Saved state to {path}")

    def _prune_old_checkpoints(self) -> None:
        """Keep only the latest configured local checkpoint directories."""
        max_to_keep = getattr(self.config_checkpoint, "max_to_keep", None)
        if max_to_keep is None or max_to_keep <= 0:
            return
        if self.save_to_object_store:
            log.warning("checkpoint.max_to_keep is currently only applied to local checkpoint saves.")
            return
        if not distributed.is_rank0():
            return

        try:
            entries = easy_io.list_dir_or_file(
                self.save_dirname,
                list_dir=True,
                list_file=False,
                recursive=False,
                backend_key=self.save_s3_backend_key,
            )
            checkpoints: list[tuple[int, str]] = []
            for entry in entries:
                checkpoint_file = entry.rstrip("/")
                match = re.fullmatch(r"iter_(\d{9})", checkpoint_file)
                if match is None:
                    continue
                checkpoints.append((int(match.group(1)), checkpoint_file))

            checkpoints.sort()
            stale_checkpoints = checkpoints[: max(0, len(checkpoints) - max_to_keep)]
            for _, checkpoint_file in stale_checkpoints:
                checkpoint_path = os.path.join(self.save_dirname, checkpoint_file)
                log.info(f"Removing old checkpoint due to max_to_keep={max_to_keep}: {checkpoint_path}")
                easy_io.rmtree(checkpoint_path, backend_key=self.save_s3_backend_key)
        except Exception as e:
            log.warning(f"Failed to prune old checkpoints: {e}")

    def _get_save_retry_settings(self) -> tuple[int, float, float]:
        max_attempts = max(1, int(getattr(self.config_checkpoint, "save_max_attempts", 3)))
        initial_backoff = float(getattr(self.config_checkpoint, "save_retry_initial_backoff", 5.0))
        max_backoff = float(getattr(self.config_checkpoint, "save_retry_max_backoff", 60.0))
        return max_attempts, initial_backoff, max_backoff

    def _synchronize_checkpoint_save_failure(self, local_failed: bool) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return local_failed

        backend = str(dist.get_backend()).lower()
        device = torch.device("cpu")
        if "nccl" in backend:
            device = torch.device("cuda", torch.cuda.current_device())

        failure = torch.tensor(int(local_failed), device=device)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        return bool(failure.item())

    def _save_state_dict_worker_once(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        self._prepare_local_space_for_checkpoint(checkpoint_file)
        for key, (v, full_checkpoint_path) in to_save_dict.items():
            if key == "dataloader":
                self._save_as_pkl(v, full_checkpoint_path)
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                # Note that it is ok to create a new CustomSavePlanner object
                # for each checkpoint save since the save plans are cached in a
                # class dictionary.
                save_planner = CustomSavePlanner(
                    dedup_save_to_lowest_rank=True,
                    enable_plan_caching=True,
                    cache_plans_key=f"custom_planner_{key}",
                )
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=save_planner,
                )

        if distributed.is_rank0():
            log.info(f"Saving last checkpoint file {checkpoint_file}")
            self._write_latest_checkpoint_file(checkpoint_file)

        log.info(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}")
        self._prune_old_checkpoints()

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        max_attempts, initial_backoff, max_backoff = self._get_save_retry_settings()
        backoff = initial_backoff
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            local_exception: Exception | None = None
            try:
                self._save_state_dict_worker_once(to_save_dict, checkpoint_file)
            except Exception as exc:
                local_exception = exc
                last_exception = exc

            any_failed = self._synchronize_checkpoint_save_failure(local_exception is not None)
            if not any_failed:
                return

            if attempt == max_attempts:
                if local_exception is not None:
                    raise local_exception
                if last_exception is not None:
                    raise last_exception
                raise RuntimeError("Checkpoint save failed on another rank")

            # Let the next attempt clear the partially written target directory
            # before writing any component again.
            self._prepared_checkpoint_files.discard(checkpoint_file)

            if local_exception is not None:
                log.warning(
                    f"Checkpoint save attempt {attempt}/{max_attempts} failed: {type(local_exception).__name__}: {local_exception}",
                    rank0_only=False,
                )
            else:
                log.warning(
                    f"Checkpoint save attempt {attempt}/{max_attempts} failed on another rank.",
                    rank0_only=False,
                )

            current_backoff = min(backoff, max_backoff)
            log.warning(
                f"Retrying checkpoint save in {current_backoff:.1f} seconds",
                rank0_only=False,
            )
            time.sleep(current_backoff)
            backoff = min(backoff * 2.0, max_backoff)

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._wait_for_previous_async_checkpoint()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"

        # In async mode the dataloader state is written by the training process
        # before the background DCP worker starts. Prepare the reusable local
        # directory here so no checkpoint component creates the target first.
        self._prepare_local_space_for_checkpoint(checkpoint_file)

        # Use rank-specific key for RNG state to ensure each rank saves its own state
        rng_key = f"rng_state_{dist.get_rank()}"

        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
                rng_key: get_rand_state_dict(),
            },
        }
        dataloader_wrapper = _DataloaderWrapper(self.callbacks)
        if dataloader_wrapper.has_state():
            to_save_dict["dataloader"] = dataloader_wrapper.state_dict()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            dataloader_entry = to_save_dict.pop("dataloader", None)
            if dataloader_entry is not None:
                dataloader_state, dataloader_save_dir = dataloader_entry
                self._save_as_pkl(dataloader_state, dataloader_save_dir)
            self._checkpoint_async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            self.save_state_dict_worker(to_save_dict, checkpoint_file)
            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint save completed: Time taken: {elapsed_time:.2f} seconds")

            if self.callbacks is not None:
                self.callbacks.on_save_checkpoint_success(iteration=iteration, elapsed_time=elapsed_time)

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                # Wait for the previous checkpoint to complete.
                self._wait_for_previous_async_checkpoint()

                self.mp_queue_send.put(Terminate())
                self.mp.join()
