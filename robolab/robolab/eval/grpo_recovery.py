# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in recovery contract for GRPO rollouts at RoboLab chunk boundaries.

This module deliberately has no Isaac imports and is not wired into the normal
evaluation launcher.  It defines the fail-closed part of initial-state reset +
exact executed-action prefix replay.  A simulator adapter is responsible for
implementing :class:`RecoveryBackend` when the protocol is enabled explicitly.

``chunk_count`` is exclusive: a prefix with ``chunk_count=2`` replays
``chunk_0000`` and ``chunk_0001``.  This avoids the inclusive/exclusive
ambiguity of ``replay_through_chunk`` in runtime-facing interfaces.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class RecoveryProtocolError(RuntimeError):
    """Raised when recovery cannot be performed or verified safely."""


class ActionArtifactError(RecoveryProtocolError):
    """Raised when an executed-action artifact is missing, invalid, or changed."""


class BoundaryDigestMismatch(RecoveryProtocolError):
    """Raised when replay does not reach the expected chunk-boundary state."""

    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            "Recovered boundary digest does not match the canonical boundary "
            f"(expected={expected!r}, actual={actual!r})"
        )
        self.expected = expected
        self.actual = actual


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ActionArtifactError(f"Could not hash executed actions: {path}") from exc
    return digest.hexdigest()


def _load_action_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ActionArtifactError(f"Missing executed-action artifact: {path}")
    try:
        actions = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ActionArtifactError(f"Could not load executed actions: {path}") from exc
    if actions.ndim != 2:
        raise ActionArtifactError(
            f"Executed actions must be a 2-D matrix, got shape {actions.shape} at {path}"
        )
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ActionArtifactError(
            f"Executed actions must have non-empty step and action dimensions at {path}"
        )
    if not np.issubdtype(actions.dtype, np.number):
        raise ActionArtifactError(f"Executed actions must be numeric at {path}")
    if not np.all(np.isfinite(actions)):
        raise ActionArtifactError(f"Executed actions contain NaN/Inf at {path}")
    return np.asarray(actions)


@dataclass(frozen=True)
class ExecutedActionChunk:
    """Immutable manifest entry for one canonical executed-action chunk."""

    chunk_index: int
    path: Path
    sha256: str
    shape: tuple[int, int]
    dtype: str

    @classmethod
    def from_path(cls, *, chunk_index: int, path: str | Path) -> ExecutedActionChunk:
        if chunk_index < 0:
            raise ActionArtifactError(f"chunk_index must be >= 0, got {chunk_index}")
        resolved_path = Path(path).expanduser().resolve()
        actions = _load_action_matrix(resolved_path)
        return cls(
            chunk_index=chunk_index,
            path=resolved_path,
            sha256=_sha256_file(resolved_path),
            shape=(int(actions.shape[0]), int(actions.shape[1])),
            dtype=actions.dtype.str,
        )

    def validate_and_load(self) -> np.ndarray:
        """Revalidate the artifact and return a detached, replay-ready copy."""

        actual_sha256 = _sha256_file(self.path)
        if not hmac.compare_digest(actual_sha256, self.sha256):
            raise ActionArtifactError(
                f"Executed-action artifact changed after manifest creation: {self.path}"
            )
        actions = _load_action_matrix(self.path)
        actual_shape = (int(actions.shape[0]), int(actions.shape[1]))
        if actual_shape != self.shape or actions.dtype.str != self.dtype:
            raise ActionArtifactError(
                "Executed-action metadata changed after manifest creation: "
                f"{self.path} (expected shape={self.shape}, dtype={self.dtype}; "
                f"got shape={actual_shape}, dtype={actions.dtype.str})"
            )
        # Recovery uses this in-memory copy, so no artifact reads occur after
        # backend side effects have begun.
        return actions.copy()


@dataclass(frozen=True)
class CanonicalActionPrefix:
    """Validated manifest of chunks required to reach one branch boundary."""

    task_name: str
    episode_seed: int
    chunks: tuple[ExecutedActionChunk, ...]

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ActionArtifactError("task_name must be non-empty")
        indices = tuple(chunk.chunk_index for chunk in self.chunks)
        expected_indices = tuple(range(len(self.chunks)))
        if indices != expected_indices:
            raise ActionArtifactError(
                "Canonical chunks must be contiguous and start at zero: "
                f"expected {expected_indices}, got {indices}"
            )
        action_dims = {chunk.shape[1] for chunk in self.chunks}
        if len(action_dims) > 1:
            raise ActionArtifactError(
                f"Canonical chunks have inconsistent action dimensions: {sorted(action_dims)}"
            )

    @property
    def chunk_count(self) -> int:
        """Exclusive prefix length (chunks ``0 .. chunk_count - 1``)."""

        return len(self.chunks)

    @classmethod
    def from_replay_directory(
        cls,
        replay_source_dir: str | Path,
        *,
        chunk_count: int,
        task_name: str,
        episode_seed: int,
        env_id: int = 0,
    ) -> CanonicalActionPrefix:
        """Build a manifest from InteractiveCosmos3Client replay artifacts.

        ``replay_source_dir`` may be either the run root containing ``envN`` or
        the ``envN`` directory itself.  Every required chunk is checked before
        a recovery backend can be invoked.
        """

        if chunk_count < 0:
            raise ActionArtifactError(f"chunk_count must be >= 0, got {chunk_count}")
        if env_id < 0:
            raise ActionArtifactError(f"env_id must be >= 0, got {env_id}")
        source = Path(replay_source_dir).expanduser().resolve()
        env_dir = source if source.name == f"env{env_id}" else source / f"env{env_id}"
        chunks = tuple(
            ExecutedActionChunk.from_path(
                chunk_index=index,
                path=env_dir / f"chunk_{index:04d}" / "executed_actions.npy",
            )
            for index in range(chunk_count)
        )
        return cls(
            task_name=task_name,
            episode_seed=int(episode_seed),
            chunks=chunks,
        )


@runtime_checkable
class RecoveryBackend(Protocol):
    """Simulator-facing adapter required by the opt-in recovery coordinator."""

    def reset_policy_state(self) -> None:
        """Clear client/session state (for example, ``client.reset()``)."""

    def reset_eval_state(self) -> None:
        """Clear RoboLab termination and predicate bookkeeping."""

    def reset_environment(self) -> None:
        """Reset the already-running simulator to its canonical initial state."""

    def replay_executed_actions(self, actions: np.ndarray, *, chunk_index: int) -> None:
        """Step the exact supplied action values without model post-processing."""

    def capture_boundary_digest(self) -> str:
        """Return a deterministic digest for the current branch boundary."""


@dataclass(frozen=True)
class RecoveryResult:
    """Verified result of one reset + prefix replay operation."""

    task_name: str
    episode_seed: int
    replayed_chunks: int
    replayed_steps: int
    boundary_digest: str
    digest_verified: bool


class InitialReplayRecoveryCoordinator:
    """Execute reset + exact prefix replay with an optional digest barrier.

    This class is inert until instantiated and called by an external adapter;
    importing it does not initialize or mutate a simulator.
    """

    def __init__(
        self,
        backend: RecoveryBackend,
        *,
        digest_equal: Callable[[str, str], bool] = hmac.compare_digest,
    ) -> None:
        self._backend = backend
        self._digest_equal = digest_equal

    def recover(
        self,
        prefix: CanonicalActionPrefix,
        *,
        expected_boundary_digest: str | None,
    ) -> RecoveryResult:
        """Reset and replay ``prefix``, failing closed on invalid data or drift.

        All artifacts are loaded and revalidated before the first backend
        call.  Passing ``None`` captures a baseline digest without claiming it
        was verified; GRPO branch rollouts should pass the canonical digest.
        """

        loaded_chunks = self._preflight(prefix)

        self._backend.reset_policy_state()
        self._backend.reset_eval_state()
        self._backend.reset_environment()
        for chunk, actions in zip(prefix.chunks, loaded_chunks, strict=True):
            try:
                self._backend.replay_executed_actions(
                    actions,
                    chunk_index=chunk.chunk_index,
                )
            except Exception as exc:
                raise RecoveryProtocolError(
                    f"Backend failed while replaying chunk {chunk.chunk_index}"
                ) from exc

        actual_digest = self._backend.capture_boundary_digest()
        if not isinstance(actual_digest, str) or not actual_digest:
            raise RecoveryProtocolError("Backend returned an empty or non-string boundary digest")
        digest_verified = expected_boundary_digest is not None
        if expected_boundary_digest is not None and not self._digest_equal(
            expected_boundary_digest, actual_digest
        ):
            raise BoundaryDigestMismatch(
                expected=expected_boundary_digest,
                actual=actual_digest,
            )
        return RecoveryResult(
            task_name=prefix.task_name,
            episode_seed=prefix.episode_seed,
            replayed_chunks=prefix.chunk_count,
            replayed_steps=sum(actions.shape[0] for actions in loaded_chunks),
            boundary_digest=actual_digest,
            digest_verified=digest_verified,
        )

    @staticmethod
    def _preflight(prefix: CanonicalActionPrefix) -> tuple[np.ndarray, ...]:
        # Recheck continuity even if a caller bypassed the normal constructor
        # (or a future deserializer is added).
        expected_indices: Sequence[int] = range(prefix.chunk_count)
        actual_indices = tuple(chunk.chunk_index for chunk in prefix.chunks)
        if tuple(expected_indices) != actual_indices:
            raise ActionArtifactError(
                "Canonical chunks are no longer contiguous: "
                f"expected {tuple(expected_indices)}, got {actual_indices}"
            )
        loaded = tuple(chunk.validate_and_load() for chunk in prefix.chunks)
        action_dims = {actions.shape[1] for actions in loaded}
        if len(action_dims) > 1:
            raise ActionArtifactError(
                f"Canonical chunks have inconsistent action dimensions: {sorted(action_dims)}"
            )
        return loaded
