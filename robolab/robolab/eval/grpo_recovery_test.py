from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robolab.eval.grpo_recovery import (
    ActionArtifactError,
    BoundaryDigestMismatch,
    CanonicalActionPrefix,
    InitialReplayRecoveryCoordinator,
    RecoveryProtocolError,
)


class FakeBackend:
    def __init__(self, *, digest: str = "boundary-ok", fail_chunk: int | None = None) -> None:
        self.calls: list[object] = []
        self.replayed: list[np.ndarray] = []
        self.digest = digest
        self.fail_chunk = fail_chunk

    def reset_policy_state(self) -> None:
        self.calls.append("reset_policy_state")

    def reset_eval_state(self) -> None:
        self.calls.append("reset_eval_state")

    def reset_environment(self) -> None:
        self.calls.append("reset_environment")

    def replay_executed_actions(self, actions: np.ndarray, *, chunk_index: int) -> None:
        self.calls.append(("replay", chunk_index))
        if self.fail_chunk == chunk_index:
            raise RuntimeError("simulated step failure")
        self.replayed.append(actions.copy())

    def capture_boundary_digest(self) -> str:
        self.calls.append("capture_boundary_digest")
        return self.digest


def _write_chunk(root: Path, index: int, actions: np.ndarray) -> Path:
    directory = root / "env0" / f"chunk_{index:04d}"
    directory.mkdir(parents=True)
    path = directory / "executed_actions.npy"
    np.save(path, actions)
    return path


def _prefix(root: Path, *, chunk_count: int) -> CanonicalActionPrefix:
    return CanonicalActionPrefix.from_replay_directory(
        root,
        chunk_count=chunk_count,
        task_name="BagelsOnPlateTask",
        episode_seed=17,
    )


def test_reset_order_and_exact_prefix_replay(tmp_path: Path) -> None:
    first = np.arange(24, dtype=np.float32).reshape(3, 8)
    second = (np.arange(16, dtype=np.float32) + 100).reshape(2, 8)
    _write_chunk(tmp_path, 0, first)
    _write_chunk(tmp_path, 1, second)
    backend = FakeBackend()

    result = InitialReplayRecoveryCoordinator(backend).recover(
        _prefix(tmp_path, chunk_count=2),
        expected_boundary_digest="boundary-ok",
    )

    assert backend.calls == [
        "reset_policy_state",
        "reset_eval_state",
        "reset_environment",
        ("replay", 0),
        ("replay", 1),
        "capture_boundary_digest",
    ]
    np.testing.assert_array_equal(backend.replayed[0], first)
    np.testing.assert_array_equal(backend.replayed[1], second)
    assert result.replayed_chunks == 2
    assert result.replayed_steps == 5
    assert result.digest_verified is True


def test_isolated_gripper_pulse_survives_replay(tmp_path: Path) -> None:
    actions = np.zeros((32, 8), dtype=np.float32)
    actions[7, -1] = 1.0
    _write_chunk(tmp_path, 0, actions)
    backend = FakeBackend()

    InitialReplayRecoveryCoordinator(backend).recover(
        _prefix(tmp_path, chunk_count=1),
        expected_boundary_digest="boundary-ok",
    )

    np.testing.assert_array_equal(backend.replayed[0], actions)
    assert backend.replayed[0][7, -1] == 1.0


def test_missing_chunk_fails_before_backend_side_effects(tmp_path: Path) -> None:
    _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    backend = FakeBackend()

    with pytest.raises(ActionArtifactError, match="Missing executed-action artifact"):
        # Manifest construction itself is part of preflight.
        prefix = _prefix(tmp_path, chunk_count=2)
        InitialReplayRecoveryCoordinator(backend).recover(
            prefix,
            expected_boundary_digest="boundary-ok",
        )

    assert backend.calls == []


def test_artifact_tampering_after_manifest_fails_before_reset(tmp_path: Path) -> None:
    path = _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    prefix = _prefix(tmp_path, chunk_count=1)
    np.save(path, np.ones((2, 8), dtype=np.float32))
    backend = FakeBackend()

    with pytest.raises(ActionArtifactError, match="changed after manifest creation"):
        InitialReplayRecoveryCoordinator(backend).recover(
            prefix,
            expected_boundary_digest="boundary-ok",
        )

    assert backend.calls == []


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((2, 2, 2), dtype=np.float32), "2-D matrix"),
        (np.zeros((2, 0), dtype=np.float32), "non-empty"),
        (np.zeros((0, 8), dtype=np.float32), "non-empty"),
        (np.full((2, 8), np.nan, dtype=np.float32), "NaN/Inf"),
        (np.array([["bad"]], dtype="U3"), "numeric"),
    ],
)
def test_invalid_action_artifacts_are_rejected(
    tmp_path: Path, actions: np.ndarray, message: str
) -> None:
    _write_chunk(tmp_path, 0, actions)

    with pytest.raises(ActionArtifactError, match=message):
        _prefix(tmp_path, chunk_count=1)


def test_inconsistent_action_dimensions_are_rejected(tmp_path: Path) -> None:
    _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    _write_chunk(tmp_path, 1, np.zeros((2, 7), dtype=np.float32))

    with pytest.raises(ActionArtifactError, match="inconsistent action dimensions"):
        _prefix(tmp_path, chunk_count=2)


def test_non_contiguous_manifest_is_rejected(tmp_path: Path) -> None:
    _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    prefix = _prefix(tmp_path, chunk_count=1)

    with pytest.raises(ActionArtifactError, match="contiguous"):
        replace(prefix, chunks=(replace(prefix.chunks[0], chunk_index=1),))


def test_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    backend = FakeBackend(digest="drifted")

    with pytest.raises(BoundaryDigestMismatch) as error:
        InitialReplayRecoveryCoordinator(backend).recover(
            _prefix(tmp_path, chunk_count=1),
            expected_boundary_digest="canonical",
        )

    assert error.value.expected == "canonical"
    assert error.value.actual == "drifted"


def test_empty_prefix_resets_and_captures_unverified_initial_digest(tmp_path: Path) -> None:
    backend = FakeBackend(digest="initial")

    result = InitialReplayRecoveryCoordinator(backend).recover(
        _prefix(tmp_path, chunk_count=0),
        expected_boundary_digest=None,
    )

    assert backend.calls == [
        "reset_policy_state",
        "reset_eval_state",
        "reset_environment",
        "capture_boundary_digest",
    ]
    assert result.replayed_chunks == 0
    assert result.replayed_steps == 0
    assert result.boundary_digest == "initial"
    assert result.digest_verified is False


def test_backend_replay_failure_does_not_return_success(tmp_path: Path) -> None:
    _write_chunk(tmp_path, 0, np.zeros((2, 8), dtype=np.float32))
    backend = FakeBackend(fail_chunk=0)

    with pytest.raises(RecoveryProtocolError, match="replaying chunk 0"):
        InitialReplayRecoveryCoordinator(backend).recover(
            _prefix(tmp_path, chunk_count=1),
            expected_boundary_digest="boundary-ok",
        )

    assert "capture_boundary_digest" not in backend.calls
