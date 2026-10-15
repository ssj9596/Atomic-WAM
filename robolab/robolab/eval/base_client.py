# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is the source of truth. A verbatim copy lives at
# droid_plus/eval/base_client.py — keep both in sync when editing.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EpisodeStop(Exception):
    """Request a clean episode stop without shutting down the simulator."""


class EpisodeReset(Exception):
    """Request an in-place environment reset without stopping the simulator.

    Policy clients raise this control signal at a safe inference boundary (for
    example after an interactive ``/reset`` command).  The episode runner owns
    the environment object and handles the actual ``env.reset()`` call while
    keeping the Isaac process alive.
    """

    def __init__(self, reason: str = "interactive supervisor requested reset") -> None:
        super().__init__(reason)
        self.reason = reason


class EpisodeResume(Exception):
    """Request an in-place resume from a saved chunk boundary.

    ``direct_boundary_reset`` asks the runner to load the chunk's portable
    ``boundary_state.json`` and call ``env.reset_to()``.  The legacy exact
    action-replay path remains available by leaving it false.
    """

    def __init__(
        self,
        *,
        source_dir: str,
        through_chunk: int,
        destination_dir: str | None = None,
        env_id: int | None = None,
        direct_boundary_reset: bool = True,
    ) -> None:
        if through_chunk < 0:
            raise ValueError("through_chunk must be >= 0")
        self.source_dir = str(source_dir)
        self.through_chunk = int(through_chunk)
        self.destination_dir = str(destination_dir) if destination_dir else None
        self.env_id = env_id
        self.direct_boundary_reset = bool(direct_boundary_reset)
        super().__init__(
            f"resume replay through chunk {self.through_chunk} from {self.source_dir}"
        )


class InferenceClient(ABC):
    """Root client for policy inference.

    Subclass override surface, in order of increasing commitment:

    1. Implement the four hooks (``_extract_observation``, ``_pack_request``,
       ``_query_server``, ``_unpack_response``). Chunking, env-id bookkeeping,
       visualization, and reset are handled by the base.
    2. Additionally override ``_postprocess_chunk`` or ``_build_visualization``
       for action-space / logging quirks (gripper binarization, 7->8 padding).
    3. Override ``infer`` entirely if your flow isn't query-then-step-chunk
       (e.g. server-side session state, pre-step caching).

    Two hooks are meant to be split per concern:

      ``_extract_observation``  <- repo-specific (real-robot flat numpy dict vs
                                   sim nested torch batched dict)
      ``_pack_request``         <- backend-specific (wire keys, image sizes)

    Keeping these separate lets the same backend client be paired with
    different observation sources without duplicating the wire format.
    """

    # Subclasses override to match their server's chunk length.
    # horizon=1 is correct for single-action servers.
    open_loop_horizon: int = 1

    def __init__(self) -> None:
        # Per-env chunking state. Subclasses may ignore and manage state however
        # they want.
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}
        # Keep non-action response fields (such as a Reasoner subtask caption)
        # alongside the open-loop action chunk that they describe.
        self._response_metadata: dict[int, dict[str, Any]] = {}


    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        """Return ``{"action": np.ndarray, "viz": np.ndarray | None}``.

        Default flow: extract -> pack -> query -> unpack -> postprocess ->
        cache chunk -> step one action. Override entirely if your client needs
        a different control loop.
        """
        extracted = self._extract_observation(obs, env_id=env_id)

        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            chunk = self._unpack_response(response)
            chunk = self._postprocess_chunk(chunk)
            self._set_chunk(env_id, chunk)
            if isinstance(response, dict):
                self._response_metadata[env_id] = {
                    key: response[key]
                    for key in ("subtask", "gripper_trace", "mode")
                    if key in response
                }
            else:
                self._response_metadata.pop(env_id, None)

        action = self._next_action(env_id)
        viz = self._build_visualization(extracted)
        result = {"action": action, "viz": viz}
        result.update(self._response_metadata.get(env_id, {}))
        return result

    def reset(self, *, env_id: int | None = None) -> None:
        """Clear per-episode state. ``env_id=None`` resets all envs.

        Subclasses with server-side session state should override to notify
        the server, then call ``super().reset(env_id=env_id)``.
        """
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
            self._response_metadata.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)
            self._response_metadata.pop(env_id, None)

    def close(self) -> None:
        """Release transport resources. Default: no-op."""
        return None

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        """Public convenience wrapper for callers that want the viz image."""
        return self._build_visualization(self._extract_observation(obs, env_id=env_id))

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict:
        """Convert the caller's native obs into a flat dict of numpy arrays.

        Repo-specific seam. Return whatever keys ``_pack_request`` expects;
        the contract between these two methods is owned by the subclass pair.
        """

    @abstractmethod
    def _pack_request(self, extracted_obs: dict, instruction: str) -> Any:
        """Build the server's wire-format request. Backend-specific."""

    @abstractmethod
    def _query_server(self, request: Any) -> Any:
        """Send the request and return the raw response. Transport-specific."""

    @abstractmethod
    def _unpack_response(self, response: Any) -> np.ndarray:
        """Return a ``(horizon, action_dim)`` numpy array."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Action post-processing (binarization, padding, sign flips).

        Default: identity.
        """
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray | None:
        """Image for logging/recording. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Chunking helpers (usable or ignorable by subclasses)
    # ------------------------------------------------------------------

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action
