# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy episode runner for RoboLab.

This module contains the run_episode function that executes a single
policy-controlled episode given any :class:`InferenceClient` subclass.
The function stays policy-agnostic — concrete clients live under
``policies/<policy>/client.py``.

Supports multi-env: one PolicyClient per env, per-env video writers,
actions inferred per active env and stacked for env.step().
"""

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _state_to_tensors(value, device):
    """Convert JSON scene-state arrays to the tensors expected by ``reset_to``."""

    if isinstance(value, dict):
        return {key: _state_to_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        try:
            return torch.as_tensor(value, dtype=torch.float32, device=device)
        except (TypeError, ValueError):
            return [_state_to_tensors(item, device) for item in value]
    return value


def _resume_boundary_path(source_dir: str, env_id: int, through_chunk: int) -> Path:
    """Resolve a chunk boundary in either flat-task or historical ``envN`` layout."""

    source = Path(source_dir).expanduser().resolve()
    chunk_name = f"chunk_{through_chunk:04d}"
    candidates = [
        source / chunk_name / "boundary_state.json",
        source / f"env{env_id}" / chunk_name / "boundary_state.json",
    ]
    if source.name == f"env{env_id}":
        candidates.insert(0, source / chunk_name / "boundary_state.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No boundary_state.json for chunk {through_chunk} under {source_dir}; searched: {searched}"
    )


def _format_gripper_caption(trace: object) -> str | None:
    """Format optional interactive gripper metadata for a video frame."""
    if not isinstance(trace, dict):
        return None
    def number(name: str, *fallback: str) -> str:
        value = trace.get(name)
        for key in (name,) + fallback:
            value = trace.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return f"{float(value):.3f}"
        return "n/a"
    command = trace.get("gripper_command")
    command_text = (
        f"{int(command)} ({'CLOSE' if int(command) else 'OPEN'})"
        if command in (0, 1) else "n/a"
    )
    pulse = " | isolated pulse filtered" if trace.get("gripper_isolated_filter_applied") is True else ""
    return (
        f"Gripper obs: {number('gripper_obs_pre')} | "
        f"model raw: {number('gripper_model_raw_command', 'gripper_raw_command')} | "
        f"server raw: {number('gripper_server_raw_command', 'gripper_raw_command')} | "
        f"exec: {command_text}{pulse}"
    )


def _overlay_subtask_text(frame, subtask: object, gripper_trace: object = None):
    """Draw optional subtask/gripper captions without changing raw frames."""
    captions = []
    if isinstance(subtask, str) and subtask.strip():
        captions.append("Subtask: " + " ".join(subtask.split()))
    gripper = _format_gripper_caption(gripper_trace)
    if gripper:
        captions.append(gripper)
    if frame is None or not captions:
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(0.7, width / 900.0))
    thickness = max(1, round(scale * 2))
    margin = max(6, round(min(height, width) * 0.012))
    max_width = max(1, width - 2 * margin)
    lines: list[str] = []
    for caption in captions:
        current = ""
        for word in caption.split():
            candidate = f"{current} {word}".strip()
            if current and cv2.getTextSize(candidate, font, scale, thickness)[0][0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    text_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
    line_height = text_height + max(4, margin // 2)
    max_lines = max(1, min(4, (height - 2 * margin) // max(1, line_height)))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "..."
    panel_height = min(height, 2 * margin + line_height * len(lines))
    cv2.rectangle(output, (0, 0), (width - 1, panel_height - 1), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (margin, margin + text_height + index * line_height),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def _overlay_gripper_text(frame, trace: object):
    return _overlay_subtask_text(frame, None, trace)

class TimingStats:
    """Simple timing utility for profiling code sections."""

    def __init__(self):
        self.times = defaultdict(list)
        self._start_times = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            del self._start_times[name]

    def to_dict(self, num_steps: int) -> dict:
        """Return timing summary as a dict for results logging."""
        d = {}
        for name, times in self.times.items():
            d[f"{name}_s"] = round(sum(times), 3)
            d[f"{name}_avg_ms"] = round(sum(times) / len(times) * 1000, 1) if times else 0
        d["wall_total_s"] = round(sum(sum(t) for t in self.times.values()), 3)
        d["it_per_sec"] = round(num_steps / d["wall_total_s"], 2) if d["wall_total_s"] > 0 else 0
        return d

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.eval.base_client import EpisodeReset, EpisodeResume, EpisodeStop, InferenceClient


def _clients_for_envs(client: InferenceClient | Sequence[InferenceClient], num_envs: int) -> list[InferenceClient]:
    """Normalize the legacy shared-client and explicit per-env client paths."""

    if isinstance(client, Sequence):
        clients = list(client)
        if len(clients) != num_envs:
            raise ValueError(f"Expected {num_envs} inference clients, got {len(clients)}")
        return clients
    return [client] * num_envs


def _save_initial_scene_states(
    env,
    client: InferenceClient | Sequence[InferenceClient],
) -> None:
    """Persist post-reset scene state only for clients that explicitly opt in."""

    clients = _clients_for_envs(client, env.num_envs)
    enabled = [
        callable(getattr(item, "save_initial_state", None))
        and bool(getattr(item, "initial_state_capture_enabled", False))
        for item in clients
    ]
    if not any(enabled):
        return
    initial_state = env.scene.get_state(is_relative=True)
    for env_id in range(env.num_envs):
        if enabled[env_id]:
            clients[env_id].save_initial_state(initial_state, env_id=env_id)


def run_episode(
    env,
    env_cfg,
    episode,
    client: InferenceClient | Sequence[InferenceClient],
    *,
    headless=False,
    save_videos=True,
    video_mode="all",
    max_episode_seconds: float | None = None,
    initial_reset_state: dict | None = None,
    stop_after_steps: int | None = None,
    reset_state_capture: dict | None = None,
):
    """Run a policy-controlled episode across all parallel envs.

    The policy client is constructed by the caller (typically a per-policy
    runner under ``policies/<policy>/run.py``). This function stays
    policy-agnostic.

    Args:
        env: The environment instance (RobolabEnv with num_envs >= 1)
        env_cfg: Environment configuration
        episode: Run index (each run produces num_envs episodes)
        client: One shared inference client (legacy path), or one client per
            environment.  The latter permits independent policy-server
            endpoints and parallel requests while retaining one synchronized
            vectorized simulator step.
        headless: If True, don't display video
        save_videos: If True, save per-env episode videos
        video_mode: Which videos to save: 'all', 'viewport', 'sensor', or 'none'
        max_episode_seconds: Optional evaluation cap in simulated seconds. If
            set, active envs are marked failed once the cap is reached.

    Returns:
        tuple: (env_results, subtask_status, timing)
            env_results: per-env dicts with {env_id, success, step}
            subtask_status: list of per-step subtask info dicts
            timing: dict with wall-clock timing breakdown
    """
    timer = TimingStats()
    initial_env_reset_seconds = None
    direct_reset_to_seconds = None

    clients = _clients_for_envs(client, env.num_envs)
    reset_started = time.perf_counter()
    obs, _ = env.reset()
    print("[episode] env.reset returned", flush=True)
    if initial_reset_state is not None:
        initial_env_reset_seconds = time.perf_counter() - reset_started
        logger.info("Direct-reset episode: base env.reset completed in %.3fs", initial_env_reset_seconds)
    if initial_reset_state is not None:
        direct_reset_started = time.perf_counter()
        obs, _ = env.reset_to(
            initial_reset_state,
            env_ids=torch.arange(env.num_envs, device=env.device, dtype=torch.long),
            is_relative=True,
        )
        if reset_state_capture is not None:
            reset_state_capture.clear()
            reset_state_capture.update(env.scene.get_state(is_relative=True))
        direct_reset_to_seconds = time.perf_counter() - direct_reset_started
        logger.info("Direct-reset episode: env.reset_to completed in %.3fs", direct_reset_to_seconds)
        print("[episode] env.reset_to returned", flush=True)
    elif env.num_envs == 1 and os.environ.get(
        "ROBO_INTERACTIVE_DIRECT_RESET_TO", "1"
    ).lower() in {"1", "true", "yes", "on"}:
        # CLI/branch-plan resumes can use the same portable boundary path as
        # an interactive EpisodeResume.  If anything is unavailable, retain
        # the existing replay configuration and let the client replay 0..N.
        resume_client = clients[0]
        source = getattr(resume_client, "replay_source_dir", None)
        through = getattr(resume_client, "replay_through_chunk", None)
        set_boundary_resume = getattr(resume_client, "set_boundary_resume", None)
        configure_episode = getattr(resume_client, "configure_episode", None)
        if source is not None and through is not None and callable(set_boundary_resume) and callable(configure_episode):
            try:
                boundary_path = _resume_boundary_path(str(source), 0, int(through))
                boundary_state = _state_to_tensors(
                    json.loads(boundary_path.read_text(encoding="utf-8")),
                    env.device,
                )
                obs, _ = env.reset_to(
                    boundary_state,
                    env_ids=torch.arange(
                        env.num_envs, device=env.device, dtype=torch.long
                    ),
                    is_relative=True,
                )
                destination = getattr(resume_client, "supervisor_dir", None)
                if destination is None:
                    raise RuntimeError("resume client has no supervisor_dir")
                configure_episode(
                    supervisor_dir=str(destination),
                    replay_source_dir=None,
                    replay_through_chunk=None,
                )
                set_boundary_resume(
                    source_dir=str(source),
                    through_chunk=int(through),
                    env_id=0,
                )
                seed_boundary_history = getattr(resume_client, "seed_boundary_history", None)
                if callable(seed_boundary_history):
                    seed_boundary_history(
                        source_dir=str(source),
                        through_chunk=int(through),
                        env_id=0,
                    )
                logger.info(
                    "Direct startup boundary reset succeeded: %s; next live chunk=%04d",
                    boundary_path,
                    int(through) + 1,
                )
            except Exception:
                logger.exception(
                    "Direct startup boundary reset failed; retaining exact action replay"
                )
    _save_initial_scene_states(env, clients)
    max_steps = env.max_episode_length
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    eval_max_steps = max_steps
    if max_episode_seconds is not None:
        if max_episode_seconds <= 0:
            raise ValueError(f"max_episode_seconds must be positive, got {max_episode_seconds}")
        eval_max_steps = min(max_steps, max(1, math.ceil(max_episode_seconds / step_dt)))
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz
    instruction = env_cfg.instruction
    # Preserve HDF5 trajectories by default; compact artifacts can still be
    # requested explicitly for lightweight runs.
    compact_artifacts = os.environ.get("ROBO_INTERACTIVE_COMPACT_ARTIFACTS", "0").lower() in {"1", "true", "yes", "on"}
    overlay_subtask_on_video = os.environ.get("OVERLAY_SUBTASK_ON_VIDEO", "0").lower() in {"1", "true", "yes", "on"}
    overlay_gripper_on_video = os.environ.get("OVERLAY_GRIPPER_ON_VIDEO", "0").lower() in {"1", "true", "yes", "on"}
    rotate_viewpoint_180 = os.environ.get("VIEWPOINT_ROTATE_180", "0").lower() in {"1", "true", "yes", "on"}
    # Pull action dim from the env's action manager (IsaacLab canonical),
    # falling back to the gym action space if the manager isn't available.
    action_dim = getattr(
        getattr(env, "action_manager", None),
        "total_action_dim",
        None,
    ) or env.action_space.shape[-1]

    subtask_status = []

    # Per-step mode snapshot for adaptive clients (empty list for regular clients).
    # Populated only when client exposes get_episode_mode_log() (opt-in, zero overhead otherwise).
    _has_mode_log = len(set(map(id, clients))) == 1 and callable(
        getattr(clients[0], "get_all_env_mode_stats", None)
    )

    # Distinct per-env clients are used by K-way rollouts so their independent
    # WAM servers receive the same pre-step observation concurrently.  The
    # legacy shared-client path remains sequential because one client may keep
    # mutable request context keyed by env_id.
    distinct_clients = len(set(map(id, clients))) == env.num_envs and env.num_envs > 1
    inference_pool = ThreadPoolExecutor(max_workers=env.num_envs) if distinct_clients else None

    # Set up per-run HDF5 file and per-env demo indices
    if not compact_artifacts and env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    # Setup per-env streaming video writers
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    # Define unconditionally so the finally clause below can iterate them either way.
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    actual_steps = 0
    _pbar = None  # tqdm progress bar reference, captured on first access
    adaptive_mode_stats = None
    forced_time_limit_envs: list[int] = []
    predicted_subtasks_seen: dict[int, list[str]] = defaultdict(list)
    gripper_traces_seen: dict[int, list[dict]] = defaultdict(list)
    supervisor_stopped = False
    in_place_resets = 0
    try:
        for step in (pbar := tqdm(range(eval_max_steps))):
            _pbar = pbar
            logger.info("Episode step %d: entering timeline/inference", actual_steps)
            print(f"[episode] step {actual_steps} timeline_playing={timeline.is_playing()}", flush=True)

            while not timeline.is_playing():
                # reset()/reset_to() may pause Isaac's timeline.  Restart it
                # here instead of spinning on kit_app.update() indefinitely;
                # the first inference/action must run on a live physics clock.
                timeline.play()
                kit_app.update()
            print(f"[episode] step {actual_steps} timeline_ready={timeline.is_playing()}", flush=True)

            timer.start("policy_inference")
            # Infer actions for all active (non-frozen) envs
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            last_mode: str | None = None
            predicted_subtasks: dict[int, str] = {}
            gripper_traces: dict[int, dict] = {}
            try:
                active_env_ids = list(env.active_env_ids)
                for env_id in active_env_ids:
                    # Persist a portable scene snapshot at every completed
                    # action-chunk boundary.  Replay clients additionally keep
                    # the historical replay_boundary_state.json alias, but a
                    # first-pass winner must also be continuable.
                    boundary_reached = getattr(clients[env_id], "chunk_boundary_reached", None)
                    save_boundary_state = getattr(clients[env_id], "save_boundary_state", None)
                    if callable(boundary_reached) and boundary_reached(env_id=env_id) and callable(save_boundary_state):
                        save_boundary_state(env.scene.get_state(is_relative=True), env_id=env_id)

                if inference_pool is None:
                    inferred = [
                        (env_id, clients[env_id].infer(obs, instruction, env_id=env_id))
                        for env_id in active_env_ids
                    ]
                else:
                    futures = {
                        env_id: inference_pool.submit(
                            clients[env_id].infer,
                            obs,
                            instruction,
                            env_id=env_id,
                        )
                        for env_id in active_env_ids
                    }
                    inferred = [(env_id, futures[env_id].result()) for env_id in active_env_ids]

                for env_id, ret in inferred:
                    actions[env_id] = torch.tensor(ret["action"], device=env.device)
                    predicted_subtask = ret.get("subtask")
                    if isinstance(predicted_subtask, str) and predicted_subtask.strip():
                        normalized = " ".join(predicted_subtask.split())
                        if not predicted_subtasks_seen[env_id] or predicted_subtasks_seen[env_id][-1] != normalized:
                            predicted_subtasks_seen[env_id].append(normalized)
                        if overlay_subtask_on_video:
                            predicted_subtasks[env_id] = normalized
                    trace = ret.get("gripper_trace")
                    if isinstance(trace, dict):
                        trace = dict(trace)
                        gripper_traces_seen[env_id].append(trace)
                        if overlay_gripper_on_video:
                            gripper_traces[env_id] = trace
                    if env_id == 0 or last_viz is None:
                        last_viz = ret.get("viz")
                    if env_id == 0 and "mode" in ret:
                        last_mode = ret["mode"]
            except EpisodeResume as exc:
                timer.stop("policy_inference")
                in_place_resets += 1
                logger.info("Resuming saved RoboLab chunks in place: %s", exc)
                reset_clients: set[int] = set()
                for env_client in clients:
                    if id(env_client) in reset_clients:
                        continue
                    reset_clients.add(id(env_client))
                    try:
                        env_client.reset()
                    except Exception:
                        logger.exception("Failed to reset inference client before resume")
                env_id = getattr(exc, "env_id", None)
                if env_id is None:
                    env_id = 0
                target_client = clients[int(env_id)]
                configure_episode = getattr(target_client, "configure_episode", None)
                if not callable(configure_episode):
                    raise RuntimeError(
                        "resume control requires a client with configure_episode()"
                    ) from exc
                destination = exc.destination_dir
                if destination is None:
                    destination = str(
                        getattr(target_client, "supervisor_dir", "")
                    ) + f"_resume_{exc.through_chunk:04d}"
                reset_eval_state = getattr(env, "reset_eval_state", None)
                if callable(reset_eval_state):
                    reset_eval_state()
                direct_reset = bool(getattr(exc, "direct_boundary_reset", True))
                direct_reset = direct_reset and os.environ.get(
                    "ROBO_INTERACTIVE_DIRECT_RESET_TO", "1"
                ).lower() in {"1", "true", "yes", "on"}
                direct_reset = direct_reset and env.num_envs == 1
                if direct_reset:
                    try:
                        boundary_path = _resume_boundary_path(
                            exc.source_dir, int(env_id), exc.through_chunk
                        )
                        boundary_state = _state_to_tensors(
                            json.loads(boundary_path.read_text(encoding="utf-8")),
                            env.device,
                        )
                        # A normal reset initializes the scene and all sensor
                        # handles; reset_to then replaces only the serialized
                        # physical state at the requested chunk boundary.
                        obs, _ = env.reset()
                        obs, _ = env.reset_to(
                            boundary_state,
                            env_ids=torch.arange(
                                env.num_envs, device=env.device, dtype=torch.long
                            ),
                            is_relative=True,
                        )
                        configure_episode(
                            supervisor_dir=destination,
                            replay_source_dir=None,
                            replay_through_chunk=None,
                        )
                        set_boundary_resume = getattr(target_client, "set_boundary_resume", None)
                        if callable(set_boundary_resume):
                            set_boundary_resume(
                                source_dir=exc.source_dir,
                                through_chunk=exc.through_chunk,
                                env_id=int(env_id),
                            )
                        else:
                            raise RuntimeError(
                                "direct boundary reset requires client.set_boundary_resume()"
                            )
                        seed_boundary_history = getattr(target_client, "seed_boundary_history", None)
                        if callable(seed_boundary_history):
                            seed_boundary_history(
                                source_dir=exc.source_dir,
                                through_chunk=exc.through_chunk,
                                env_id=int(env_id),
                            )
                        logger.info(
                            "Direct boundary reset succeeded: %s; continuing live inference in %s",
                            boundary_path,
                            destination,
                        )
                    except Exception:
                        logger.exception(
                            "Direct boundary reset failed; falling back to exact action replay"
                        )
                        configure_episode(
                            supervisor_dir=destination,
                            replay_source_dir=exc.source_dir,
                            replay_through_chunk=exc.through_chunk,
                        )
                        obs, _ = env.reset()
                else:
                    configure_episode(
                        supervisor_dir=destination,
                        replay_source_dir=exc.source_dir,
                        replay_through_chunk=exc.through_chunk,
                    )
                    obs, _ = env.reset()
                _save_initial_scene_states(env, clients)
                continue
            except EpisodeReset as exc:
                # Interactive supervisors may request a fresh task state while
                # debugging a live Isaac session.  Reset the policy caches and
                # RoboLab eval bookkeeping in place; do not close/relaunch the
                # simulator.  A vectorized rollout is reset as one unit so all
                # envs remain synchronized for the next action batch.
                timer.stop("policy_inference")
                in_place_resets += 1
                logger.info(
                    "Resetting RoboLab episode in place at step %d: %s",
                    actual_steps,
                    exc,
                )
                reset_clients: set[int] = set()
                for env_client in clients:
                    if id(env_client) in reset_clients:
                        continue
                    reset_clients.add(id(env_client))
                    try:
                        env_client.reset()
                    except Exception:
                        logger.exception("Failed to reset inference client in place")
                # RobolabEnv.reset_eval_state() clears its frozen-eval flags and
                # marks the next env.reset() as an initial reset.  Keep this
                # order so a terminated env is actually restored, rather than
                # being treated as another frozen episode.
                reset_eval_state = getattr(env, "reset_eval_state", None)
                if callable(reset_eval_state):
                    reset_eval_state()
                obs, _ = env.reset()
                _save_initial_scene_states(env, clients)
                continue
            except EpisodeStop as exc:
                timer.stop("policy_inference")
                supervisor_stopped = True
                logger.info("Stopping episode cleanly at a supervisor chunk boundary: %s", exc)
                break
            timer.stop("policy_inference")

            # Show adaptive mode in the tqdm bar (no-op for regular clients).
            if last_mode is not None and _pbar is not None:
                _pbar.set_postfix(mode=last_mode, refresh=False)

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            # Collect per-env subtask info (list of dicts, one per env)
            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            # Write per-env video frames (skip frozen envs)
            if save_videos:
                timer.start("video_write")
                for env_id in range(env.num_envs):
                    if env._frozen_envs[env_id]:
                        continue
                    if save_sensor:
                        frame_obs = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        if overlay_subtask_on_video or overlay_gripper_on_video:
                            frame_obs = _overlay_subtask_text(frame_obs, predicted_subtasks.get(env_id) if overlay_subtask_on_video else None, gripper_traces.get(env_id) if overlay_gripper_on_video else None)
                        video_writers_obs[env_id].write(frame_obs)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        if rotate_viewpoint_180 and frame_vp is not None:
                            frame_vp = cv2.flip(frame_vp, -1)
                        if overlay_subtask_on_video or overlay_gripper_on_video:
                            frame_vp = _overlay_subtask_text(frame_vp, predicted_subtasks.get(env_id) if overlay_subtask_on_video else None, gripper_traces.get(env_id) if overlay_gripper_on_video else None)
                        video_writers_viewport[env_id].write(frame_vp)
                timer.stop("video_write")

            actual_steps += 1

            if stop_after_steps is not None and actual_steps >= stop_after_steps:
                break

            # RobolabEnv freezes terminated envs and exports recordings automatically
            if env.all_terminated:
                break

        if (
            max_episode_seconds is not None
            and eval_max_steps < max_steps
            and not env.all_terminated
            and not supervisor_stopped
        ):
            forced_time_limit_envs = list(env.active_env_ids)
            if forced_time_limit_envs:
                logger.info(
                    "Stopping episode %s after max_episode_seconds=%s (%d/%d steps); "
                    "marking active envs as failed: %s",
                    episode,
                    max_episode_seconds,
                    actual_steps,
                    max_steps,
                    forced_time_limit_envs,
                )
                for env_id in forced_time_limit_envs:
                    env._frozen_envs[env_id] = True
                    env._env_results[env_id] = False
                    env._env_term_step[env_id] = actual_steps
                    episode_data = getattr(getattr(env, "recorder_manager", None), "_episodes", {}).get(env_id)
                    if episode_data is not None and hasattr(episode_data, "success"):
                        episode_data.success = False
                if not compact_artifacts and env.recorder_manager is not None:
                    try:
                        env.recorder_manager.export_episodes(env_ids=forced_time_limit_envs)
                    except Exception:
                        logger.exception(
                            "Failed to export time-limited recordings for env_ids=%s",
                            forced_time_limit_envs,
                        )
    finally:
        if inference_pool is not None:
            inference_pool.shutdown(wait=True, cancel_futures=True)
        for env_id, env_client in enumerate(clients):
            finalize_episode = getattr(env_client, "finalize_episode", None)
            if callable(finalize_episode):
                try:
                    finalize_episode(obs, instruction, env_ids=[env_id])
                except Exception:
                    logger.exception("Failed to persist final client episode artifacts for env%d", env_id)
            # A task can end in the middle of a chunk.  Save its exact scene
            # boundary after finalization so that the partial chunk is also
            # directly restorable with env.reset_to().
            save_boundary_state = getattr(env_client, "save_boundary_state", None)
            if callable(save_boundary_state):
                try:
                    save_boundary_state(env.scene.get_state(is_relative=True), env_id=env_id)
                except Exception:
                    logger.exception("Failed to persist final boundary state for env%d", env_id)
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        if _has_mode_log:
            try:
                adaptive_mode_stats = clients[0].get_all_env_mode_stats()
            except Exception:
                logger.exception("Failed to snapshot adaptive mode log")
        reset_clients: set[int] = set()
        for env_id, env_client in enumerate(clients):
            if id(env_client) in reset_clients:
                continue
            reset_clients.add(id(env_client))
            try:
                env_client.reset()
            except Exception:
                logger.exception("Failed to reset client after episode for env%d", env_id)

    timing = timer.to_dict(actual_steps)
    if initial_env_reset_seconds is not None:
        timing["initial_env_reset_s"] = round(initial_env_reset_seconds, 3)
    if direct_reset_to_seconds is not None:
        timing["direct_reset_to_s"] = round(direct_reset_to_seconds, 3)
    if predicted_subtasks_seen:
        timing["predicted_subtasks"] = {f"env{env_id}": values for env_id, values in predicted_subtasks_seen.items()}
    if gripper_traces_seen and not compact_artifacts:
        timing["gripper_trace_files"] = {}
        for env_id, traces in gripper_traces_seen.items():
            trace_path = os.path.join(get_output_dir(), f"gripper_trace_{episode}_env{env_id}.json")
            with open(trace_path, "w", encoding="utf-8") as stream:
                json.dump(traces, stream, indent=2)
                stream.write("\n")
            timing["gripper_trace_files"][f"env{env_id}"] = trace_path
    if max_episode_seconds is not None:
        timing["max_episode_seconds"] = float(max_episode_seconds)
        timing["max_episode_steps"] = int(eval_max_steps)
        timing["time_limited_envs"] = forced_time_limit_envs
    if supervisor_stopped:
        timing["supervisor_stopped"] = True
    if in_place_resets:
        timing["in_place_resets"] = in_place_resets

    # ------------------------------------------------------------------
    # Adaptive-mode log (opt-in: only if client exposes the API).
    # Writes one JSON file per env alongside the HDF5 / video outputs and
    # also injects compact per-env stats into the timing dict so they reach
    # episode_results.jsonl without any changes to summarize_run / runner.
    # ------------------------------------------------------------------
    if _has_mode_log and not compact_artifacts:
        try:
            all_stats = adaptive_mode_stats or {}
            timing["adaptive_mode"] = {}
            for env_id, stats in all_stats.items():
                # Compact stats for episode_results.jsonl (no per-step detail).
                compact = {
                    k: v for k, v in stats.items()
                    if k not in ("transitions", "chunk_events")
                }
                timing["adaptive_mode"][f"env{env_id}"] = compact

                # Full log with every transition → standalone JSON file.
                log_path = os.path.join(
                    get_output_dir(),
                    f"adaptive_mode_log_{episode}_env{env_id}.json",
                )
                with open(log_path, "w") as _fh:
                    json.dump(stats, _fh, indent=2)
                logger.info("[AdaptiveChunk] mode log written → %s", log_path)
        except Exception:
            logger.exception("Failed to write adaptive mode log")

    return env.get_env_results(), subtask_status, timing
