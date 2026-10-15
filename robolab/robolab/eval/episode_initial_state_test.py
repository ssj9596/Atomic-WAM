from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_episode_module(monkeypatch):
    """Load the CPU-testable episode helper without bootstrapping Isaac Sim."""

    stubs = {
        "robolab.core.logging.results": {
            "get_all_env_subtask_infos": lambda env: [],
        },
        "robolab.core.observations.observation_utils": {
            "unpack_image_obs": lambda *args, **kwargs: {},
            "unpack_viewport_cams": lambda *args, **kwargs: {},
        },
        "robolab.core.utils.video_utils": {
            "VideoWriter": object,
        },
        "robolab.core.world.world_state": {
            "get_world": lambda env: None,
        },
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).with_name("episode.py")
    spec = importlib.util.spec_from_file_location("robolab_eval_episode_cpu_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Scene:
    def __init__(self) -> None:
        self.calls = 0
        self.state = {"robot": {"joint_position": [[1.0, 2.0]]}}

    def get_state(self, *, is_relative: bool):
        assert is_relative is True
        self.calls += 1
        return self.state


def test_initial_scene_state_hook_queries_once_and_saves_every_env(monkeypatch) -> None:
    episode = _load_episode_module(monkeypatch)
    scene = _Scene()
    env = SimpleNamespace(scene=scene, num_envs=2)
    calls = []
    client = SimpleNamespace(
        initial_state_capture_enabled=True,
        save_initial_state=lambda state, *, env_id: calls.append((state, env_id)),
    )

    episode._save_initial_scene_states(env, client)

    assert scene.calls == 1
    assert calls == [(scene.state, 0), (scene.state, 1)]


def test_initial_scene_state_hook_does_not_query_for_ordinary_client(monkeypatch) -> None:
    episode = _load_episode_module(monkeypatch)
    scene = _Scene()
    env = SimpleNamespace(scene=scene, num_envs=1)
    client = SimpleNamespace(initial_state_capture_enabled=False)

    episode._save_initial_scene_states(env, client)

    assert scene.calls == 0
