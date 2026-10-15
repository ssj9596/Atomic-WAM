# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket inference server for RoboLab/openpi-style Action policy serving.

The server uses OpenPI's WebsocketPolicyServer and speaks its msgpack+NumPy protocol:

- on connection, it sends an empty metadata dict;
- each client message is an observation dict;
- each response is a dict with ``action`` and, when enabled, ``video`` and/or
  an autoregressively decoded ``subtask``.

Example:

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
    --port 8000
"""

# Initialize the script runtime before any cosmos-framework imports, mirroring
# the LIBERO server.  The shared helpers below (single-rank distributed init,
# local IP discovery, frozen-config EMA disable) live in
# ``action_policy_server_utils`` so this module doesn't have to import from
# the sibling LIBERO server just to share runtime utilities.
from cosmos_framework.inference.common.init import init_script

init_script()

import json
import os
import socket
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import Any, Literal

import numpy as np
import omegaconf
import pydantic
import safetensors.torch
import torch
import torch.nn.functional as F
import tyro
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.tensor import DTensor

from cosmos_framework.data.vfm.action.camera_condition import (
    compose_exterior_plucker_condition,
    opengl_camera_pose_to_opencv_matrix,
    scaled_camera_matrix,
)
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.vfm.action.prompt_templates import format_action_prediction_prompt
from cosmos_framework.data.vfm.action.reasoner_targets import (
    reasoner_final_subtask,
    rewrite_move_open_gripper_text,
)
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import ConfigFileType, ConfigOverrides, tyro_cli
from cosmos_framework.inference.common.config import deserialize_config_dict
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.vfm.lora import iter_lora_modules, lora_runtime_context
from cosmos_framework.callbacks.reasoner_lora_export import canonical_reasoner_lora_key

_DEFAULT_DROID_POLICY_CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"
_DEFAULT_CONDITIONING_FPS = 15.0
_DEFAULT_ACTION_CHUNK_SIZE = 32
_DEFAULT_IMAGE_HEIGHT = 540
_DEFAULT_IMAGE_WIDTH = 640
_DEFAULT_ACTION_DIM = 8
_DEFAULT_ROBOLAB_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robolab"
_REASONER_HISTORY_KEY = "reasoner/history_images"
_REASONER_SEPARATE_VIEWS_KEY = "reasoner/separate_views"
_REASONER_PREVIEW_ACTION_KEY = "reasoner/preview_action"
# Match the trained three-image Reasoner choice exactly: two causal frames at
# the raw 15 Hz control rate followed by the current frame.
_REASONER_HISTORY_OFFSETS = (-16, -8, 0)
_REASONER_SEPARATE_HISTORY_OFFSETS = (-16, 0)
_REASONER_SEPARATE_THREE_GROUP_HISTORY_OFFSETS = (-16, -8, 0)
_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
    "sides, with the robot visible."
)
_DEFAULT_HF_REVISION = "main"
_ROBOLAB_EXTERNAL_CAMERA_MATRIX = (500.0, 640.0, 500.0, 360.0)  # [fx,cx,fy,cy] at 1280x720
_ROBOLAB_EXTERNAL_SOURCE_SIZE = (1280, 720)
_ROBOLAB_LEFT_CAMERA_POSE = ((0.05, 0.57, 0.66), (-0.393, -0.195, 0.399, 0.805))
_ROBOLAB_RIGHT_CAMERA_POSE = ((0.05, -0.57, 0.66), (0.805, 0.399, -0.195, -0.393))
_ROBOLAB_POLICY_HF_REPOSITORIES = {
    "Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
    "nvidia/Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
}

ActionSpace = Literal["joint_pos", "midtrain"]


@dataclass(frozen=True)
class _VitTextConditioning:
    conditional_tokens: list[list[int]]
    unconditional_tokens: list[list[int]]
    conditional_position_ids: list[torch.Tensor]
    unconditional_position_ids: list[torch.Tensor]
    conditional_soft_plan_insert_indexes: list[int] | None = None
    unconditional_soft_plan_insert_indexes: list[int] | None = None


def _single_tensor(value: Any, key: str) -> torch.Tensor:
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one {key} tensor for RoboLab inference, got {len(value)}.")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {key} to be a tensor, got {type(value).__name__}.")
    return value


def _single_text_token_list(value: Any, key: str) -> list[int]:
    tokens = _single_tensor(value, key)
    while tokens.ndim > 1 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 1:
        raise ValueError(f"Expected {key} shape [N] or [1,N], got {tuple(tokens.shape)}.")
    return tokens.to(dtype=torch.long).tolist()


def _single_text_position_ids(value: Any, key: str) -> torch.Tensor:
    position_ids = _single_tensor(value, key)
    while position_ids.ndim > 2 and position_ids.shape[0] == 1:
        position_ids = position_ids[0]
    if position_ids.ndim != 2 or position_ids.shape[0] != 3:
        raise ValueError(f"Expected {key} shape [3,N] or [1,3,N], got {tuple(position_ids.shape)}.")
    return position_ids


@contextmanager
def _use_vit_text_conditioning(model: Any, conditioning: _VitTextConditioning):
    """Use training-format multimodal text inputs for one inference request."""
    original_get_text_tokens = model._get_inference_text_tokens
    original_pack_input_sequence = model._pack_input_sequence

    def get_text_tokens(_model: Any, _data_batch: dict, has_negative_prompt: bool):
        if has_negative_prompt:
            raise ValueError("RoboLab ViT conditioning does not support an explicit negative prompt.")
        return conditioning.conditional_tokens, conditioning.unconditional_tokens

    def pack_input_sequence(
        _model: Any,
        sequence_plans: Any,
        input_text_indexes: list[list[int]],
        gen_data_clean: Any,
        input_timesteps: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ):
        if input_text_indexes is conditioning.conditional_tokens:
            text_position_ids = conditioning.conditional_position_ids
            soft_plan_insert_indexes = conditioning.conditional_soft_plan_insert_indexes
        elif input_text_indexes is conditioning.unconditional_tokens:
            text_position_ids = conditioning.unconditional_position_ids
            soft_plan_insert_indexes = conditioning.unconditional_soft_plan_insert_indexes
        else:
            text_position_ids = gen_data_clean.text_position_ids
            soft_plan_insert_indexes = gen_data_clean.soft_plan_insert_indexes
        if soft_plan_insert_indexes is not None:
            # The empty CFG prompt is shorter than the task-conditioned prompt.
            # Keep each branch's semantic P-token boundary paired with its own
            # tokens/position IDs instead of reusing the conditional boundary.
            kwargs["soft_plan_insert_indexes"] = soft_plan_insert_indexes
        return original_pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            replace(
                gen_data_clean,
                text_position_ids=text_position_ids,
                soft_plan_insert_indexes=soft_plan_insert_indexes,
            ),
            input_timesteps,
            *args,
            **kwargs,
        )

    missing = object()
    instance_dict = vars(model)
    previous_get_text_tokens = instance_dict.get("_get_inference_text_tokens", missing)
    previous_pack_input_sequence = instance_dict.get("_pack_input_sequence", missing)
    model._get_inference_text_tokens = MethodType(get_text_tokens, model)
    model._pack_input_sequence = MethodType(pack_input_sequence, model)
    try:
        yield
    finally:
        if previous_get_text_tokens is missing:
            delattr(model, "_get_inference_text_tokens")
        else:
            model._get_inference_text_tokens = previous_get_text_tokens
        if previous_pack_input_sequence is missing:
            delattr(model, "_pack_input_sequence")
        else:
            model._pack_input_sequence = previous_pack_input_sequence


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_policy_timing(args: "RobolabServerArgs", inferred: dict[str, Any]) -> tuple[int, float, float]:
    video_sample_stride = _maybe_int(args.video_sample_stride)
    if video_sample_stride is None:
        video_sample_stride = _maybe_int(inferred.get("video_sample_stride"))
    if video_sample_stride is None:
        video_sample_stride = 1
    if video_sample_stride < 1:
        raise ValueError("--video-sample-stride must be >= 1")

    action_fps = _maybe_float(args.action_fps)
    if action_fps is None:
        action_fps = _maybe_float(inferred.get("action_fps"))
    if action_fps is None:
        action_fps = _DEFAULT_CONDITIONING_FPS
    if action_fps <= 0:
        raise ValueError("--action-fps must be > 0")

    conditioning_fps = _maybe_float(args.conditioning_fps)
    if conditioning_fps is None:
        conditioning_fps = _maybe_float(inferred.get("conditioning_fps"))
    if conditioning_fps is None:
        conditioning_fps = action_fps / video_sample_stride
    if conditioning_fps <= 0:
        raise ValueError("--conditioning-fps must be > 0")

    return video_sample_stride, conditioning_fps, action_fps


def _load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    if "://" in checkpoint_path:
        return None
    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not metadata_path.exists():
        return None
    metadata = deserialize_config_dict(metadata_path)
    # Portable exports keep the training config next to checkpoint.json.  The
    # metadata intentionally stores a relative name so the bundle can be
    # copied anywhere; resolve it only at runtime from this checkpoint.
    config_file = metadata.get("config_file")
    if isinstance(config_file, str) and config_file and not Path(config_file).is_absolute():
        metadata["config_file"] = str(checkpoint_dir / config_file)
    return metadata


def _load_training_config_from_metadata(metadata: dict[str, Any]) -> Any | None:
    config_file = metadata.get("config_file")
    if not isinstance(config_file, str) or not config_file:
        return None
    config_file_type_raw = metadata.get("config_file_type")
    config_overrides = ConfigOverrides(
        config_file=config_file,
        experiment=str(metadata.get("experiment", "")),
        experiment_overrides=list(metadata.get("experiment_overrides", [])),
    )
    config_args = config_overrides.build_config()
    if isinstance(config_file_type_raw, str) and config_file_type_raw:
        config_args = config_args.model_copy(update={"config_file_type": ConfigFileType(config_file_type_raw)})
    return config_args.load_config()


def _load_training_config(setup_args: OmniSetupArgs, checkpoint_path: str) -> Any | None:
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if metadata is not None:
        try:
            config = _load_training_config_from_metadata(metadata)
        except Exception as exc:
            log.warning(f"[robolab-policy-server] could not load checkpoint metadata config for transforms: {exc}")
            config = None
            config_file = metadata.get("config_file")
            config_file_type = metadata.get("config_file_type")
            if config_file_type == "yaml" and isinstance(config_file, str) and Path(config_file).is_file():
                try:
                    config = omegaconf.OmegaConf.load(config_file)
                except Exception as yaml_exc:
                    log.warning(
                        f"[robolab-policy-server] could not load checkpoint metadata yaml fallback for transforms: "
                        f"{yaml_exc}"
                    )
        if config is not None:
            return config

    if setup_args.config_file_type == ConfigFileType.MODULE and not setup_args.experiment:
        return None

    try:
        return setup_args.load_config()
    except Exception as exc:
        log.warning(f"[robolab-policy-server] could not load training config for transforms: {exc}")
        return None


def _load_nearest_local_training_yaml(checkpoint_path: str) -> Any | None:
    """Load a persisted run ``config.yaml`` from a local checkpoint ancestry.

    A DCP iteration contains only ``model/.metadata`` and shard files; the
    training run's resolved transform settings live one or more directories
    above it.  This lookup is used only by mode2's standalone WAM child, so
    legacy serving paths keep their historical metadata/module fallback.
    """

    if "://" in checkpoint_path:
        return None
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        path = path.parent
    # The run root is normally at most four parents above
    # ``.../checkpoints/iter_N/model``.  Walking to the filesystem root is
    # harmless and also supports a caller passing the run root directly.
    for ancestor in (path, *path.parents):
        config_path = ancestor / "config.yaml"
        if not config_path.is_file():
            continue
        try:
            return omegaconf.OmegaConf.load(config_path)
        except Exception as exc:
            log.warning(f"[robolab-policy-server] could not load WAM run config {config_path} for transforms: {exc}")
            return None

    return None


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an OmegaConf mapping or a namespace-like config."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_bool(value: Any) -> bool:
    """Interpret persisted config booleans without treating ``"false"`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _find_rank_partitioned_action_dataset_config(training_config: Any) -> Any | None:
    """Find the action dataset config used by the serving transform.

    Older module recipes expose ``dataloaders.action_data.dataloader.dataset``;
    the single-stream DROID TOML recipe exposes
    ``dataloader_train.dataloader.datasets.droid.dataset``.  The opt-in
    Reasoner/WAM recipe instead wraps two child loaders and stores the WAM
    dataset at
    ``dataloader_train.wam_loader.distributor.datasets.*.dataset``.  WAM is
    deliberately preferred when that paired layout is present because it
    carries the action-serving transform settings (stride, resolution, and
    first-frame ViT options); the Reasoner child has a VLM-only schema and
    must never be used to construct an action transform.

    The fallback order is intentional: paired WAM -> legacy rank-partitioned
    action loader.  Existing non-paired configs therefore follow exactly the
    previous path.
    """

    def _dataset_from_container(container: Any, *, prefer_wam: bool) -> Any | None:
        datasets = _config_value(container, "datasets")
        if datasets is None:
            return None
        if hasattr(datasets, "items"):
            candidates = datasets.items()
        elif hasattr(datasets, "values"):
            candidates = ((None, entry) for entry in datasets.values())
        elif isinstance(datasets, (list, tuple)):
            candidates = enumerate(datasets)
        else:
            candidates = ((None, entry) for entry in ())
        first_candidate: Any | None = None
        for dataset_name, dataset_entry in candidates:
            candidate = _config_value(dataset_entry, "dataset")
            if candidate is None:
                continue
            if first_candidate is None:
                first_candidate = candidate
            target = str(_config_value(candidate, "_target_", "")).lower()
            if prefer_wam:
                # The paired recipe's WAM factory is intentionally explicit;
                # avoid accidentally selecting the sibling Reasoner stream if
                # a caller provides both entries under one distributor.
                name = str(dataset_name or "").lower()
                if name == "wam" or "get_droid_wam_stream_dataset" in target:
                    return candidate
            elif "get_action_" in target and "sft_dataset" in target:
                return candidate
        # Preserve the historical first-entry behavior for legacy action
        # loaders.  A paired loader with an unexpected/non-WAM target should
        # fall through to the legacy lookup rather than selecting the Reasoner
        # child by accident.
        return first_candidate if not prefer_wam else None

    dataloader_train = _config_value(training_config, "dataloader_train")

    # Reasoner/WAM paired loader.  Only inspect the WAM child; the Reasoner
    # child emits ShareGPT/VLM records and is not compatible with
    # ActionTransformPipeline.
    wam_loader = _config_value(dataloader_train, "wam_loader")
    if wam_loader is not None:
        paired_candidate = _dataset_from_container(
            _config_value(wam_loader, "distributor"),
            prefer_wam=True,
        )
        if paired_candidate is not None:
            return paired_candidate

    # Legacy rank-partitioned action loader used by baseline/mid-soft runs.
    dataloader = _config_value(dataloader_train, "dataloader")
    return _dataset_from_container(dataloader, prefer_wam=False)


def _build_action_transform_from_config(
    action_dataset_config: Any,
    *,
    args: Any,
    max_action_dim: int,
    model_mid_soft: bool,
) -> tuple[ActionTransformPipeline, dict[str, Any]]:
    """Build an eval transform directly from a ``get_action_*_sft_dataset`` config."""
    use_exterior_plucker = bool(_config_value(action_dataset_config, "use_exterior_plucker_condition", False))
    video_sample_stride = _maybe_int(_config_value(action_dataset_config, "video_sample_stride")) or 1
    if video_sample_stride < 1:
        raise ValueError(f"training config video_sample_stride must be >= 1, got {video_sample_stride}")
    fps = _maybe_float(_config_value(action_dataset_config, "fps"))
    mid_soft = bool(_config_value(action_dataset_config, "mid_soft", model_mid_soft))
    tokenizer_config = _config_value(action_dataset_config, "tokenizer_config")
    persisted_json_mode = _config_bool(_config_value(action_dataset_config, "format_prompt_as_json", False))
    format_prompt_as_json = (
        persisted_json_mode if args.format_prompt_as_json is None else bool(args.format_prompt_as_json)
    )
    persisted_execution_success = _config_bool(
        _config_value(action_dataset_config, "include_execution_success", False)
    )
    include_execution_success = (
        persisted_execution_success
        if args.include_execution_success is None
        else bool(args.include_execution_success)
    )

    transform = ActionTransformPipeline(
        pad_keys=["video", "plucker_condition"] if use_exterior_plucker else None,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=0.0,
        camera_cfg_dropout_rate=0.0,
        max_action_dim=max_action_dim,
        append_viewpoint_info=bool(_config_value(action_dataset_config, "append_viewpoint_info", True)),
        append_duration_fps_timestamps=bool(
            _config_value(action_dataset_config, "append_duration_fps_timestamps", True)
        ),
        append_resolution_info=bool(_config_value(action_dataset_config, "append_resolution_info", True)),
        append_idle_frames=bool(_config_value(action_dataset_config, "append_idle_frames", False)),
        format_prompt_as_json=format_prompt_as_json,
        include_vit_first_frame=bool(args.include_vit_first_frame),
        vit_processor_model_name=args.vit_processor_model_name,
        vit_processor_config_variant=args.vit_processor_config_variant,
        vit_processor_cache_dir=args.vit_processor_cache_dir,
        vit_first_frame_max_pixels=args.vit_first_frame_max_pixels,
        vit_first_frame_min_pixels=args.vit_first_frame_min_pixels,
        mid_soft=mid_soft,
        # The training-only SIM-CoT decoder is disabled in exported checkpoints;
        # inference must not require subtask labels just to build the prompt.
        tokenize_subtask=False,
        subtask_max_length=128,
    )
    inferred: dict[str, Any] = {
        "video_sample_stride": video_sample_stride,
        "format_prompt_as_json": format_prompt_as_json,
        "include_execution_success": include_execution_success,
    }
    chunk_length = _maybe_int(_config_value(action_dataset_config, "chunk_length"))
    if chunk_length is not None:
        inferred["action_chunk_size"] = chunk_length
    if fps is not None:
        inferred["action_fps"] = fps
        inferred["conditioning_fps"] = fps / video_sample_stride
    resolution = _config_value(action_dataset_config, "resolution")
    if resolution is not None:
        inferred["resolution"] = str(resolution)
    return transform, inferred


def _resolve_checkpoint_path(checkpoint_path: str, *, hf_revision: str) -> str:
    if Path(checkpoint_path).expanduser().exists():
        return checkpoint_path

    repository = _ROBOLAB_POLICY_HF_REPOSITORIES.get(checkpoint_path)
    if repository is None:
        return checkpoint_path

    log.info(
        f"[robolab-policy-server] downloading consolidated checkpoint from Hugging Face: "
        f"repository={repository!r} revision={hf_revision!r}"
    )
    return CheckpointDirHf(repository=repository, revision=hf_revision).download()


def _resolve_latest_dcp_checkpoint(checkpoint_path: str) -> str:
    """Resolve a mode2 WAM job/output root to its newest DCP iteration.

    The ordinary server still receives an explicit checkpoint directory.  The
    mode2 launcher is often handed a training job root instead (the path that
    contains ``checkpoints/iter_*``), so resolve that shape only when an
    unambiguous iteration directory exists.  Safetensors and explicit DCP
    paths are returned unchanged.
    """

    if "://" in checkpoint_path:
        return checkpoint_path
    root = Path(checkpoint_path).expanduser()
    candidates: list[Path] = []
    for parent in (root, root / "checkpoints"):
        if not parent.is_dir():
            continue
        candidates.extend(
            path
            for path in parent.glob("iter_*")
            if path.is_dir() and (path / "model").is_dir() and (path / "model" / ".metadata").is_file()
        )
    # Training launchers commonly expose the outer output root rather than
    # the innermost job root, for example::
    #
    #   <output>/<project>/<group>/<job>/checkpoints/iter_*
    #
    # Search that nested shape only when the direct forms above did not
    # match.  Refuse to choose between multiple job roots: comparing their
    # iteration numbers would silently route inference to an unrelated run.
    if not candidates and root.is_dir():
        nested_candidates = [
            path
            for checkpoints_dir in root.rglob("checkpoints")
            if checkpoints_dir.is_dir()
            for path in checkpoints_dir.glob("iter_*")
            if path.is_dir() and (path / "model").is_dir() and (path / "model" / ".metadata").is_file()
        ]
        nested_run_roots = {path.parent.parent.resolve() for path in nested_candidates}
        if len(nested_run_roots) > 1:
            preview = ", ".join(str(path) for path in sorted(nested_run_roots)[:3])
            raise ValueError(
                "WAM output root contains multiple checkpoint runs; pass one concrete job root or DCP iteration: "
                f"{preview}"
            )
        candidates.extend(nested_candidates)
    if not candidates:
        return checkpoint_path

    # Prefer the numeric iteration suffix; lexical ordering would put 10000
    # before 9000 in some mixed naming schemes.
    def _iteration(path: Path) -> int:
        try:
            return int(path.name.removeprefix("iter_"))
        except ValueError:
            return -1

    return str(max(candidates, key=_iteration))


def _ensure_distinct_mode2_checkpoints(reasoner_path: str, wam_path: str) -> None:
    """Reject an accidental single-checkpoint mode2 configuration."""

    if "://" in reasoner_path or "://" in wam_path:
        same = reasoner_path.rstrip("/") == wam_path.rstrip("/")
    else:
        same = Path(reasoner_path).expanduser().resolve() == Path(wam_path).expanduser().resolve()
    if same:
        raise ValueError(
            "Reasoner-WAM mode2 requires distinct checkpoints: pass "
            "--reasoner-checkpoint-path for the LoRA Reasoner and "
            "--wam-checkpoint-path for the standalone no-LoRA WAM."
        )


def _iter_lora_safetensor_files(checkpoint_path: Path) -> list[Path]:
    """Return adapter shard files from a file/directory artifact."""

    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Reasoner LoRA path does not exist: {checkpoint_path}")
    # Allow callers to pass the DCP iteration root directly.  The standalone
    # Reasoner exporter writes the deployment artifact as a sibling of
    # model/optim/scheduler/trainer inside that directory.
    canonical = checkpoint_path / "reasoner_lora.safetensors"
    if canonical.exists():
        return [canonical]
    preferred = checkpoint_path / "adapter_model.safetensors"
    if preferred.exists():
        return [preferred]
    index = checkpoint_path / "model.safetensors.index.json"
    if not index.exists():
        index = checkpoint_path / "adapter_model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map", {})
        names = sorted({str(name) for name in weight_map.values()})
        files = [checkpoint_path / name for name in names]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Reasoner LoRA index references missing shards: {missing[:3]}")
        return files
    # A plain ``model.safetensors`` is convenient for the small extracted
    # artifact and keeps the command independent of the export shard layout.
    plain = checkpoint_path / "model.safetensors"
    if plain.exists():
        return [plain]
    files = sorted(checkpoint_path.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors files found under Reasoner LoRA path: {checkpoint_path}")


def _iter_projector_safetensor_files(checkpoint_path: Path) -> list[Path]:
    """Return the standalone Reasoner projector artifact from a file/directory."""

    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Reasoner projector path does not exist: {checkpoint_path}")
    canonical = checkpoint_path / "reasoner_projector.safetensors"
    if canonical.exists():
        return [canonical]
    raise FileNotFoundError(f"No reasoner_projector.safetensors found under: {checkpoint_path}")


def _load_reasoner_partial_weights(
    model: Any,
    *,
    source: dict[str, torch.Tensor],
    artifact_path: str,
    label: str,
) -> int:
    """Map and load a validated partial Reasoner state into plain/FSDP models.

    Exported Cosmos keys are commonly prefixed with ``model.`` while the
    inner ``OmniMoTModel`` used by the RoboLab service starts at ``net.``.
    Matching therefore tries exact keys, known prefix removals, and a unique
    suffix match. ``source`` must already be restricted to the intended
    component so this helper cannot overwrite unrelated WAM/base weights.
    """

    # ``OmniInference`` normally returns an FSDP/DTensor-backed
    # ``OmniMoTModel``.  The ordinary ``Module.load_state_dict`` path expects
    # local shard shapes and therefore cannot reliably ingest the full
    # adapter tensors exported by safetensors.  The distributed state-dict
    # API performs the necessary global-to-local mapping.  Keep a plain
    # ``state_dict`` fallback for the lightweight CPU test doubles and for
    # older torch versions that do not expose the state-dict API.
    try:
        target_state = get_model_state_dict(model)
    except Exception:
        target_state = model.state_dict()
    target_is_distributed = any(isinstance(value, DTensor) for value in target_state.values())
    target_keys = set(target_state)
    mapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    for source_key, value in source.items():
        candidates = [
            source_key,
            source_key.removeprefix("model."),
            source_key.removeprefix("module."),
            source_key.removeprefix("model.").removeprefix("module."),
        ]
        target_key = next((candidate for candidate in candidates if candidate in target_keys), None)
        if target_key is None:
            suffix_matches = [key for key in target_keys if key.endswith(candidates[1])]
            if len(suffix_matches) == 1:
                target_key = suffix_matches[0]
        if target_key is None and label == "projector":
            # FSDP/activation-checkpoint wrappers may be inserted inside the
            # visual module path, so ``visual.<component>`` is not guaranteed
            # to remain contiguous.  The exported artifact is already limited
            # to exactly these two projector components; match their stable
            # terminal path, require a unique target, then enforce exact shape.
            component_suffix = None
            for marker in ("deepstack_merger_list.", "merger."):
                marker_pos = source_key.find(marker)
                if marker_pos >= 0:
                    component_suffix = source_key[marker_pos:]
                    break
            if component_suffix is not None:
                component_matches = [
                    key
                    for key in target_keys
                    if key.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "").endswith(
                        component_suffix
                    )
                ]
                if len(component_matches) == 1:
                    target_key = component_matches[0]
        if target_key is None:
            unmatched.append(source_key)
            continue
        if tuple(target_state[target_key].shape) != tuple(value.shape):
            raise ValueError(
                f"Reasoner {label} tensor shape mismatch for {source_key!r} -> {target_key!r}: "
                f"artifact={tuple(value.shape)} model={tuple(target_state[target_key].shape)}"
            )
        mapped[target_key] = value

    if unmatched:
        preview = ", ".join(unmatched[:3])
        target_preview = ""
        if label == "projector":
            projector_targets = sorted(
                f"{key}:{tuple(value.shape)}"
                for key, value in target_state.items()
                if "merger" in key or "deepstack" in key or "visual" in key
            )
            named_projector_targets = sorted(
                f"{key}:{tuple(value.shape)}"
                for key, value in model.named_parameters()
                if "merger" in key or "deepstack" in key or "visual" in key
            )
            target_preview = (
                f"; state-dict projector/visual targets={projector_targets[:40]}"
                f"; named projector/visual parameters={named_projector_targets[:40]}"
            )
        raise ValueError(
            f"Could not map {len(unmatched)} Reasoner {label} tensors into the model ({preview})"
            f"{target_preview}"
        )
    if not mapped:
        raise ValueError(f"Reasoner {label} artifact mapped zero tensors into the model")
    try:
        incompatible = set_model_state_dict(
            model,
            # ``set_model_state_dict`` may augment/mutate the mapping while
            # normalizing a partial FSDP state dict.  Keep our adapter-only
            # mapping intact so the acceptance count and safety checks below
            # cannot accidentally include unrelated base parameters.
            model_state_dict=dict(mapped),
            # The adapter is read from safetensors as ordinary CPU tensors.
            # FSDP2's default ``full_state_dict=False`` expects DTensor values
            # and raises ``mixed torch.Tensor and DTensor`` while copying the
            # LoRA parameters.  ``full_state_dict=True`` lets the distributed
            # loader scatter these complete adapter tensors into each local
            # shard; it does not gather the (30+ GB) model because only the
            # supplied partial ``mapped`` state is loaded.
            options=StateDictOptions(strict=False, full_state_dict=True),
        )
    except Exception as exc:
        if target_is_distributed:
            # A failed FSDP load must be fatal. Falling through to ordinary
            # ``Module.load_state_dict`` cannot correctly scatter full CPU
            # adapter tensors and may leave a partially-updated model behind.
            raise RuntimeError(f"Failed to load the Reasoner {label} weights into the FSDP model") from exc
        # A plain nn.Module (including the unit-test double) has no FSDP
        # sharding to reconcile.  Falling back here keeps that path usable;
        # for a real distributed model the first branch is expected to work.
        log.debug(f"[robolab-policy-server] distributed {label} load fallback: {exc}")
        incompatible = model.load_state_dict(mapped, strict=False)
    # ``load_state_dict`` can report unrelated missing keys; only enforce that
    # every adapter key was accepted by the target model.
    unexpected = set(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise ValueError(f"Unexpected Reasoner {label} keys after load: {sorted(unexpected)[:3]}")
    log.info(
        f"[robolab-policy-server] loaded standalone Reasoner {label}: "
        f"tensors={len(mapped)} path={artifact_path!r}"
    )
    return len(mapped)


def _load_reasoner_lora_weights(model: Any, lora_path: str) -> int:
    """Load only ``lora_*`` tensors from an extracted adapter artifact."""

    files = _iter_lora_safetensor_files(Path(lora_path).expanduser())
    source: dict[str, torch.Tensor] = {}
    for file_path in files:
        source.update(safetensors.torch.load_file(str(file_path), device="cpu"))
    source = {
        key: value.to(dtype=torch.bfloat16) if torch.is_floating_point(value) else value
        for key, value in source.items()
        if "lora_" in key
    }
    if not source:
        raise ValueError(f"Reasoner LoRA artifact contains no lora_ tensors: {lora_path}")
    return _load_reasoner_partial_weights(
        model,
        source=source,
        artifact_path=lora_path,
        label="LoRA",
    )


def _load_reasoner_projector_weights(model: Any, projector_path: str) -> int:
    """Load only visual merger/deepstack-merger tensors into the Reasoner."""

    files = _iter_projector_safetensor_files(Path(projector_path).expanduser())
    source: dict[str, torch.Tensor] = {}
    for file_path in files:
        source.update(safetensors.torch.load_file(str(file_path), device="cpu"))
    allowed = ("language_model.visual.merger.", "language_model.visual.deepstack_merger_list.")
    source = {
        key: value.to(dtype=torch.bfloat16) if torch.is_floating_point(value) else value
        for key, value in source.items()
        if key.startswith(allowed)
    }
    if not source:
        raise ValueError(f"Reasoner projector artifact contains no visual merger tensors: {projector_path}")
    return _load_reasoner_partial_weights(
        model,
        source=source,
        artifact_path=projector_path,
        label="projector",
    )


# Retain private compatibility aliases for the existing focused unit tests and
# any out-of-tree diagnostics. The public serving mode and launcher no longer
# expose or create an overlay checkpoint.
_load_reasoner_lora_overlay = _load_reasoner_lora_weights
_load_reasoner_projector_overlay = _load_reasoner_projector_weights


def _model_lora_runtime_context(model: Any, *, enabled: bool):
    """Return a no-op context for lightweight/fake models used by tests."""

    network = getattr(model, "net", model)
    if not isinstance(network, torch.nn.Module):
        return nullcontext()
    return lora_runtime_context(network, enabled=enabled)


def _validate_checkpoint(checkpoint_path: str, *, allow_dcp_checkpoint: bool) -> None:
    if checkpoint_path in OmniSetupOverrides.CHECKPOINTS:
        return
    if "://" in checkpoint_path:
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated local safetensors checkpoint directory, not a DCP path. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )

    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    # A training iteration is commonly passed as ``.../iter_N`` while its
    # state-dict files live below ``iter_N/model``.  Normalize that layout
    # before checking the artifact type; otherwise a valid local DCP is
    # incorrectly rejected as an invalid safetensors directory.
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    has_dcp_metadata = (checkpoint_dir / ".metadata").is_file()
    has_dcp_shards = any(checkpoint_dir.glob("*.distcp"))
    if has_dcp_metadata or has_dcp_shards:
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated safetensors checkpoint, but found a DCP checkpoint. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )
    has_config = (checkpoint_dir / "config.json").exists()
    has_consolidated_safetensors = any(checkpoint_dir.glob("*.safetensors"))
    has_diffusers_safetensors_index = (checkpoint_dir / "model.safetensors.index.json").exists()
    if (
        not checkpoint_dir.is_dir()
        or not has_config
        or not (has_consolidated_safetensors or has_diffusers_safetensors_index)
    ):
        raise ValueError(f"Invalid safetensors checkpoint directory: {checkpoint_dir}")


def _ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _ensure_2d_float_array(value: Any, key: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{key!r} must have shape [T,D] or [D], got {array.shape}")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{key!r} must have width {width}, got {array.shape[-1]}")
    return np.ascontiguousarray(array)


def _ensure_gripper_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"'observation/gripper_position' must have shape [T,1], [T], or scalar, got {array.shape}")
    return np.ascontiguousarray(array)


def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)  # [1,3,H2,W2]
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [H2,W2,3]


def _compose_roboarena_views(obs: dict[str, Any]) -> np.ndarray | None:
    required_keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    if not all(key in obs for key in required_keys):
        return None
    wrist = _ensure_rgb_uint8_image(obs["observation/wrist_image_left"], "observation/wrist_image_left")
    left_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_1_left"], "observation/exterior_image_1_left")
    right_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_2_left"], "observation/exterior_image_2_left")
    half_h, half_w = wrist.shape[0] // 2, wrist.shape[1] // 2
    left = _resize_rgb_uint8(left_raw, (half_h, half_w))
    right = _resize_rgb_uint8(right_raw, (half_h, half_w))
    return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)


def _extract_roboarena_separate_views(obs: dict[str, Any]) -> list[np.ndarray] | None:
    """Return physical LEFT, RIGHT, WRIST images without concat-view downscale."""
    required_keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    if not all(key in obs for key in required_keys):
        return None
    # The RoboLab camera preset exposes physical left/right separately.  Keep
    # their native pixels; only the wrist orientation is transformed later.
    left = _ensure_rgb_uint8_image(obs["observation/exterior_image_1_left"], "observation/exterior_image_1_left")
    right = _ensure_rgb_uint8_image(obs["observation/exterior_image_2_left"], "observation/exterior_image_2_left")
    wrist = _ensure_rgb_uint8_image(obs["observation/wrist_image_left"], "observation/wrist_image_left")
    return [left, right, wrist]


def _extract_observation_image(obs: dict[str, Any]) -> np.ndarray:
    if "observation/image" in obs:
        return _ensure_rgb_uint8_image(obs["observation/image"], "observation/image")
    image = _compose_roboarena_views(obs)
    if image is not None:
        return image
    raise ValueError("Observation must contain 'observation/image' or RoBoArena wrist/exterior image keys")


def _build_robolab_exterior_plucker_condition(height: int, width: int) -> torch.Tensor:
    """Build the fixed RoboLab physical-left/right raymap for a concat observation."""
    bottom_height = height // 3
    left_width = width // 2
    right_width = width - left_width
    source_width, source_height = _ROBOLAB_EXTERNAL_SOURCE_SIZE
    left_intrinsic = scaled_camera_matrix(
        _ROBOLAB_EXTERNAL_CAMERA_MATRIX,
        source_width=source_width,
        source_height=source_height,
        target_width=left_width,
        target_height=bottom_height,
    )
    right_intrinsic = scaled_camera_matrix(
        _ROBOLAB_EXTERNAL_CAMERA_MATRIX,
        source_width=source_width,
        source_height=source_height,
        target_width=right_width,
        target_height=bottom_height,
    )
    left_pose = opengl_camera_pose_to_opencv_matrix(*_ROBOLAB_LEFT_CAMERA_POSE)
    right_pose = opengl_camera_pose_to_opencv_matrix(*_ROBOLAB_RIGHT_CAMERA_POSE)
    return compose_exterior_plucker_condition(
        left_pose,
        left_intrinsic,
        right_pose,
        right_intrinsic,
        height=height,
        width=width,
    )


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]  # value: [...], batch item: [1,...]
        else:
            data_batch[key] = [value]
    return data_batch


def _load_openpi_websocket_policy_server() -> type[Any]:
    try:
        from openpi_server.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        try:
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboLab WebSocket serving uses OpenPI's WebsocketPolicyServer. Install it with "
                "`uv sync --all-extras --group=cu130-train --group=policy-server`, "
                "or install the full Physical-Intelligence/openpi package."
            ) from exc
    return WebsocketPolicyServer


class _DummyDataset(torch.utils.data.IterableDataset):
    def __iter__(self) -> Any:
        return iter(())


@dataclass(frozen=True)
class RobolabPolicyConfig:
    checkpoint_path: str
    domain_name: str
    decode_video: bool
    seed: int
    deterministic_seed: bool
    guidance: float
    num_steps: int
    shift: float
    conditioning_fps: float
    resolution: str | None
    action_chunk_size: int
    action_dim: int
    decode_subtask: bool = False
    subtask_max_new_tokens: int = 128
    # Opt-in ordinary VLM Reasoner -> WAM cascade.  This is deliberately
    # separate from ``decode_subtask``: the latter is the mid-soft auxiliary
    # decoder and is not present in Reasoner-WAM checkpoints.
    reasoner_wam: bool = False
    # ``shared`` is the historical single-checkpoint cascade. ``mode2`` keeps
    # two complete models resident. ``adapter`` loads standalone Reasoner LoRA
    # and projector files onto one native WAM model at runtime: LoRA is enabled
    # only for the Reasoner forward and disabled for the WAM action forward.
    reasoner_wam_mode: Literal["shared", "mode2", "adapter"] = "shared"
    reasoner_max_new_tokens: int = 128
    reasoner_do_sample: bool = False
    reasoner_observe_only: bool = False
    reasoner_include_preview_action: bool = False
    reasoner_open_gripper_move_augment: bool = False
    reasoner_require_think: bool = False
    reasoner_separate_views: bool = False
    reasoner_three_group_history: bool = False
    reasoner_image_long_edge: int = 448
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    image_width: int = _DEFAULT_IMAGE_WIDTH
    action_space: ActionSpace = "joint_pos"
    use_state: bool = True
    history_length: int = 1
    wrist_rotate_180: bool = False
    video_sample_stride: int = 1
    action_fps: float | None = None
    include_vit_first_frame: bool = False
    exterior_plucker_condition_enabled: bool = False
    camera_guidance: float | None = None
    camera_cfg_mode: Literal["camera", "both", "separate"] | None = None
    include_execution_success: bool = False


class RobolabServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str = _DEFAULT_DROID_POLICY_CHECKPOINT
    """Consolidated local safetensors checkpoint directory, registered checkpoint name, or DCP path with --allow-dcp-checkpoint."""
    hf_revision: str = _DEFAULT_HF_REVISION
    """Hugging Face revision used when --checkpoint-path is a supported public RoboLab policy repository."""
    allow_dcp_checkpoint: bool = False
    """If set, allow direct DCP/S3 checkpoint loading instead of requiring a consolidated safetensors export."""
    experiment: str | None = None
    """Experiment name for DCP checkpoints using module configs, e.g. droid_lerobot_8b_policy."""
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra experiment overrides forwarded to OmniSetup for DCP checkpoint loading."""
    credential_path: str | None = None
    """Optional checkpoint object-store credential path for DCP/S3 loading."""

    port: int = 8000
    """WebSocket port to bind."""
    host: str = "0.0.0.0"
    """WebSocket host to bind."""
    domain_name: str = "droid_lerobot"
    """Action domain name passed to get_domain_id()."""
    decode_video: bool = False
    """If set, decode and return the predicted rollout video as a uint8 NumPy array."""
    decode_subtask: bool = False
    """If set, autoregressively decode a subtask from the checkpoint's soft-plan P states."""
    subtask_max_new_tokens: int = 128
    """Maximum tokens generated by the optional subtask decoder."""
    reasoner_wam: bool = False
    """If set, first decode a VLM subgoal and use it as the WAM caption."""
    reasoner_then_wam: bool = False
    """Backward-compatible alias for ``--reasoner-wam``."""
    reasoner_wam_mode: Literal["shared", "mode2", "adapter"] = "shared"
    """Reasoner-WAM routing: shared legacy model, two-model mode2, or runtime standalone adapters."""
    reasoner_checkpoint_path: str | None = None
    """Mode2 Reasoner checkpoint (usually the exported LoRA checkpoint)."""
    reasoner_lora_path: str | None = None
    """Standalone Reasoner LoRA safetensors file or its artifact directory."""
    reasoner_projector_path: str | None = None
    """Standalone trained Reasoner visual-projector file or its artifact directory."""
    wam_checkpoint_path: str | None = None
    """Standalone WAM checkpoint. Required and distinct from the Reasoner path in mode2."""
    reasoner_max_new_tokens: int = 128
    """Maximum tokens generated by the opt-in Reasoner-WAM cascade."""
    reasoner_do_sample: bool = False
    reasoner_observe_only: bool = False
    """Use sampling for the opt-in Reasoner-WAM text decode (default: greedy)."""
    reasoner_include_preview_action: bool = False
    """Condition three-image Reasoner requests on the previous chunk's predicted subtask."""
    reasoner_open_gripper_move_augment: bool = False
    """Rewrite ``Move the open gripper`` in the Reasoner caption to an explicit open-then-move command."""
    reasoner_require_think: bool = False
    """Ask the Reasoner to emit ``<think>...</think>`` before its final executable subtask."""
    reasoner_separate_views: bool = False
    """Use separate LEFT/RIGHT/WRIST images with PREV/CURRENT groups."""
    reasoner_three_group_history: bool = False
    """In separate-view mode, receive t-16, t-8, and current groups rather than t-16/current."""
    reasoner_image_long_edge: int = 448
    """Long-edge resize applied to each separate Reasoner camera image."""
    format_prompt_as_json: bool | None = None
    """Force the WAM action prompt to structured JSON; default restores the WAM training config value."""
    include_execution_success: bool | None = None
    """Condition each JSON action prompt on ``execution_success: true``; default restores the WAM training config value."""
    guardrails: bool = True
    """Enable Cosmos inference safety guardrails."""
    offload_guardrail_models: bool = False
    """Offload guardrail models to CPU when guardrails are enabled."""

    output_dir: Path | None = None
    """Output directory for OmniInference. Defaults to /tmp/cosmos3_action_server/robolab."""
    use_torch_compile: bool = True
    """Compile model regions with torch.compile; disable for short smoke tests or compiler troubleshooting."""
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by OmniInference."""

    seed: int = 0
    """Base generation seed used to initialize the request RNG."""
    deterministic_seed: bool = False
    """Use the same seed for every request. If false, advance a NumPy RNG seeded by --seed."""
    guidance: float = 3.0
    """Text guidance scale for denoising."""
    camera_guidance: float | None = None
    """Independent camera guidance scale; enables three-forward CFG when provided."""
    camera_cfg_mode: Literal["camera", "both", "separate"] | None = None
    """Camera CFG mode: camera (C→TC), both (U→TC), or separate (U→T→TC)."""
    num_steps: int = 4
    """Number of denoising steps."""
    shift: float = 5.0
    """UniPC sampler shift."""

    resolution: str | None = "480"
    """Action transform resolution. The default matches the released DROID RoboLab policy."""
    conditioning_fps: float | None = None
    """Conditioning video FPS. If omitted, derived from action_fps / video_sample_stride."""
    action_chunk_size: int | None = _DEFAULT_ACTION_CHUNK_SIZE
    """Number of action steps to predict. The default matches the released DROID RoboLab policy."""
    action_dim: int | None = _DEFAULT_ACTION_DIM
    """Raw action dimension. The default matches the released DROID RoboLab policy."""
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    """Input observation image height. The default matches the released DROID RoboLab policy."""
    image_width: int = _DEFAULT_IMAGE_WIDTH
    """Input observation image width. The default matches the released DROID RoboLab policy."""
    action_space: ActionSpace = "joint_pos"
    """RoboLab action representation to serve."""
    use_state: bool = True
    """Whether the first action row contains the current state."""
    history_length: int = 1
    """State/history action rows to trim from the generated action output."""
    wrist_rotate_180: bool = False
    """Rotate the wrist camera image by 180 degrees before composing the concat view."""
    video_sample_stride: int | None = None
    """Video temporal stride used at training time. 2 serves the viddown2 policy with 17 model video frames."""
    action_fps: float | None = None
    """Action control frequency in Hz. When different from conditioning FPS, the prompt includes this metadata."""
    include_vit_first_frame: bool = False
    """Tokenize the first observation frame with the Qwen3-VL/Cosmos3-Nano vision processor."""
    vit_processor_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    """Local path or model name used to construct the first-frame vision processor."""
    vit_processor_config_variant: str = "hf"
    """Processor config variant used by the Cosmos tokenizer builder."""
    vit_processor_cache_dir: str | None = None
    """Optional processor cache directory."""
    vit_first_frame_max_pixels: int | None = None
    """Optional maximum first-frame pixel budget passed to the vision processor."""
    vit_first_frame_min_pixels: int | None = None
    """Optional minimum first-frame pixel budget passed to the vision processor."""
    exterior_plucker_condition: bool | None = None
    """Override camera-ray conditioning. By default it follows the exported model config."""

    online_dpo_rank: int = 0
    """Rank of this server in the separate online-DPO gradient group."""
    online_dpo_world_size: int = 1
    """Number of resident Reasoner/WAM servers participating in online DPO."""
    online_dpo_master_addr: str = "127.0.0.1"
    """Address used only by the separate online-DPO gradient group."""
    online_dpo_master_port: int = 19400
    """Port used only by the separate online-DPO gradient group."""
    online_dpo_reference_lora_path: str | None = None
    """Fixed reference adapter for DPO; separate from a resumed actor adapter."""


class RobolabPolicyService:
    def __init__(self, args: RobolabServerArgs) -> None:
        # A PPO sample is created by the same Reasoner model that serves WAM.
        # Keep only one pending sample per actor request; no second model
        # replica is ever constructed for online training.
        self._ppo_pending: dict[str, dict[str, Any]] = {}
        self._ppo_optimizer: torch.optim.Optimizer | None = None
        self._spin_optimizer: torch.optim.Optimizer | None = None
        self._spin_reference_lora: dict[str, torch.Tensor] | None = None
        self._spin_lora_metadata: dict[str, Any] | None = None
        # Deferred online-DPO accumulation is used by the multi-simulation
        # runner.  The resident model is updated only by the explicit
        # finalize control after every local case has reported.
        self._spin_accumulation_steps: int | None = None
        self._spin_accumulated_cases = 0
        self._spin_accumulated_grad: dict[str, torch.Tensor] = {}
        self._spin_accumulated_losses: list[float] = []
        self._online_dpo_group = None
        self._online_dpo_rank = int(args.online_dpo_rank)
        self._online_dpo_world_size = int(args.online_dpo_world_size)
        if self._online_dpo_world_size <= 0 or not 0 <= self._online_dpo_rank < self._online_dpo_world_size:
            raise ValueError(
                "online DPO rank/world size must satisfy "
                f"0 <= rank < world_size, got rank={self._online_dpo_rank} world_size={self._online_dpo_world_size}"
            )
        reasoner_wam_enabled = bool(args.reasoner_wam or args.reasoner_then_wam)
        if args.reasoner_include_preview_action and not reasoner_wam_enabled:
            raise ValueError("--reasoner-include-preview-action requires --reasoner-wam")
        if args.reasoner_wam_mode == "mode2" and not reasoner_wam_enabled:
            raise ValueError("--reasoner-wam-mode=mode2 requires --reasoner-wam")
        if args.reasoner_wam_mode == "adapter" and not reasoner_wam_enabled:
            raise ValueError("--reasoner-wam-mode=adapter requires --reasoner-wam")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")
        mode2_enabled = reasoner_wam_enabled and args.reasoner_wam_mode == "mode2"
        adapter_enabled = reasoner_wam_enabled and args.reasoner_wam_mode == "adapter"
        if adapter_enabled:
            if not args.reasoner_lora_path:
                raise ValueError("--reasoner-wam-mode=adapter requires --reasoner-lora-path")
            lora_root = Path(args.reasoner_lora_path).expanduser()
            lora_sidecar = lora_root / "reasoner_lora.json" if lora_root.is_dir() else lora_root.with_suffix(".json")
            lora_metadata = json.loads(lora_sidecar.read_text(encoding="utf-8"))
            self._spin_lora_metadata = dict(lora_metadata)
            if lora_metadata.get("format") != "cosmos_custom_lora_v1":
                raise ValueError(f"Unsupported Reasoner LoRA metadata: {lora_sidecar}")
            lora_rank = int(lora_metadata["lora_rank"])
            lora_alpha = int(lora_metadata["lora_alpha"])
            lora_targets = lora_metadata["lora_target_modules"]
            if lora_rank <= 0 or lora_alpha <= 0 or not isinstance(lora_targets, list) or not lora_targets:
                raise ValueError(f"Invalid Reasoner LoRA metadata: {lora_sidecar}")
            adapter_overrides = list(args.experiment_overrides)
            adapter_overrides.extend(
                [
                    "model.config.lora_enabled=true",
                    f"model.config.lora_rank={lora_rank}",
                    f"model.config.lora_alpha={lora_alpha}",
                    f"model.config.lora_target_modules={','.join(str(x) for x in lora_targets)}",
                    "model.config.lora_freeze_non_lora=false",
                    "model.config.lora_soft_plan_only=false",
                    "model.config.vlm_config.model_instance.config.include_visual=true",
                    "model.config.precision=bfloat16",
                ]
            )
            args = args.model_copy(update={"experiment_overrides": adapter_overrides})
        if mode2_enabled:
            # Mode2 intentionally makes the two roles explicit.  For
            # backwards-friendly shell invocations ``--checkpoint-path`` is
            # the WAM base checkpoint, while ``--reasoner-checkpoint-path``
            # names the exported Reasoner (normally the 15000 LoRA export).
            wam_source = args.wam_checkpoint_path or args.checkpoint_path
            reasoner_source = args.reasoner_checkpoint_path or args.checkpoint_path
            resolved_wam_path = _resolve_latest_dcp_checkpoint(
                _resolve_checkpoint_path(wam_source, hf_revision=args.hf_revision)
            )
            resolved_reasoner_path = _resolve_checkpoint_path(reasoner_source, hf_revision=args.hf_revision)
            # Mode2's contract is intentionally strict: the WAM must be an
            # independent no-LoRA checkpoint.  Silently reusing the Reasoner
            # path would make the apparent ``wam_lora_runtime=False`` switch
            # insufficient under compiled graphs and could feed a LoRA-trained
            # generation model to the WAM role.
            _ensure_distinct_mode2_checkpoints(resolved_reasoner_path, resolved_wam_path)
            args = args.model_copy(
                update={
                    "checkpoint_path": resolved_reasoner_path,
                    "wam_checkpoint_path": resolved_wam_path,
                    "reasoner_checkpoint_path": resolved_reasoner_path,
                }
            )
            _validate_checkpoint(args.checkpoint_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
            _validate_checkpoint(resolved_wam_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
        else:
            resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, hf_revision=args.hf_revision)
            args = args.model_copy(update={"checkpoint_path": resolved_checkpoint_path})
            _validate_checkpoint(args.checkpoint_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
        maybe_init_distributed()
        if self._online_dpo_world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("online DPO gradient synchronization requires torch.distributed")
            # The serving process already owns a singleton world-size-1
            # process group for model construction.  Build a second private
            # NCCL group so four complete model replicas can synchronize only
            # their LoRA gradients without turning model loading into FSDP.
            import torch.distributed.distributed_c10d as distributed_c10d

            store = torch.distributed.TCPStore(
                args.online_dpo_master_addr,
                int(args.online_dpo_master_port),
                self._online_dpo_world_size,
                is_master=self._online_dpo_rank == 0,
                timeout=timedelta(minutes=30),
            )
            self._online_dpo_group, _ = distributed_c10d._new_process_group_helper(
                self._online_dpo_world_size,
                self._online_dpo_rank,
                list(range(self._online_dpo_world_size)),
                "nccl",
                store,
                group_name=f"robolab_online_dpo_{args.online_dpo_master_port}",
                timeout=timedelta(minutes=30),
                pg_tag=f"robolab_online_dpo_{args.online_dpo_master_port}",
                group_desc="RoboLab online DPO LoRA gradients",
            )
            log.info(
                f"[robolab-policy-server] online DPO gradient group ready "
                f"rank={self._online_dpo_rank}/{self._online_dpo_world_size}"
            )

        # The Reasoner path is an image-conditioned autoregressive VLM call.
        # Its runtime-dependent Qwen3-VL mRoPE/image-grid shapes are not safe
        # to capture in the language compile region (the first real request
        # can otherwise fail inside the rotary CUBLAS matmul).  Mode2 disables
        # compile for this primary model only; the standalone WAM child keeps
        # the normal compiled path below.  Legacy/shared serving is unchanged.
        setup_args = self._build_setup_args(args, disable_compile=mode2_enabled or adapter_enabled)
        log.info(
            f"[robolab-policy-server] loading model: checkpoint_path={setup_args.checkpoint_path!r} "
            f"config_file={setup_args.config_file!r} experiment={setup_args.experiment!r}"
        )
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args

        # ``mode2`` uses an independent WAM model.  Keeping the historical
        # ``self.model`` alias pointed at the primary model means every
        # non-mode2 caller follows exactly the old code path.  The explicit
        # separation check above guarantees that this child cannot silently
        # become the LoRA Reasoner model.
        self.wam_pipe: OmniInference | None = None
        self.wam_model = self.model
        self.wam_setup_args: OmniSetupArgs | None = None
        wam_training_config: Any | None = None
        if mode2_enabled:
            wam_path = str(args.wam_checkpoint_path)
            if Path(wam_path).absolute() != Path(args.checkpoint_path).absolute():
                # A DCP action experiment's module config normally names the
                # public Qwen repository.  The evaluation machines keep the
                # matching Cosmos3-Nano tokenizer locally and are often
                # offline; route only the mode2 WAM child through that local
                # path.  The exported Reasoner checkpoint already carries its
                # own tokenizer configuration, so the legacy/shared path is
                # untouched.
                wam_experiment_overrides = list(args.experiment_overrides)
                local_tokenizer = str(args.vit_processor_model_name or "").strip()
                if local_tokenizer and Path(local_tokenizer).is_dir():
                    tokenizer_override = f"model.config.vlm_config.tokenizer.pretrained_model_name={local_tokenizer}"
                    if not any(
                        item.startswith("model.config.vlm_config.tokenizer.pretrained_model_name=")
                        for item in wam_experiment_overrides
                    ):
                        wam_experiment_overrides.append(tokenizer_override)
                # The paired DCP launcher exports WAN_VAE_PATH.  Forward it
                # when present so a direct mode2 invocation does not depend on
                # a relative pretrained/tokenizers path.
                local_vae = os.environ.get("WAN_VAE_PATH", "").strip()
                if (
                    local_vae
                    and Path(local_vae).is_file()
                    and not any(
                        item.startswith("model.config.tokenizer.vae_path=") for item in wam_experiment_overrides
                    )
                ):
                    wam_experiment_overrides.append(f"model.config.tokenizer.vae_path={local_vae}")
                wam_args = args.model_copy(
                    update={
                        "checkpoint_path": wam_path,
                        "reasoner_wam": False,
                        "reasoner_then_wam": False,
                        "reasoner_wam_mode": "shared",
                        "reasoner_checkpoint_path": None,
                        "reasoner_lora_path": None,
                        "reasoner_projector_path": None,
                        # The standalone WAM checkpoint has no Qwen visual
                        # tower.  Keep its setup/compile path text-only; the
                        # primary Reasoner instance receives the ViT flag.
                        "include_vit_first_frame": False,
                        "experiment_overrides": wam_experiment_overrides,
                        "output_dir": (
                            (args.output_dir / "mode2_wam")
                            if args.output_dir is not None
                            else (_DEFAULT_ROBOLAB_OUTPUT_DIR / "mode2_wam")
                        ),
                    }
                )
                wam_setup_args = self._build_setup_args(wam_args)
                log.info(
                    f"[robolab-policy-server] mode2 loading WAM base checkpoint: "
                    f"checkpoint_path={wam_setup_args.checkpoint_path!r} "
                    f"config_file={wam_setup_args.config_file!r} experiment={wam_setup_args.experiment!r}"
                )
                self.wam_pipe = OmniInference.create(wam_setup_args)
                self.wam_model = self.wam_pipe.model
                self.wam_model.eval()
                assert isinstance(self.wam_pipe.setup_args, OmniSetupArgs)
                self.wam_setup_args = self.wam_pipe.setup_args
                # Prefer the resolved run config when the caller supplied a
                # training job root.  The iteration itself has no
                # checkpoint.json, so the generic loader would otherwise fall
                # back to the baseline experiment and silently lose WAM-only
                # transform flags.
                # Resolve the concrete iteration first.  ``wam_source`` may be
                # the outer output root (``.../outputs/<job-family>``), which
                # has no config.yaml of its own; the selected iteration's
                # ancestry always reaches the standalone WAM run config.
                wam_training_config = _load_nearest_local_training_yaml(wam_path)
                if wam_training_config is None:
                    wam_training_config = _load_training_config(self.wam_setup_args, wam_path)

        training_config = _load_training_config(self.setup_args, args.checkpoint_path)
        transform_config = wam_training_config or training_config
        if mode2_enabled or adapter_enabled:
            # The standalone WAM recipe intentionally has no ViT first-frame
            # tokenization.  Keep that transform for the action forward, and
            # build a separate Reasoner-only transform with the Qwen3-VL
            # processor requested by ``--include-vit-first-frame``.
            wam_transform_args = args.model_copy(update={"include_vit_first_frame": False})
            self._transform, inferred = self._build_transform(
                transform_config,
                wam_transform_args,
                model_override=self.wam_model,
            )
            self._reasoner_transform, _ = self._build_transform(
                training_config,
                args,
                model_override=self.model,
            )
        else:
            self._transform, inferred = self._build_transform(
                transform_config,
                args,
                model_override=self.wam_model,
            )
            self._reasoner_transform = self._transform
        if (mode2_enabled or adapter_enabled) and args.reasoner_lora_path:
            _load_reasoner_lora_weights(self.model, args.reasoner_lora_path)
        if (mode2_enabled or adapter_enabled) and args.reasoner_projector_path:
            _load_reasoner_projector_weights(self.model, args.reasoner_projector_path)
        if adapter_enabled:
            actor_lora = {
                name: parameter.detach().clone()
                for name, parameter in self.model.net.named_parameters()
                if "lora_" in name
            }
            if args.online_dpo_reference_lora_path:
                _load_reasoner_lora_weights(self.model, args.online_dpo_reference_lora_path)
            self._spin_reference_lora = {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.model.net.named_parameters()
                if "lora_" in name
            }
            if args.online_dpo_reference_lora_path:
                with torch.no_grad():
                    for name, parameter in self.model.net.named_parameters():
                        value = actor_lora.get(name)
                        if value is not None:
                            parameter.copy_(value)
            if not self._spin_reference_lora:
                raise RuntimeError("online DPO requires loaded Reasoner LoRA parameters")
        model_mid_soft = bool(
            getattr(self.model, "mid_soft", getattr(getattr(self.model, "config", None), "mid_soft", False))
        )
        transform_mid_soft = bool(getattr(self._transform, "mid_soft", False))
        if model_mid_soft != transform_mid_soft:
            raise ValueError(
                "Checkpoint/model and inference transform disagree on mid_soft: "
                f"model={model_mid_soft}, transform={transform_mid_soft}. "
                "The serving transform must be built from the checkpoint's training config."
            )
        if model_mid_soft and not args.include_vit_first_frame:
            raise ValueError(
                "This checkpoint has mid_soft=True and requires --include-vit-first-frame so the "
                "mid-soft prompt boundary and visual tokens are constructed during inference."
            )
        wam_model_mid_soft = bool(
            getattr(self.wam_model, "mid_soft", getattr(getattr(self.wam_model, "config", None), "mid_soft", False))
        )
        if reasoner_wam_enabled:
            if args.decode_subtask:
                raise ValueError(
                    "--reasoner-wam is incompatible with --decode-subtask. "
                    "Reasoner-WAM uses the ordinary VLM CE path; --decode-subtask is the mid-soft auxiliary decoder."
                )
            if not args.include_vit_first_frame:
                raise ValueError(
                    "--reasoner-wam requires --include-vit-first-frame for the visual Reasoner decode. "
                    "In mode2 the WAM transform remains text-only; shared mode uses the same visual observation "
                    "for both paths."
                )
            if model_mid_soft:
                raise ValueError(
                    "--reasoner-wam requires a Reasoner-WAM checkpoint with mid_soft=false; "
                    "use --decode-subtask for a mid-soft auxiliary decoder instead."
                )
            if mode2_enabled and wam_model_mid_soft:
                raise ValueError("Reasoner-WAM mode2 requires the WAM checkpoint to have mid_soft=false.")
            model_config = getattr(self.model, "config", None)
            reasoner_ce_enabled = getattr(self.model, "reasoner_ce_enabled", None)
            if reasoner_ce_enabled is None:
                reasoner_ce_enabled = _config_value(model_config, "reasoner_ce_enabled")
            two_forward_enabled = getattr(self.model, "reasoner_wam_two_forward", None)
            if two_forward_enabled is None:
                two_forward_enabled = _config_value(model_config, "reasoner_wam_two_forward")
            if (
                not adapter_enabled
                and not _config_bool(reasoner_ce_enabled)
                and not _config_bool(two_forward_enabled)
            ):
                raise ValueError(
                    "--reasoner-wam requires a checkpoint trained with reasoner_ce_enabled=true "
                    "(reasoner_wam_two_forward=true)."
                )
            if not callable(getattr(self.model, "generate_reasoner_text", None)):
                raise ValueError("--reasoner-wam requires a checkpoint/model exposing generate_reasoner_text().")
            if int(args.reasoner_max_new_tokens) <= 0:
                raise ValueError("--reasoner-max-new-tokens must be positive.")
            if mode2_enabled or adapter_enabled:
                reasoner_network = getattr(self.model, "net", self.model)
                adapter_count = sum(1 for _ in iter_lora_modules(reasoner_network))
                if adapter_count == 0:
                    raise ValueError(
                        f"Reasoner-WAM {args.reasoner_wam_mode} requires LoRA-injected Reasoner weights, but the supplied "
                        "Reasoner checkpoint exposes no lora_ modules. Pass the exported Reasoner checkpoint "
                        "(and its --reasoner-lora-path), not the standalone no-LoRA WAM checkpoint."
                    )
                log.info(
                    f"[robolab-policy-server] {args.reasoner_wam_mode} Reasoner LoRA modules={adapter_count}"
                )
        model_subtask_decoder_enabled = bool(getattr(self.model, "subtask_decoder_enabled", False))
        if args.decode_subtask and not model_subtask_decoder_enabled:
            raise ValueError(
                "--decode-subtask requires a checkpoint exported with --include-subtask-decoder "
                "and subtask_decoder.enabled=true."
            )
        if int(args.subtask_max_new_tokens) <= 0:
            raise ValueError("--subtask-max-new-tokens must be positive.")
        configured_subtask_max_tokens = int(
            getattr(self.model, "subtask_decoder_config", {}).get("max_tokens", args.subtask_max_new_tokens)
        )
        if args.decode_subtask and int(args.subtask_max_new_tokens) > configured_subtask_max_tokens:
            raise ValueError(
                f"--subtask-max-new-tokens={args.subtask_max_new_tokens} exceeds the checkpoint decoder "
                f"max_tokens={configured_subtask_max_tokens}."
            )
        video_sample_stride, conditioning_fps, action_fps = _resolve_policy_timing(args, inferred)
        model_plucker_enabled = bool(
            getattr(getattr(self.wam_model, "config", None), "exterior_plucker_condition_enabled", False)
        )
        exterior_plucker_condition_enabled = (
            model_plucker_enabled if args.exterior_plucker_condition is None else bool(args.exterior_plucker_condition)
        )
        if exterior_plucker_condition_enabled and not model_plucker_enabled:
            raise ValueError(
                "--exterior-plucker-condition cannot enable conditioning for a checkpoint whose model config "
                "has exterior_plucker_condition_enabled=False."
            )
        camera_cfg_mode = args.camera_cfg_mode
        if camera_cfg_mode is not None and not exterior_plucker_condition_enabled:
            raise ValueError("--camera-cfg-mode requires exterior Plucker conditioning to be enabled.")
        if args.camera_guidance is not None and not exterior_plucker_condition_enabled:
            raise ValueError("--camera-guidance requires exterior Plucker conditioning to be enabled.")
        if camera_cfg_mode in {"camera", "both"} and args.camera_guidance is not None:
            raise ValueError(
                f"--camera-guidance is only valid with --camera-cfg-mode separate, got {camera_cfg_mode!r}."
            )
        if camera_cfg_mode == "separate" and args.camera_guidance is None:
            # Keep the explicit mode convenient for direct server invocation;
            # the launcher normally supplies its configured default (1.0).
            camera_guidance = 1.0
        else:
            camera_guidance = args.camera_guidance
        if camera_cfg_mode is None and exterior_plucker_condition_enabled:
            # Legacy calls with camera_guidance use separate CFG; otherwise
            # they use the historical joint U→TC CFG path.
            camera_cfg_mode = "separate" if camera_guidance is not None else "both"
        if exterior_plucker_condition_enabled:
            video_resize = getattr(self._transform, "video_resize", None)
            pad_keys = getattr(video_resize, "pad_keys", None)
            if not isinstance(pad_keys, list):
                raise TypeError("Plucker-conditioned serving requires an ActionTransformPipeline-compatible transform.")
            if "plucker_condition" not in pad_keys:
                pad_keys.append("plucker_condition")

        # The released server keeps resolution='480' as its legacy default.
        # For the new opt-in paired Reasoner-WAM recipe, prefer the WAM
        # training-config resolution when the caller did not explicitly pass a
        # different value.  This branch is unreachable for all legacy modes.
        effective_resolution = args.resolution
        if reasoner_wam_enabled and args.resolution == "480" and inferred.get("resolution") is not None:
            effective_resolution = str(inferred["resolution"])

        self.cfg = RobolabPolicyConfig(
            checkpoint_path=self.setup_args.checkpoint_path,
            domain_name=args.domain_name,
            decode_video=bool(args.decode_video),
            decode_subtask=bool(args.decode_subtask),
            subtask_max_new_tokens=int(args.subtask_max_new_tokens),
            reasoner_wam=reasoner_wam_enabled,
            reasoner_wam_mode=args.reasoner_wam_mode,
            reasoner_max_new_tokens=int(args.reasoner_max_new_tokens),
            reasoner_do_sample=bool(args.reasoner_do_sample),
            reasoner_observe_only=bool(args.reasoner_observe_only),
            reasoner_include_preview_action=bool(args.reasoner_include_preview_action),
            reasoner_open_gripper_move_augment=bool(args.reasoner_open_gripper_move_augment),
            reasoner_require_think=bool(args.reasoner_require_think),
            reasoner_separate_views=bool(args.reasoner_separate_views),
            reasoner_three_group_history=bool(args.reasoner_three_group_history),
            reasoner_image_long_edge=int(args.reasoner_image_long_edge),
            seed=int(args.seed),
            deterministic_seed=bool(args.deterministic_seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            shift=float(args.shift),
            conditioning_fps=conditioning_fps,
            resolution=effective_resolution or inferred.get("resolution"),
            action_chunk_size=int(
                args.action_chunk_size or inferred.get("action_chunk_size") or _DEFAULT_ACTION_CHUNK_SIZE
            ),
            action_dim=int(args.action_dim or (8 if args.action_space == "joint_pos" else 10)),
            image_height=int(args.image_height),
            image_width=int(args.image_width),
            action_space=args.action_space,
            use_state=bool(args.use_state),
            history_length=int(args.history_length),
            wrist_rotate_180=bool(args.wrist_rotate_180),
            video_sample_stride=video_sample_stride,
            action_fps=action_fps,
            # ``cfg.include_vit_first_frame`` controls the WAM action sample;
            # mode2's ViT processor lives on ``_reasoner_transform`` instead.
            include_vit_first_frame=bool(args.include_vit_first_frame and not (mode2_enabled or adapter_enabled)),
            exterior_plucker_condition_enabled=exterior_plucker_condition_enabled,
            camera_guidance=None if camera_guidance is None else float(camera_guidance),
            camera_cfg_mode=camera_cfg_mode,
            include_execution_success=bool(inferred.get("include_execution_success", False)),
        )
        if self.cfg.history_length < (1 if self.cfg.use_state else 0):
            raise ValueError("--history-length must be >= 1 when --use-state is true")
        if self.cfg.image_height <= 0 or self.cfg.image_width <= 0:
            raise ValueError("--image-height and --image-width must be positive")

        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.include_execution_success and not getattr(self._transform, "prompt_json_formatter", None):
            log.warning(
                "[robolab-policy-server] include_execution_success=True but the serving transform has no JSON "
                "prompt formatter; execution_success will not reach the prompt. Enable format_prompt_as_json."
            )
        log.info(
            f"[robolab-policy-server] ready domain={self.cfg.domain_name!r} resolution={self.cfg.resolution!r} "
            f"action_space={self.cfg.action_space} action_dim={self.cfg.action_dim} "
            f"chunk={self.cfg.action_chunk_size} history={self.cfg.history_length} use_state={self.cfg.use_state} "
            f"image={self.cfg.image_height}x{self.cfg.image_width} conditioning_fps={self.cfg.conditioning_fps} "
            f"video_stride={self.cfg.video_sample_stride} action_fps={self.cfg.action_fps} "
            f"wrist_rotate_180={self.cfg.wrist_rotate_180} "
            f"vit_first_frame={self.cfg.include_vit_first_frame} "
            f"mid_soft={transform_mid_soft} "
            f"decode_subtask={self.cfg.decode_subtask} "
            f"subtask_max_new_tokens={self.cfg.subtask_max_new_tokens} "
            f"reasoner_wam={self.cfg.reasoner_wam} "
            f"reasoner_wam_mode={self.cfg.reasoner_wam_mode} "
            f"reasoner_max_new_tokens={self.cfg.reasoner_max_new_tokens} "
            f"reasoner_do_sample={self.cfg.reasoner_do_sample} "
            f"reasoner_include_preview_action={self.cfg.reasoner_include_preview_action} "
            f"reasoner_open_gripper_move_augment={self.cfg.reasoner_open_gripper_move_augment} "
            f"reasoner_require_think={self.cfg.reasoner_require_think} "
            f"reasoner_separate_views={self.cfg.reasoner_separate_views} "
            f"reasoner_three_group_history={self.cfg.reasoner_three_group_history} "
            f"format_prompt_as_json={bool(getattr(self._transform, 'prompt_json_formatter', None))} "
            f"include_execution_success={self.cfg.include_execution_success} "
            f"exterior_plucker={self.cfg.exterior_plucker_condition_enabled} "
            f"guidance={self.cfg.guidance} camera_cfg_mode={self.cfg.camera_cfg_mode} "
            f"camera_guidance={self.cfg.camera_guidance} "
            f"num_steps={self.cfg.num_steps} shift={self.cfg.shift} "
            f"seed={self.cfg.seed} deterministic_seed={self.cfg.deterministic_seed}"
        )
        if mode2_enabled:
            log.info(
                f"[robolab-policy-server] mode2 roles: reasoner_checkpoint={args.checkpoint_path!r} "
                f"wam_checkpoint={args.wam_checkpoint_path!r} reasoner_lora_enabled=True "
                "wam_lora_runtime=False"
            )
        elif adapter_enabled:
            log.info(
                f"[robolab-policy-server] adapter roles: wam_checkpoint={args.checkpoint_path!r} "
                "reasoner_lora_runtime=True wam_lora_runtime=False "
                f"lora={args.reasoner_lora_path!r} projector={args.reasoner_projector_path!r} dtype=bfloat16"
            )

    def _build_setup_args(self, args: RobolabServerArgs, *, disable_compile: bool = False) -> OmniSetupArgs:
        setup_overrides: dict[str, Any] = {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir or _DEFAULT_ROBOLAB_OUTPUT_DIR,
            "sampler": args.sampler,
            "guardrails": args.guardrails,
            "offload_guardrail_models": args.offload_guardrail_models,
            "use_torch_compile": args.use_torch_compile,
        }
        if disable_compile:
            setup_overrides["use_torch_compile"] = False
        elif args.include_vit_first_frame:
            # Qwen3-VL's visual tower receives a runtime-dependent image_grid_thw.
            # Keep visual preprocessing/encoding eager while retaining compiled
            # language blocks. This avoids compiling the runtime-dependent
            # image_grid_thw preprocessing path.
            setup_overrides["compiled_region"] = "language"
        if args.experiment is not None:
            setup_overrides["experiment"] = args.experiment
        if args.experiment_overrides:
            setup_overrides["experiment_overrides"] = list(args.experiment_overrides)
        if args.credential_path is not None:
            setup_overrides["credential_path"] = args.credential_path
        overrides = OmniSetupOverrides.model_validate(setup_overrides)
        setup_args = overrides.build_setup()
        log.info(
            f"[robolab-policy-server] compile enabled={setup_args.use_torch_compile} "
            f"region={setup_args.compiled_region} dynamic={setup_args.compile_dynamic}"
        )
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _build_transform(
        self,
        training_config: Any | None,
        args: RobolabServerArgs,
        *,
        model_override: Any | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        inferred: dict[str, Any] = {}
        transform_model = model_override if model_override is not None else self.model
        model_max_action_dim = getattr(getattr(transform_model, "config", None), "max_action_dim", None)
        max_action_dim = int(model_max_action_dim) if isinstance(model_max_action_dim, int) else 64
        model_mid_soft = bool(
            getattr(transform_model, "mid_soft", getattr(getattr(transform_model, "config", None), "mid_soft", False))
        )

        try:
            dataset_config = (
                training_config.dataloader_train.dataloaders.action_data.dataloader.dataset
                if training_config is not None
                else None
            )
            dataset_entry = dataset_config.list_of_datasets[0] if dataset_config is not None else None
            action_dataset_config = dataset_entry.dataset if dataset_entry is not None else None
        except (AttributeError, IndexError, TypeError):
            dataset_config = None
            dataset_entry = None
            action_dataset_config = None

        if dataset_config is None or dataset_entry is None:
            action_dataset_config = _find_rank_partitioned_action_dataset_config(training_config)
            if action_dataset_config is not None:
                log.info("[robolab-policy-server] using rank-partitioned action dataset config for inference transform")
                return _build_action_transform_from_config(
                    action_dataset_config,
                    args=args,
                    max_action_dim=max_action_dim,
                    model_mid_soft=model_mid_soft,
                )
            log.warning(
                "[robolab-policy-server] no training action dataset config found; using default ActionTransformPipeline"
            )
            inferred["include_execution_success"] = bool(args.include_execution_success)
            return (
                ActionTransformPipeline(
                    max_action_dim=max_action_dim,
                    cfg_dropout_rate=0.0,
                    format_prompt_as_json=bool(args.format_prompt_as_json),
                    include_vit_first_frame=bool(args.include_vit_first_frame),
                    vit_processor_model_name=args.vit_processor_model_name,
                    vit_processor_config_variant=args.vit_processor_config_variant,
                    vit_processor_cache_dir=args.vit_processor_cache_dir,
                    vit_first_frame_max_pixels=args.vit_first_frame_max_pixels,
                    vit_first_frame_min_pixels=args.vit_first_frame_min_pixels,
                    mid_soft=model_mid_soft,
                ),
                inferred,
            )

        if action_dataset_config is not None:
            if args.include_vit_first_frame:
                action_dataset_config.include_vit_first_frame = True
                action_dataset_config.vit_processor_model_name = args.vit_processor_model_name
                action_dataset_config.vit_processor_config_variant = args.vit_processor_config_variant
                action_dataset_config.vit_processor_cache_dir = args.vit_processor_cache_dir
                action_dataset_config.vit_first_frame_max_pixels = args.vit_first_frame_max_pixels
                action_dataset_config.vit_first_frame_min_pixels = args.vit_first_frame_min_pixels
            chunk_length = getattr(action_dataset_config, "chunk_length", None)
            if isinstance(chunk_length, int):
                inferred["action_chunk_size"] = chunk_length
            fps = _maybe_float(getattr(action_dataset_config, "fps", None))
            video_sample_stride = _maybe_int(getattr(action_dataset_config, "video_sample_stride", None))
            if video_sample_stride is None:
                video_sample_stride = 1
            if video_sample_stride < 1:
                raise ValueError(f"training config video_sample_stride must be >= 1, got {video_sample_stride}")
            inferred["video_sample_stride"] = video_sample_stride
            if fps is not None:
                inferred["action_fps"] = fps
                inferred["conditioning_fps"] = fps / video_sample_stride
            persisted_execution_success = _config_bool(
                _config_value(action_dataset_config, "include_execution_success", False)
            )
            inferred["include_execution_success"] = (
                persisted_execution_success
                if args.include_execution_success is None
                else bool(args.include_execution_success)
            )

        if args.resolution is not None:
            dataset_entry.resolution = args.resolution
        else:
            inferred_resolution = dataset_entry.resolution
            if inferred_resolution is None:
                inferred_resolution = dataset_config.resolution
            if inferred_resolution is not None:
                inferred["resolution"] = str(inferred_resolution)

        if dataset_config.cfg_dropout_rate != 0.0:
            dataset_config.cfg_dropout_rate = 0.0
        if getattr(dataset_config, "camera_cfg_dropout_rate", 0.0) != 0.0:
            dataset_config.camera_cfg_dropout_rate = 0.0

        dataset_entry.dataset = {"_target_": f"{__name__}._DummyDataset"}

        wrapped_dataset = instantiate(dataset_config)
        return wrapped_dataset.transform, inferred

    def _next_seed(self) -> int:
        if self.cfg.deterministic_seed:
            return self.cfg.seed
        return int(self._rng.integers(0, 2**31))

    def _build_vit_text_conditioning(self, sample: dict[str, Any]) -> _VitTextConditioning:
        transform = self._transform
        tokenize_with_vit = getattr(transform, "_tokenize_text_with_vit_first_frame", None)
        if not callable(tokenize_with_vit):
            raise TypeError("--include-vit-first-frame requires an ActionTransformPipeline-compatible transform.")

        caption_key = getattr(transform, "caption_key", "ai_caption")
        text_token_key = getattr(transform, "text_token_key", "text_token_ids")
        required_keys = {text_token_key, "text_position_ids", "pixel_values", "image_grid_thw"}
        missing_keys = sorted(required_keys.difference(sample))
        if missing_keys:
            raise ValueError(f"ViT transform output is missing required keys: {missing_keys}.")

        unconditional_sample = dict(sample)
        unconditional_sample[caption_key] = ""
        unconditional_sample = tokenize_with_vit(unconditional_sample)

        conditional_tokens = [_single_text_token_list(sample[text_token_key], text_token_key)]
        unconditional_tokens = [
            _single_text_token_list(unconditional_sample[text_token_key], f"unconditional_{text_token_key}")
        ]
        conditional_position_ids = [_single_text_position_ids(sample["text_position_ids"], "text_position_ids")]
        unconditional_position_ids = [
            _single_text_position_ids(
                unconditional_sample["text_position_ids"],
                "unconditional_text_position_ids",
            )
        ]
        conditional_soft_plan_insert_indexes: list[int] | None = None
        unconditional_soft_plan_insert_indexes: list[int] | None = None
        if bool(getattr(transform, "mid_soft", False)):
            for branch, branch_sample in (
                ("conditional", sample),
                ("unconditional", unconditional_sample),
            ):
                if "soft_plan_insert_index" not in branch_sample:
                    raise ValueError(f"ViT mid-soft {branch} transform output is missing soft_plan_insert_index.")
            conditional_soft_plan_insert_indexes = [int(sample["soft_plan_insert_index"])]
            unconditional_soft_plan_insert_indexes = [int(unconditional_sample["soft_plan_insert_index"])]

        for branch, tokens, position_ids, insert_indexes in (
            (
                "conditional",
                conditional_tokens[0],
                conditional_position_ids[0],
                conditional_soft_plan_insert_indexes,
            ),
            (
                "unconditional",
                unconditional_tokens[0],
                unconditional_position_ids[0],
                unconditional_soft_plan_insert_indexes,
            ),
        ):
            if position_ids.shape[1] != len(tokens):
                raise ValueError(
                    f"ViT {branch} token/position mismatch: {len(tokens)} tokens and "
                    f"position shape {tuple(position_ids.shape)}."
                )
            if insert_indexes is not None and not 0 <= insert_indexes[0] <= len(tokens):
                raise ValueError(
                    f"ViT mid-soft {branch} insert_index={insert_indexes[0]} "
                    f"outside text token range [0, {len(tokens)}]."
                )

        conditional_grid = _single_tensor(sample["image_grid_thw"], "image_grid_thw")
        unconditional_grid = _single_tensor(
            unconditional_sample["image_grid_thw"],
            "unconditional_image_grid_thw",
        )
        if not torch.equal(conditional_grid, unconditional_grid):
            raise ValueError("Conditional and unconditional ViT image grids must match.")

        processor = getattr(transform, "_vit_processor", None)
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("ViT processor does not expose image_token_id.")
        for branch, tokens in (
            ("conditional", conditional_tokens[0]),
            ("unconditional", unconditional_tokens[0]),
        ):
            if int(image_token_id) not in tokens:
                raise ValueError(f"ViT {branch} text sequence does not contain image tokens.")

        return _VitTextConditioning(
            conditional_tokens=conditional_tokens,
            unconditional_tokens=unconditional_tokens,
            conditional_position_ids=conditional_position_ids,
            unconditional_position_ids=unconditional_position_ids,
            conditional_soft_plan_insert_indexes=conditional_soft_plan_insert_indexes,
            unconditional_soft_plan_insert_indexes=unconditional_soft_plan_insert_indexes,
        )

    @staticmethod
    def _clean_reasoner_text(value: Any, *, fallback_prompt: str | None = None) -> str:
        """Normalize one generated Reasoner answer before using it as a caption.

        ``generate_reasoner_text`` normally removes special tokens while
        detokenizing.  Keep this small defensive cleanup here as exported
        tokenizer variants may still expose an assistant boundary marker.
        Do not otherwise rewrite the model's plan: the text is both the WAM
        condition and the value returned to RoboLab for optional overlay.
        """
        if not isinstance(value, str):
            return ""
        cleaned = value.replace("<|im_end|>", "").strip()
        # ``generate_reasoner_text`` deliberately returns the original prompt
        # when an AR decode is empty.  That prompt is an instruction template,
        # not a valid WAM condition, so treat it as an empty prediction and let
        # the caller use the task fallback.
        if fallback_prompt is not None and cleaned == fallback_prompt.strip():
            return ""
        # Rationale-SFT checkpoints may emit
        # ``<think>decision rationale</think>final subtask``. The rationale is
        # useful supervision but is not a deployable WAM caption. Parse it
        # strictly and pass only the final subtask downstream; malformed or
        # unfinished think blocks fail closed to the task fallback.
        return reasoner_final_subtask(cleaned)

    def _prepare_concat_image(self, image: Any, *, key: str) -> np.ndarray:
        """Apply the shared wrist rotation and resize to one concat frame."""

        image = _ensure_rgb_uint8_image(image, key)
        if self.cfg.wrist_rotate_180:
            wrist_h = image.shape[0] * 2 // 3
            image = image.copy()
            image[:wrist_h] = np.flip(image[:wrist_h], axis=(0, 1))
        image_h = self.cfg.image_height
        image_w = self.cfg.image_width
        if image.shape[:2] != (image_h, image_w):
            image = _resize_rgb_uint8(image, (image_h, image_w))
        return image

    def _prepare_separate_image(self, image: Any, *, key: str) -> np.ndarray:
        image = _ensure_rgb_uint8_image(image, key)
        if key.endswith("wrist_image_left") or "wrist" in key.lower():
            if self.cfg.wrist_rotate_180:
                image = np.flip(image, axis=(0, 1)).copy()
        edge = int(self.cfg.reasoner_image_long_edge)
        if edge <= 0:
            raise ValueError("--reasoner-image-long-edge must be positive")
        height, width = image.shape[:2]
        scale = edge / max(height, width)
        target = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
        if target != (height, width):
            image = _resize_rgb_uint8(image, target)
        return image

    def _prepare_observation_image(self, obs: dict[str, Any]) -> np.ndarray:
        """Return the exact concat frame used by the WAM sample transform."""

        return self._prepare_concat_image(_extract_observation_image(obs), key="observation/image")

    def _prepare_reasoner_images(self, obs: dict[str, Any]) -> list[np.ndarray]:
        """Return one current image or two causal images plus current.

        The optional wire field contains only the two past frames.  Reusing the
        ordinary current observation here guarantees that the newest Reasoner
        image and the WAM conditioning frame are identical.
        """

        if self.cfg.reasoner_separate_views:
            separate_payload = obs.get(_REASONER_SEPARATE_VIEWS_KEY)
            separate = None
            if separate_payload is not None:
                if isinstance(separate_payload, np.ndarray):
                    if separate_payload.ndim != 4 or separate_payload.shape[0] != 3:
                        raise ValueError(
                            f"{_REASONER_SEPARATE_VIEWS_KEY!r} must have shape [3,H,W,3], got {separate_payload.shape}"
                        )
                    separate = [separate_payload[index] for index in range(3)]
                elif isinstance(separate_payload, (list, tuple)) and len(separate_payload) == 3:
                    separate = list(separate_payload)
                else:
                    raise ValueError(f"{_REASONER_SEPARATE_VIEWS_KEY!r} must contain exactly three images")
            if separate is None:
                separate = _extract_roboarena_separate_views(obs)
            if separate is not None:
                current_views = [
                    self._prepare_separate_image(image, key=key)
                    for image, key in zip(
                        separate,
                        ("observation/exterior_image_1_left", "observation/exterior_image_2_left", "observation/wrist_image_left"),
                        strict=True,
                    )
                ]
                raw_history = obs.get(_REASONER_HISTORY_KEY)
                if raw_history is None:
                    return current_views
                if isinstance(raw_history, np.ndarray):
                    if raw_history.ndim not in (4, 5):
                        raise ValueError(f"{_REASONER_HISTORY_KEY!r} must have shape [1,H,W,3], [2,H,W,3], or [1,3,H,W,3], got {raw_history.shape}")
                    history_values = [raw_history[index] for index in range(raw_history.shape[0])]
                elif isinstance(raw_history, (list, tuple)):
                    history_values = list(raw_history)
                else:
                    raise TypeError(f"{_REASONER_HISTORY_KEY!r} must be a NumPy array or image sequence")
                if len(history_values) not in (1, 2):
                    raise ValueError(f"{_REASONER_HISTORY_KEY!r} must contain one separate-view past image or two legacy past images, got {len(history_values)}")
                history_views = []
                for index, image in enumerate(history_values):
                    if isinstance(image, np.ndarray) and image.ndim == 4 and image.shape[0] == 3:
                        for view_index, view in enumerate(image):
                            view_key = (
                                f"{_REASONER_HISTORY_KEY}[{index}][wrist]"
                                if view_index == 2
                                else f"{_REASONER_HISTORY_KEY}[{index}][{view_index}]"
                            )
                            history_views.append(
                                self._prepare_separate_image(view, key=view_key)
                            )
                    else:
                        # Legacy composite history is retained for backwards
                        # compatibility only when the payload is 4-D absent.
                        prepared = self._prepare_concat_image(image, key=f"{_REASONER_HISTORY_KEY}[{index}]")
                        history_views.extend([prepared, prepared, prepared])
                return [*history_views, *current_views]

        current = self._prepare_observation_image(obs)
        raw_history = obs.get(_REASONER_HISTORY_KEY)
        if raw_history is None:
            return [current]

        if isinstance(raw_history, np.ndarray):
            if raw_history.ndim != 4:
                raise ValueError(f"{_REASONER_HISTORY_KEY!r} must have shape [2,H,W,3], got {raw_history.shape}")
            history_values = [raw_history[index] for index in range(raw_history.shape[0])]
        elif isinstance(raw_history, (list, tuple)):
            history_values = list(raw_history)
        else:
            raise TypeError(
                f"{_REASONER_HISTORY_KEY!r} must be a NumPy array or image sequence, got {type(raw_history).__name__}."
            )
        if len(history_values) != 2:
            raise ValueError(
                f"{_REASONER_HISTORY_KEY!r} must contain exactly two past images, got {len(history_values)}."
            )

        history = [
            self._prepare_concat_image(image, key=f"{_REASONER_HISTORY_KEY}[{index}]")
            for index, image in enumerate(history_values)
        ]
        return [*history, current]

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        image = self._prepare_observation_image(obs)
        image_h = self.cfg.image_height
        image_w = self.cfg.image_width
        if self.cfg.video_sample_stride < 1:
            raise ValueError("--video-sample-stride must be >= 1")
        t_frames = (self.cfg.action_chunk_size + self.cfg.video_sample_stride) // self.cfg.video_sample_stride
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)  # [3,T,H,W]
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)  # [3,H,W]

        use_state_rows = 1 if self.cfg.use_state else 0
        action = torch.zeros(
            (self.cfg.action_chunk_size + use_state_rows, self.cfg.action_dim),
            dtype=torch.float32,
        )  # [T,D]
        history_action: torch.Tensor | None = None
        num_history_rows = self.cfg.history_length - use_state_rows
        gripper_position = 1.0 - _ensure_gripper_array(obs["observation/gripper_position"])

        if self.cfg.action_space == "joint_pos":
            joint_position = _ensure_2d_float_array(obs["observation/joint_position"], "observation/joint_position", 7)
            if self.cfg.use_state:
                action[0] = torch.from_numpy(np.concatenate((joint_position[-1], gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                if len(joint_position) < num_history_rows + 1:
                    raise ValueError("Not enough joint_position rows for requested history_length")
                history_np = np.concatenate(
                    (joint_position[-num_history_rows - 1 : -1], gripper_position[-num_history_rows - 1 : -1]),
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        if self.cfg.action_space == "midtrain":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            if self.cfg.use_state:
                rot6d = convert_rotation(eef_quat[-1], "quat_xyzw", "rot6d")
                action[0] = torch.from_numpy(np.concatenate((eef_pos[-1], rot6d, gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                if len(eef_pos) < num_history_rows + 1 or len(eef_quat) < num_history_rows + 1:
                    raise ValueError("Not enough eef_pos/eef_quat rows for requested history_length")
                poses_abs = build_abs_pose_from_components(eef_pos, eef_quat, "quat_xyzw")
                poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
                history_np = np.concatenate(
                    [poses_rel[-num_history_rows:], gripper_position[-num_history_rows:]],
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.float),  # []
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id(self.cfg.domain_name), dtype=torch.long),  # []
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        if self.cfg.include_execution_success:
            # Training mixes true/false outcomes; online rollouts always
            # condition the policy on the subtask succeeding.
            sample["execution_success"] = True
        if self.cfg.exterior_plucker_condition_enabled:
            sample["plucker_condition"] = _build_robolab_exterior_plucker_condition(image_h, image_w)
        if self.cfg.action_fps is not None:
            sample["action_fps"] = torch.tensor(self.cfg.action_fps, dtype=torch.float)
            sample["condition_first_action"] = torch.tensor(self.cfg.use_state, dtype=torch.bool)
        if history_action is not None:
            sample["history_action"] = history_action
        return self._transform(sample, self.cfg.resolution)

    def _online_ppo_update(self, rollout_id: str, reward: float, advantage: float, output_lora: str) -> dict[str, Any]:
        sample = self._ppo_pending.pop(rollout_id, None)
        if sample is None:
            raise KeyError(f"unknown or already-consumed PPO rollout_id: {rollout_id}")
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"PPO reward must be in [0,1], got {reward}")
        from cosmos_framework.rl.cosmos_reasoner_ppo import _encoded_prompt, _reasoner_logprobs
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

        with self._lock:
            trainable = [p for name, p in self.model.net.named_parameters() if "lora_" in name]
            if not trainable:
                raise RuntimeError("serving Reasoner exposes no trainable LoRA parameters")
            for name, parameter in self.model.net.named_parameters():
                parameter.requires_grad_("lora_" in name)
            if self._ppo_optimizer is None:
                self._ppo_optimizer = torch.optim.AdamW(trainable, lr=float(sample["learning_rate"]))
            processor = sample["processor"]
            encoded = _encoded_prompt(sample["prompt"], sample["images"], processor, next(self.model.parameters()).device)
            response_ids = self.model.vlm_tokenizer(sample["response"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(next(self.model.parameters()).device)
            self._ppo_optimizer.zero_grad(set_to_none=True)
            self.model.train()
            # The normal WAM branch disables the Reasoner adapter while it
            # renders actions.  This control request must both re-enable that
            # adapter and override any serving inference context, otherwise
            # the forward path reaches only frozen base weights and produces a
            # detached log-probability.
            with torch.inference_mode(False), torch.enable_grad(), _model_lora_runtime_context(
                self.model, enabled=True
            ):
                with torch.no_grad():
                    old_logprob = _reasoner_logprobs(self.model, response_ids=response_ids, encoded_prompt=encoded)
                new_logprob = _reasoner_logprobs(self.model, response_ids=response_ids, encoded_prompt=encoded)
                if not new_logprob.requires_grad:
                    raise RuntimeError(
                        "Reasoner PPO forward is detached: verify LoRA runtime is enabled and LoRA parameters "
                        "have requires_grad=True."
                    )
                mask = torch.ones_like(new_logprob)
                config = OmegaConf.create({"clip_ratio": float(sample["clip_ratio"]), "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.0, "loss_agg_mode": "token-mean", "global_batch_info": {"dp_size": 1, "batch_num_tokens": int(mask.numel()), "global_batch_size": 1, "loss_scale_factor": None}})
                loss, metrics = compute_policy_loss_vanilla(old_log_prob=old_logprob.unsqueeze(0), log_prob=new_logprob.unsqueeze(0), advantages=torch.full_like(new_logprob, advantage).unsqueeze(0), response_mask=mask.unsqueeze(0), config=config)
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            self._ppo_optimizer.step()
            self.model.eval()
            values = {canonical_reasoner_lora_key(name): parameter.detach().cpu().contiguous() for name, parameter in self.model.net.named_parameters() if ".lora_" in name}
            destination = Path(output_lora).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(values, str(destination))
            return {"status": "passed", "rollout_id": rollout_id, "checkpoint": str(destination), "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "old_logprob_mean": float(old_logprob.mean().cpu()), "new_logprob_mean": float(new_logprob.mean().detach().cpu()), "actor/ppo_kl": float(metrics.get("actor/ppo_kl", 0.0))}

    def _online_spin_update(
        self,
        rollout_id: str,
        chosen_response: str,
        rejected_response: str,
        output_lora: str,
        beta: float,
        *,
        synchronize: bool = False,
        apply_update: bool = True,
        learning_rate: float = 1e-6,
        step: int | None = None,
        defer_optimizer_step: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, Any]:
        """Apply one online-DPO update, optionally as a synchronized global step.

        Every participating server calls this method for every global step.
        Local preference pairs contribute local gradients; tie/skip ranks
        contribute zeros.  The private NCCL group then averages LoRA
        gradients before exactly one optimizer step on every replica.
        """

        from cosmos_framework.rl.cosmos_reasoner_ppo import _encoded_prompt, _reasoner_logprobs
        from cosmos_framework.rl.online_spin import online_dpo_loss, validate_preference_pair

        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if defer_optimizer_step and synchronize:
            # Synchronization belongs to finalize, after every local case has
            # contributed.  Reject accidental per-case collectives instead of
            # silently violating the global batch contract.
            raise ValueError("deferred online-DPO updates must set synchronize=False until finalize")
        if synchronize and self._online_dpo_world_size > 1 and self._online_dpo_group is None:
            raise RuntimeError("online DPO synchronization was requested but its NCCL group is not initialized")
        sample = None
        if apply_update:
            chosen_response, rejected_response = validate_preference_pair(
                chosen_response, rejected_response
            )
            sample = self._ppo_pending.get(rollout_id)
            if sample is None:
                raise KeyError(f"unknown or already-consumed SPIN rollout_id: {rollout_id}")
            sampled_response = str(sample["response"]).strip()
            if sampled_response not in {chosen_response, rejected_response}:
                raise ValueError(
                    "SPIN chosen_response or rejected_response must exactly equal the "
                    f"Reasoner text bound to rollout_id; expected {sampled_response!r}, "
                    f"got chosen={chosen_response!r}, rejected={rejected_response!r}"
                )

        with self._lock:
            if apply_update:
                # Consume only while holding the model lock. A retry cannot
                # use a pair twice, and inference cannot see a half update.
                sample = self._ppo_pending.pop(rollout_id, None)
                if sample is None:
                    raise KeyError(f"unknown or already-consumed SPIN rollout_id: {rollout_id}")
            trainable = [p for name, p in self.model.net.named_parameters() if "lora_" in name]
            if not trainable:
                raise RuntimeError("serving Reasoner exposes no trainable LoRA parameters")
            for name, parameter in self.model.net.named_parameters():
                parameter.requires_grad_("lora_" in name)
            if self._spin_optimizer is None:
                self._spin_optimizer = torch.optim.AdamW(
                    trainable,
                    lr=float(sample["learning_rate"] if sample is not None else learning_rate),
                    weight_decay=0.0,
                )
            if defer_optimizer_step:
                if self._spin_accumulation_steps is None:
                    self._spin_accumulation_steps = int(gradient_accumulation_steps)
                elif self._spin_accumulation_steps != int(gradient_accumulation_steps):
                    raise RuntimeError(
                        "gradient_accumulation_steps changed while an online-DPO batch is pending: "
                        f"{self._spin_accumulation_steps} -> {gradient_accumulation_steps}"
                    )
                if self._spin_accumulated_cases >= self._spin_accumulation_steps:
                    raise RuntimeError(
                        "online-DPO accumulation batch is full; call finalize_spin_reasoner_lora before another case"
                    )
            device = next(self.model.parameters()).device
            self._spin_optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            loss = None
            dpo_metrics: dict[str, Any] = {}
            chosen_ids = rejected_ids = None
            reference_policy_name = "frozen_initial_reasoner_lora"
            if apply_update:
                assert sample is not None
                encoded = _encoded_prompt(sample["prompt"], sample["images"], sample["processor"], device)
                chosen_ids = self.model.vlm_tokenizer(
                    chosen_response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)
                rejected_ids = self.model.vlm_tokenizer(
                    rejected_response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)
                with torch.inference_mode(False), torch.enable_grad(), _model_lora_runtime_context(
                    self.model, enabled=True
                ):
                    # The reference is the frozen initial SFT adapter. It is
                    # swapped into the same resident model briefly, avoiding a
                    # second VLM copy on the GPU.
                    actor_state = {
                        name: parameter.detach().clone()
                        for name, parameter in self.model.net.named_parameters()
                        if "lora_" in name
                    }
                    reference_lora = self._spin_reference_lora
                    if reference_lora is None:
                        # Lightweight test doubles may construct the service
                        # without running its full model initializer. Keep a
                        # moving-reference fallback only for that case; real
                        # adapter servers always capture the fixed reference.
                        reference_lora = {
                            name: value.detach().cpu().clone() for name, value in actor_state.items()
                        }
                        reference_policy_name = "same_reasoner_lora_before_update"
                    with torch.no_grad():
                        for name, parameter in self.model.net.named_parameters():
                            reference = reference_lora.get(name)
                            if reference is not None:
                                parameter.copy_(reference.to(device=device, dtype=parameter.dtype))
                        reference_chosen = _reasoner_logprobs(
                            self.model, response_ids=chosen_ids, encoded_prompt=encoded
                        )
                        reference_rejected = _reasoner_logprobs(
                            self.model, response_ids=rejected_ids, encoded_prompt=encoded
                        )
                        for name, parameter in self.model.net.named_parameters():
                            actor = actor_state.get(name)
                            if actor is not None:
                                parameter.copy_(actor)
                        policy_chosen_eval = _reasoner_logprobs(
                            self.model, response_ids=chosen_ids, encoded_prompt=encoded
                        )
                        policy_rejected_eval = _reasoner_logprobs(
                            self.model, response_ids=rejected_ids, encoded_prompt=encoded
                        )
                        loss, dpo_metrics = online_dpo_loss(
                            chosen_logprobs=policy_chosen_eval,
                            rejected_logprobs=policy_rejected_eval,
                            reference_chosen_logprobs=reference_chosen,
                            reference_rejected_logprobs=reference_rejected,
                            beta=float(beta),
                        )
                        stream_scale = float(beta) * torch.sigmoid(-dpo_metrics["dpo_logit"])
                    chosen_logprob = _reasoner_logprobs(
                        self.model, response_ids=chosen_ids, encoded_prompt=encoded
                    )
                    (-stream_scale * chosen_logprob.sum()).backward()
                    rejected_logprob = _reasoner_logprobs(
                        self.model, response_ids=rejected_ids, encoded_prompt=encoded
                    )
                    (stream_scale * rejected_logprob.sum()).backward()

            if defer_optimizer_step:
                # Average the local case gradients at accumulation time.  A
                # tie contributes a zero gradient but still counts toward the
                # batch, so all replicas finalize after exactly the same case
                # count even when their judge outcomes differ.
                scale = float(gradient_accumulation_steps)
                for name, parameter in self.model.net.named_parameters():
                    if "lora_" not in name:
                        continue
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    contribution = gradient.detach().clone().div_(scale)
                    previous = self._spin_accumulated_grad.get(name)
                    self._spin_accumulated_grad[name] = contribution if previous is None else previous.add_(contribution)
                self._spin_accumulated_cases += 1
                if loss is not None:
                    self._spin_accumulated_losses.append(float(loss.detach().cpu()))
                self._spin_optimizer.zero_grad(set_to_none=True)
                return {
                    "status": "passed",
                    "algorithm": "online_dpo_spin",
                    "rollout_id": rollout_id,
                    "checkpoint": None,
                    "reference_policy": reference_policy_name if apply_update else "frozen_initial_reasoner_lora",
                    "reference_update_frequency": 0,
                    "synchronized": False,
                    "optimizer_step": False,
                    "accumulated": True,
                    "accumulation_cases": self._spin_accumulated_cases,
                    "accumulation_steps": self._spin_accumulation_steps,
                    "global_step": step,
                    "local_preference": bool(apply_update),
                    "beta": float(beta),
                    "chosen_response": chosen_response,
                    "rejected_response": rejected_response,
                    "loss": float(loss.detach().cpu()) if loss is not None else None,
                    "grad_norm": 0.0,
                    "chosen_tokens": int(chosen_ids.numel()) if chosen_ids is not None else 0,
                    "rejected_tokens": int(rejected_ids.numel()) if rejected_ids is not None else 0,
                    "chosen_logprob_sum_before": float(dpo_metrics["chosen_logprob_sum"].cpu()) if dpo_metrics else None,
                    "rejected_logprob_sum_before": float(dpo_metrics["rejected_logprob_sum"].cpu()) if dpo_metrics else None,
                    "reference_margin": float(dpo_metrics["reference_margin"].cpu()) if dpo_metrics else None,
                    "policy_margin_before": float(dpo_metrics["policy_margin"].cpu()) if dpo_metrics else None,
                    "dpo_logit_before": float(dpo_metrics["dpo_logit"].cpu()) if dpo_metrics else None,
                }

            if synchronize and self._online_dpo_world_size > 1:
                # One collective for the whole LoRA adapter avoids hundreds
                # of small NCCL launches for a large Reasoner.
                flat_grad = torch.cat(
                    [
                        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
                        for parameter in trainable
                    ]
                )
                torch.distributed.all_reduce(flat_grad, group=self._online_dpo_group)
                flat_grad.div_(float(self._online_dpo_world_size))
                offset = 0
                for parameter in trainable:
                    size = parameter.numel()
                    parameter.grad = flat_grad[offset : offset + size].view_as(parameter)
                    offset += size
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # A synchronized global step always advances every replica once;
            # with all-zero gradients this is a true no-op because weight
            # decay is disabled above.
            self._spin_optimizer.step()
            self.model.eval()
            destination = Path(output_lora).expanduser().resolve() if output_lora else None
            if destination is not None and self._online_dpo_rank == 0:
                values = {
                    canonical_reasoner_lora_key(name): parameter.detach().cpu().contiguous()
                    for name, parameter in self.model.net.named_parameters()
                    if ".lora_" in name
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_file(values, str(destination))
                if self._spin_lora_metadata is not None:
                    destination.with_suffix(".json").write_text(
                        json.dumps(self._spin_lora_metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            return {
                "status": "passed",
                "algorithm": "online_dpo_spin",
                "rollout_id": rollout_id,
                "checkpoint": str(destination) if destination is not None and self._online_dpo_rank == 0 else None,
                "reference_policy": reference_policy_name if apply_update else "frozen_initial_reasoner_lora",
                "reference_update_frequency": 1 if reference_policy_name == "same_reasoner_lora_before_update" else 0,
                "synchronized": bool(synchronize and self._online_dpo_world_size > 1),
                "global_step": step,
                "local_preference": bool(apply_update),
                "beta": float(beta),
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
                "rejected_responses": [rejected_response, *[str(v) for v in (rejected_responses or [])]],
                "loss": float(loss.detach().cpu()) if loss is not None else None,
                "grad_norm": float(grad_norm.detach().cpu()),
                "chosen_tokens": int(chosen_ids.numel()) if chosen_ids is not None else 0,
                "rejected_tokens": int(rejected_ids.numel()) if rejected_ids is not None else 0,
                "chosen_logprob_sum_before": float(dpo_metrics["chosen_logprob_sum"].cpu()) if dpo_metrics else None,
                "rejected_logprob_sum_before": float(dpo_metrics["rejected_logprob_sum"].cpu()) if dpo_metrics else None,
                "reference_margin": float(dpo_metrics["reference_margin"].cpu()) if dpo_metrics else None,
                "policy_margin_before": float(dpo_metrics["policy_margin"].cpu()) if dpo_metrics else None,
                "dpo_logit_before": float(dpo_metrics["dpo_logit"].cpu()) if dpo_metrics else None,
            }

    def _online_orpo_update(
        self,
        rollout_id: str,
        chosen_response: str,
        rejected_response: str,
        output_lora: str,
        beta: float,
        *,
        synchronize: bool = False,
        apply_update: bool = True,
        learning_rate: float = 1e-6,
        step: int | None = None,
        orpo_front_tokens: int = 0,
        orpo_front_weight: float = 1.0,
        rejected_responses: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply one online-ORPO update, optionally as a synchronized global step.

        ORPO is reference-free, so this path skips the frozen-reference swap
        and directly uses the ORPO odds-ratio objective.  Every participating
        server calls this method for every global step.  Local preference
        pairs contribute local gradients; tie/skip ranks contribute zeros.
        The private NCCL group then averages LoRA gradients before exactly
        one optimizer step on every replica.
        """

        from cosmos_framework.rl.cosmos_reasoner_ppo import _encoded_prompt, _reasoner_logprobs
        from cosmos_framework.rl.online_spin import orpo_loss, validate_preference_pair

        if synchronize and self._online_dpo_world_size > 1 and self._online_dpo_group is None:
            raise RuntimeError("online ORPO synchronization was requested but its NCCL group is not initialized")
        sample = None
        if apply_update:
            chosen_response, rejected_response = validate_preference_pair(
                chosen_response, rejected_response
            )
            # Multi-rejected (roll3): one chosen vs N rejected subgoals.  The
            # first rejected stays in ``rejected_response`` for back-compat;
            # extras arrive via ``rejected_responses``.  Validate each pair.
            extra_rejected: list[str] = []
            if rejected_responses:
                for value in rejected_responses:
                    value = str(value or "").strip()
                    if value and value.casefold() != chosen_response.casefold() and value.casefold() != rejected_response.casefold():
                        extra_rejected.append(value)
            sample = self._ppo_pending.get(rollout_id)
            if sample is None:
                raise KeyError(f"unknown or already-consumed ORPO rollout_id: {rollout_id}")
            sampled_response = str(sample["response"]).strip()
            if sampled_response not in {chosen_response, rejected_response}:
                raise ValueError(
                    "ORPO chosen_response or rejected_response must exactly equal the "
                    f"Reasoner text bound to rollout_id; expected {sampled_response!r}, "
                    f"got chosen={chosen_response!r}, rejected={rejected_response!r}"
                )

        with self._lock:
            if apply_update:
                # Consume only while holding the model lock. A retry cannot
                # use a pair twice, and inference cannot see a half update.
                sample = self._ppo_pending.pop(rollout_id, None)
                if sample is None:
                    raise KeyError(f"unknown or already-consumed ORPO rollout_id: {rollout_id}")
            trainable = [p for name, p in self.model.net.named_parameters() if "lora_" in name]
            if not trainable:
                raise RuntimeError("serving Reasoner exposes no trainable LoRA parameters")
            for name, parameter in self.model.net.named_parameters():
                parameter.requires_grad_("lora_" in name)
            if self._spin_optimizer is None:
                self._spin_optimizer = torch.optim.AdamW(
                    trainable,
                    lr=float(sample["learning_rate"] if sample is not None else learning_rate),
                    weight_decay=0.0,
                )
            device = next(self.model.parameters()).device
            self._spin_optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            loss = None
            orpo_metrics: dict[str, Any] = {}
            chosen_ids = rejected_ids = None
            if apply_update:
                assert sample is not None
                encoded = _encoded_prompt(sample["prompt"], sample["images"], sample["processor"], device)
                chosen_ids = self.model.vlm_tokenizer(
                    chosen_response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)
                rejected_ids = self.model.vlm_tokenizer(
                    rejected_response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)
                # Roll3 multi-rejected: tokenize each extra rejected subgoal so
                # its token logprobs can be concatenated into one rejected
                # distribution for a single odds-ratio loss.
                extra_rejected_ids = [
                    self.model.vlm_tokenizer(value, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                    for value in (extra_rejected if rejected_responses else [])
                ]
                with torch.inference_mode(False), torch.enable_grad(), _model_lora_runtime_context(
                    self.model, enabled=True
                ):
                    chosen_logprobs = _reasoner_logprobs(
                        self.model, response_ids=chosen_ids, encoded_prompt=encoded
                    )
                    rejected_logprobs_list = [
                        _reasoner_logprobs(self.model, response_ids=rejected_ids, encoded_prompt=encoded)
                    ]
                    for extra_ids in extra_rejected_ids:
                        rejected_logprobs_list.append(
                            _reasoner_logprobs(self.model, response_ids=extra_ids, encoded_prompt=encoded)
                        )
                    rejected_logprobs = (
                        torch.cat(rejected_logprobs_list)
                        if len(rejected_logprobs_list) > 1
                        else rejected_logprobs_list[0]
                    )
                    loss, orpo_metrics = orpo_loss(
                        chosen_logprobs=chosen_logprobs,
                        rejected_logprobs=rejected_logprobs,
                        beta=float(beta),
                        front_tokens=int(orpo_front_tokens),
                        front_weight=float(orpo_front_weight),
                    )
                    loss.backward()

            if synchronize and self._online_dpo_world_size > 1:
                # One collective for the whole LoRA adapter avoids hundreds
                # of small NCCL launches for a large Reasoner.
                flat_grad = torch.cat(
                    [
                        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
                        for parameter in trainable
                    ]
                )
                torch.distributed.all_reduce(flat_grad, group=self._online_dpo_group)
                flat_grad.div_(float(self._online_dpo_world_size))
                offset = 0
                for parameter in trainable:
                    size = parameter.numel()
                    parameter.grad = flat_grad[offset : offset + size].view_as(parameter)
                    offset += size
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # A synchronized global step always advances every replica once;
            # with all-zero gradients this is a true no-op because weight
            # decay is disabled above.
            self._spin_optimizer.step()
            self.model.eval()
            destination = Path(output_lora).expanduser().resolve() if output_lora else None
            if destination is not None and self._online_dpo_rank == 0:
                values = {
                    canonical_reasoner_lora_key(name): parameter.detach().cpu().contiguous()
                    for name, parameter in self.model.net.named_parameters()
                    if ".lora_" in name
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_file(values, str(destination))
                if self._spin_lora_metadata is not None:
                    destination.with_suffix(".json").write_text(
                        json.dumps(self._spin_lora_metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            return {
                "status": "passed",
                "algorithm": "online_orpo",
                "rollout_id": rollout_id,
                "checkpoint": str(destination) if destination is not None and self._online_dpo_rank == 0 else None,
                "reference_policy": "none",
                "reference_update_frequency": 0,
                "synchronized": bool(synchronize and self._online_dpo_world_size > 1),
                "global_step": step,
                "local_preference": bool(apply_update),
                "beta": float(beta),
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
                "loss": float(loss.detach().cpu()) if loss is not None else None,
                "grad_norm": float(grad_norm.detach().cpu()),
                "chosen_tokens": int(chosen_ids.numel()) if chosen_ids is not None else 0,
                "rejected_tokens": int(rejected_ids.numel()) if rejected_ids is not None else 0,
                "chosen_logprob_mean": float(orpo_metrics["chosen_logprob_mean"].cpu()) if orpo_metrics else None,
                "rejected_logprob_mean": float(orpo_metrics["rejected_logprob_mean"].cpu()) if orpo_metrics else None,
                "policy_margin_mean": float(orpo_metrics["policy_margin_mean"].cpu()) if orpo_metrics else None,
                "odds_logit": float(orpo_metrics["odds_logit"].cpu()) if orpo_metrics else None,
                "chosen_nll": float(orpo_metrics["chosen_nll"].cpu()) if orpo_metrics else None,
                "odds_ratio_loss": float(orpo_metrics["odds_ratio_loss"].cpu()) if orpo_metrics else None,
                "orpo_front_tokens": int(orpo_front_tokens),
                "orpo_front_weight": float(orpo_front_weight),
            }

    def _finalize_online_spin_update(
        self,
        output_lora: str,
        *,
        synchronize: bool = False,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-6,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Apply one averaged update after a complete local accumulation batch."""

        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if synchronize and self._online_dpo_world_size > 1 and self._online_dpo_group is None:
            raise RuntimeError("online DPO synchronization was requested but its NCCL group is not initialized")
        with self._lock:
            expected = self._spin_accumulation_steps or int(gradient_accumulation_steps)
            if expected != int(gradient_accumulation_steps):
                raise RuntimeError(
                    f"finalize accumulation_steps={gradient_accumulation_steps} does not match pending batch {expected}"
                )
            if self._spin_accumulated_cases != expected:
                raise RuntimeError(
                    "cannot finalize incomplete online-DPO batch: "
                    f"received {self._spin_accumulated_cases}/{expected} cases"
                )
            trainable = [p for name, p in self.model.net.named_parameters() if "lora_" in name]
            if not trainable:
                raise RuntimeError("serving Reasoner exposes no trainable LoRA parameters")
            for name, parameter in self.model.net.named_parameters():
                parameter.requires_grad_("lora_" in name)
            if self._spin_optimizer is None:
                self._spin_optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=0.0)
            device = next(self.model.parameters()).device
            flat_grad = torch.cat(
                [
                    self._spin_accumulated_grad.get(name, torch.zeros_like(parameter)).reshape(-1)
                    for name, parameter in self.model.net.named_parameters()
                    if "lora_" in name
                ]
            )
            if synchronize and self._online_dpo_world_size > 1:
                # Exactly one collective for the entire averaged LoRA batch.
                torch.distributed.all_reduce(flat_grad, group=self._online_dpo_group)
                flat_grad.div_(float(self._online_dpo_world_size))
            offset = 0
            for name, parameter in self.model.net.named_parameters():
                if "lora_" not in name:
                    continue
                size = parameter.numel()
                parameter.grad = flat_grad[offset : offset + size].view_as(parameter)
                offset += size
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            self._spin_optimizer.step()
            self._spin_optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            destination = Path(output_lora).expanduser().resolve() if output_lora else None
            if destination is not None and self._online_dpo_rank == 0:
                values = {
                    canonical_reasoner_lora_key(name): parameter.detach().cpu().contiguous()
                    for name, parameter in self.model.net.named_parameters()
                    if ".lora_" in name
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_file(values, str(destination))
                if self._spin_lora_metadata is not None:
                    destination.with_suffix(".json").write_text(
                        json.dumps(self._spin_lora_metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            result = {
                "status": "passed",
                "algorithm": "online_dpo_spin",
                "checkpoint": str(destination) if destination is not None and self._online_dpo_rank == 0 else None,
                "synchronized": bool(synchronize and self._online_dpo_world_size > 1),
                "optimizer_step": True,
                "accumulated": True,
                "accumulation_cases": expected,
                "accumulation_steps": expected,
                "global_step": step,
                "local_preferences": int(len(self._spin_accumulated_losses)),
                "loss_mean": (
                    float(sum(self._spin_accumulated_losses) / len(self._spin_accumulated_losses))
                    if self._spin_accumulated_losses
                    else None
                ),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            self._spin_accumulation_steps = None
            self._spin_accumulated_cases = 0
            self._spin_accumulated_grad.clear()
            self._spin_accumulated_losses.clear()
            return result

    def _discard_reasoner_rollout(self, rollout_id: str) -> dict[str, Any]:
        """Release a pending sampled response when pairwise judging skips it."""

        with self._lock:
            existed = self._ppo_pending.pop(str(rollout_id), None) is not None
        if not existed:
            raise KeyError(f"unknown or already-consumed rollout_id: {rollout_id}")
        return {
            "status": "discarded",
            "rollout_id": str(rollout_id),
            "reason": "pairwise_judge_not_eligible_for_update",
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        # PPO actors keep the expensive frozen Reasoner/WAM service alive.
        # A local control request swaps only the already-injected Reasoner
        # adapter between rollout batches.  It is deliberately serialized by
        # the same lock as inference, so no WAM request can observe half an
        # adapter update.  The WebSocket server is bound to localhost by the
        # training launcher; this is not an external serving API.
        if obs.get("_robolab_control") == "reload_reasoner_lora":
            lora_path = str(obs.get("reasoner_lora_path") or "").strip()
            if not self.cfg.reasoner_wam or self.cfg.reasoner_wam_mode != "adapter":
                raise ValueError("reload_reasoner_lora requires --reasoner-wam-mode=adapter")
            if not lora_path:
                raise ValueError("reload_reasoner_lora requires reasoner_lora_path")
            with self._lock:
                tensor_count = _load_reasoner_lora_weights(self.model, lora_path)
                # The active model tensors are authoritative; the compact
                # serving config intentionally does not duplicate the path.
                self._active_reasoner_lora_path = lora_path
            return {
                "control": "reload_reasoner_lora",
                "reasoner_lora_path": lora_path,
                "tensor_count": tensor_count,
            }
        if obs.get("_robolab_control") == "ppo_update_reasoner_lora":
            return self._online_ppo_update(
                str(obs.get("rollout_id") or ""),
                float(obs.get("reward")),
                float(obs.get("advantage")),
                str(obs.get("output_lora") or ""),
            )
        if obs.get("_robolab_control") == "spin_update_reasoner_lora":
            return self._online_spin_update(
                str(obs.get("rollout_id") or ""),
                str(obs.get("chosen_response") or ""),
                str(obs.get("rejected_response") or ""),
                str(obs.get("output_lora") or ""),
                float(obs.get("beta", 0.1)),
                synchronize=bool(obs.get("synchronize", False)),
                apply_update=bool(obs.get("apply_update", True)),
                learning_rate=float(obs.get("learning_rate", 1e-6)),
                step=(int(obs["step"]) if obs.get("step") is not None else None),
                defer_optimizer_step=bool(obs.get("defer_optimizer_step", False)),
                gradient_accumulation_steps=int(obs.get("gradient_accumulation_steps", 1)),
            )
        if obs.get("_robolab_control") == "orpo_update_reasoner_lora":
            return self._online_orpo_update(
                str(obs.get("rollout_id") or ""),
                str(obs.get("chosen_response") or ""),
                str(obs.get("rejected_response") or ""),
                str(obs.get("output_lora") or ""),
                float(obs.get("beta", 0.1)),
                synchronize=bool(obs.get("synchronize", False)),
                apply_update=bool(obs.get("apply_update", True)),
                learning_rate=float(obs.get("learning_rate", 1e-6)),
                step=(int(obs["step"]) if obs.get("step") is not None else None),
                orpo_front_tokens=int(obs.get("orpo_front_tokens", 0)),
                orpo_front_weight=float(obs.get("orpo_front_weight", 1.0)),
                rejected_responses=(
                    [str(v) for v in obs.get("rejected_responses")] if obs.get("rejected_responses") else None
                ),
            )
        if obs.get("_robolab_control") == "finalize_spin_reasoner_lora":
            return self._finalize_online_spin_update(
                str(obs.get("output_lora") or ""),
                synchronize=bool(obs.get("synchronize", False)),
                gradient_accumulation_steps=int(obs.get("gradient_accumulation_steps", 1)),
                learning_rate=float(obs.get("learning_rate", 1e-6)),
                step=(int(obs["step"]) if obs.get("step") is not None else None),
            )
        if obs.get("_robolab_control") == "discard_reasoner_rollout":
            return self._discard_reasoner_rollout(str(obs.get("rollout_id") or ""))
        task_prompt = obs.get("prompt")
        if not isinstance(task_prompt, str):
            raise ValueError("'prompt' must be a string")
        task_prompt = task_prompt.strip()
        if self.cfg.reasoner_wam and not task_prompt:
            raise ValueError("'prompt' must not be empty")

        # Reasoner-first mode may split one logical inference into a
        # reasoner-only request followed by a skip_reasoner WAM request. When
        # the client carries the first request's seed, reuse it instead of
        # advancing the server RNG and changing the WAM diffusion noise.
        seed_override = obs.get("generation_seed")
        if bool(obs.get("skip_reasoner", False)) and seed_override is not None:
            try:
                seed = int(seed_override)
            except (TypeError, ValueError) as exc:
                raise ValueError("generation_seed must be an integer") from exc
        else:
            seed = self._next_seed()
        # GRPO group-relative judging needs every candidate in one group to
        # share the SAME WAM diffusion noise so reward differences reflect the
        # sampled subtask, not sampling luck.  ``wam_seed`` pins only the WAM
        # rectified-flow draw; the Reasoner subtask decode above still uses the
        # per-request ``seed`` so candidates stay diverse within the group.
        wam_seed_override = obs.get("wam_seed")
        if wam_seed_override is not None:
            try:
                wam_seed = int(wam_seed_override)
            except (TypeError, ValueError) as exc:
                raise ValueError("wam_seed must be an integer") from exc
        else:
            wam_seed = seed
        predicted_subtask = ""
        ppo_rollout_id = ""

        with self._lock:
            with torch.inference_mode():
                wam_obs = obs
                reasoner_model = self.model
                wam_model = getattr(self, "wam_model", self.model)
                # Interactive curation can request the Reasoner pass before
                # the supervisor chooses the WAM caption.  In that phase the
                # server must not spend time sampling actions.  The later WAM
                # request carries ``skip_reasoner`` so the Reasoner is not
                # decoded a second time for the same chunk.
                if self.cfg.reasoner_wam and not bool(obs.get("skip_reasoner", False)):
                    # The first action chunk uses one current image.  Later
                    # requests may carry two causal frames; append the same
                    # current concat frame used by WAM to reproduce the trained
                    # [-16, -8, 0] three-image layout.
                    reasoner_images = self._prepare_reasoner_images(obs)
                    reasoner_image_input: Any = reasoner_images[0] if len(reasoner_images) == 1 else reasoner_images
                    preview_action: str | None = None
                    if self.cfg.reasoner_include_preview_action and len(reasoner_images) > 1:
                        if _REASONER_PREVIEW_ACTION_KEY in obs:
                            raw_preview_action = obs[_REASONER_PREVIEW_ACTION_KEY]
                        elif "preview_action" in obs:
                            # Temporary compatibility alias for early clients.
                            raw_preview_action = obs["preview_action"]
                        else:
                            raise ValueError(
                                "Prev-mode Reasoner inference requires "
                                f"{_REASONER_PREVIEW_ACTION_KEY!r} on every three-image request. "
                                "Enable the matching RoboLab client flag."
                            )
                        if not isinstance(raw_preview_action, str):
                            raise TypeError(f"{_REASONER_PREVIEW_ACTION_KEY!r} must be a string")
                        preview_action = raw_preview_action.strip() or None
                    reasoner_prompt = format_action_prediction_prompt(
                        task_prompt,
                        preview_action=preview_action,
                        separate_views=self.cfg.reasoner_separate_views,
                        require_think=self.cfg.reasoner_require_think,
                    )
                    if self.cfg.reasoner_separate_views:
                        group_count = len(reasoner_images) // 3
                        group_labels = ["CURRENT"] if group_count == 1 else ["PREV"] * (group_count - 1) + ["CURRENT"]
                        reasoner_prompt = "\n".join(f"{label}:" for label in group_labels) + "\n" + reasoner_prompt
                    # The exported Reasoner-WAM model may retain a text-only
                    # ``vlm_config.tokenizer``.  The serving ActionTransform
                    # owns the Qwen3-VL processor configured for the WAM
                    # observation resolution, so explicitly pass that
                    # processor to this opt-in image-conditioned call.  Keep
                    # all legacy paths on the model's original processor.
                    reasoner_transform = getattr(self, "_reasoner_transform", self._transform)
                    reasoner_processor = getattr(reasoner_transform, "_vit_processor", None)
                    get_reasoner_processor = getattr(reasoner_transform, "_get_vit_processor", None)
                    if callable(get_reasoner_processor):
                        reasoner_processor = get_reasoner_processor()
                    reasoner_kwargs: dict[str, Any] = {
                        "max_new_tokens": self.cfg.reasoner_max_new_tokens,
                        # Outer list is per prompt; a nested list is one
                        # prompt's ordered multi-image group.
                        "images": [reasoner_image_input],
                        "do_sample": self.cfg.reasoner_do_sample,
                        "seed": seed,
                    }
                    if reasoner_processor is not None:
                        reasoner_kwargs["processor"] = reasoner_processor
                    # Mode2 keeps the extracted adapter active for this
                    # ordinary VLM decode.  The context is a no-op for the
                    # lightweight test doubles and for models without LoRA.
                    with _model_lora_runtime_context(reasoner_model, enabled=True):
                        generated = reasoner_model.generate_reasoner_text([reasoner_prompt], **reasoner_kwargs)
                    if not isinstance(generated, list) or len(generated) != 1:
                        raise RuntimeError(
                            "Reasoner-WAM expected generate_reasoner_text() to return one string for one request."
                        )
                    raw_predicted_subtask = self._clean_reasoner_text(
                        generated[0], fallback_prompt=reasoner_prompt
                    )
                    predicted_subtask = (
                        rewrite_move_open_gripper_text(raw_predicted_subtask)
                        if self.cfg.reasoner_open_gripper_move_augment
                        else raw_predicted_subtask
                    )
                    # Keep the exact Reasoner observation and sampled text in
                    # this serving model.  The judge later references this id
                    # to run PPO backward on the same LoRA tensors.
                    if predicted_subtask:
                        ppo_rollout_id = uuid.uuid4().hex
                        self._ppo_pending[ppo_rollout_id] = {
                            "prompt": reasoner_prompt,
                            "images": [np.ascontiguousarray(image, dtype=np.uint8).copy() for image in reasoner_images],
                            # Keep PPO aligned with the actual sampled Reasoner
                            # response.  The optional caption rewrite is only
                            # for WAM conditioning and evaluation output.
                            "response": raw_predicted_subtask,
                            "processor": reasoner_processor,
                            "learning_rate": 1e-6,
                            "clip_ratio": 0.2,
                        }
                    # An empty/invalid AR answer must not turn the WAM
                    # condition into an empty caption.  Keep the original task
                    # as a safe inference-time fallback, while returning the
                    # empty prediction so RoboLab does not render stale text.
                    # In observe-only mode the Reasoner output is recorded for
                    # inspection, but the caller's verified supervisor prompt
                    # remains the WAM conditioning text.  This is useful for
                    # offline curation where Reasoner-8000 is diagnostic and
                    # the curated subgoal is the action target.
                    wam_caption = task_prompt if self.cfg.reasoner_observe_only else (predicted_subtask or task_prompt)
                    wam_obs = dict(obs)
                    wam_obs["prompt"] = wam_caption
                    log.info(
                        f"[robolab-policy-server] reasoner_prompt={reasoner_prompt!r} "
                        f"reasoner_image_count={len(reasoner_images)} "
                        f"raw_predicted_subtask={raw_predicted_subtask!r} "
                        f"predicted_subtask={predicted_subtask!r} wam_caption={wam_caption!r}"
                    )

                    if bool(obs.get("reasoner_only", False)):
                        response = {
                            "subtask": predicted_subtask,
                            "reasoner_prompt": reasoner_prompt,
                            "reasoner_only": True,
                            "generation_seed": seed,
                        }
                        if ppo_rollout_id:
                            response["ppo_rollout_id"] = ppo_rollout_id
                        return response

                elif bool(obs.get("reasoner_only", False)):
                    raise ValueError("reasoner_only requires --reasoner-wam")

                if self.cfg.reasoner_wam and bool(obs.get("skip_reasoner", False)):
                    # ``task_prompt`` is the supervisor-selected caption.
                    # Keep the native WAM checkpoint path and do not run the
                    # privileged Reasoner a second time.
                    wam_obs = dict(obs)
                    wam_obs["prompt"] = task_prompt
                    ppo_rollout_id = str(obs.get("ppo_rollout_id") or "").strip()

                sample = self._build_sample(wam_obs)
                vit_conditioning = (
                    self._build_vit_text_conditioning(sample) if self.cfg.include_vit_first_frame else None
                )
                data_batch = _build_data_batch_from_sample(sample)
                log.info(f"[robolab-policy-server] prompt={data_batch['ai_caption'][0]!r} seed={seed}")
                adapter = (
                    _use_vit_text_conditioning(wam_model, vit_conditioning)
                    if vit_conditioning is not None
                    else nullcontext()
                )
                # Mode2's defining invariant: WAM consumes the standalone
                # base checkpoint and never sees the Reasoner LoRA delta.  A
                # separate base model has no adapters; the runtime context
                # also protects the same-model fallback.  Shared/legacy
                # Reasoner-WAM deliberately keeps its historical base+LoRA
                # routing.
                wam_lora_enabled = not (
                    self.cfg.reasoner_wam and self.cfg.reasoner_wam_mode in {"mode2", "adapter"}
                )
                with _model_lora_runtime_context(wam_model, enabled=wam_lora_enabled):
                    with adapter:
                        samples = wam_model.generate_samples_from_batch(
                            data_batch,
                            guidance=self.cfg.guidance,
                            camera_guidance=self.cfg.camera_guidance,
                            camera_cfg_mode=self.cfg.camera_cfg_mode,
                            seed=[wam_seed],
                            num_steps=self.cfg.num_steps,
                            shift=self.cfg.shift,
                            generate_subtask=self.cfg.decode_subtask,
                            subtask_max_new_tokens=self.cfg.subtask_max_new_tokens,
                        )

        action = samples["action"][0][:, : self.cfg.action_dim]  # [T,D]
        action = action[self.cfg.history_length :]  # [T2,D]
        action_np = action.detach().cpu().numpy()  # [T2,D]
        action_np[:, -1] = 1.0 - action_np[:, -1]

        if self.cfg.action_space == "midtrain":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            initial_pose = np.eye(4, dtype=np.float32)
            initial_pose[:3, :3] = convert_rotation(eef_quat[-1], "quat_xyzw", "matrix")
            initial_pose[:3, 3] = eef_pos[-1]
            abs_pose = pose_rel_to_abs(
                action_np[:, :9],
                rotation_format="rot6d",
                pose_convention="backward_framewise",
                initial_pose=initial_pose,
            )
            position = abs_pose[1:, :3, 3]
            quat_xyzw = convert_rotation(abs_pose[1:, :3, :3], "matrix", "quat_xyzw")
            action_np = np.concatenate([position, quat_xyzw, action_np[:, 9:]], axis=-1)

        outputs: dict[str, Any] = {"action": action_np}
        if self.cfg.reasoner_wam:
            # RoboLab's client caches this string for the complete action
            # chunk and its existing OVERLAY_SUBTASK_ON_VIDEO path renders it.
            outputs["subtask"] = predicted_subtask
            if ppo_rollout_id:
                outputs["ppo_rollout_id"] = ppo_rollout_id
        if self.cfg.decode_subtask:
            token_rows = samples.get("subtask_token_ids")
            if not token_rows or len(token_rows) != 1:
                raise RuntimeError("Subtask decoding did not return exactly one token sequence for this request.")
            decoded = self.wam_model.detokenize_text(token_rows[0], skip_special_tokens=True)
            if not isinstance(decoded, str):
                raise TypeError(f"Expected one decoded subtask string, got {type(decoded).__name__}.")
            outputs["subtask"] = decoded.strip()
        if self.cfg.decode_video:
            pred_vision_latent = samples["vision"][0]  # [C,T,H,W]
            video = self.wam_model.decode(pred_vision_latent)  # [1,C,T,H,W]
            video = ((video[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)  # [T,H,W,3]
            outputs["video"] = video.detach().cpu().numpy()
        return outputs


def serve(args: RobolabServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[robolab-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = RobolabPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[robolab-policy-server] Server accessible at: ws://{local_ip}:{int(args.port)}/")
    log.info(f"[robolab-policy-server] Health check: http://{local_ip}:{int(args.port)}/healthz")
    server_cls = _load_openpi_websocket_policy_server()
    metadata: dict[str, Any] = {}
    if service.cfg.reasoner_wam:
        # New clients use this capability to cache every environment frame and
        # send causal history after the first action chunk. Separate-view
        # clients default to PREV(-16)/CURRENT, while the explicit three-group
        # mode carries PREV(-16), PREV(-8), CURRENT. Legacy concat clients
        # retain two past frames (-16, -8) plus CURRENT. Old clients ignore
        # metadata and continue to exercise the valid one-image path.
        metadata["reasoner_history_offsets"] = list(
            (
                _REASONER_SEPARATE_THREE_GROUP_HISTORY_OFFSETS
                if service.cfg.reasoner_three_group_history
                else _REASONER_SEPARATE_HISTORY_OFFSETS
            )
            if service.cfg.reasoner_separate_views
            else _REASONER_HISTORY_OFFSETS
        )
        metadata["reasoner_separate_views"] = bool(service.cfg.reasoner_separate_views)
        metadata["reasoner_three_group_history"] = bool(service.cfg.reasoner_three_group_history)
        if service.cfg.reasoner_separate_views:
            metadata["reasoner_image_long_edge"] = int(service.cfg.reasoner_image_long_edge)
        metadata["reasoner_wam_mode"] = service.cfg.reasoner_wam_mode
        metadata["reasoner_include_preview_action"] = service.cfg.reasoner_include_preview_action
        metadata["reasoner_open_gripper_move_augment"] = service.cfg.reasoner_open_gripper_move_augment
        metadata["reasoner_require_think"] = service.cfg.reasoner_require_think
    server_cls(policy=service, host=args.host, port=int(args.port), metadata=metadata).serve_forever()


def main() -> None:
    cascade_subcommand_args = getattr(
        tyro.conf,
        "CascadeSubcommandArgs",
        tyro.conf.ConsolidateSubcommandArgs,
    )
    args = tyro_cli(
        RobolabServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            cascade_subcommand_args,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
