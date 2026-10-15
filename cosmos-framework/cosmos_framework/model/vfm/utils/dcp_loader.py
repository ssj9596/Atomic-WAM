# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Checkpoint utilities for VFM models.
Contains utilities for loading model checkpoints with key remapping.
"""

import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader
from cosmos_framework.utils import log

_COSMOS3_OMNI_LM_PREFIX = "net.language_model."


def _resolve_dcp_model_path(checkpoint_path: str) -> str:
    """Resolve a Cosmos3 checkpoint root or its nested ``model`` directory."""

    path = Path(checkpoint_path)
    if (path / ".metadata").is_file():
        return str(path)
    nested = path / "model"
    if (nested / ".metadata").is_file():
        return str(nested)
    raise FileNotFoundError(
        f"Cosmos3 DCP model metadata was not found under {checkpoint_path!r}; expected .metadata or model/.metadata."
    )


def _omni_dcp_source_for_qwen3_vl_target(target_name: str) -> str | None:
    """Map a pure Qwen3-VL state key to the Cosmos3 Omni DCP language key."""

    normalized = target_name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    if ".lora_A." in normalized or ".lora_B." in normalized:
        return None
    if normalized == "lm_head.weight":
        return f"{_COSMOS3_OMNI_LM_PREFIX}lm_head.weight"
    for prefix in ("model.language_model.", "language_model."):
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if suffix.startswith("model."):
                suffix = suffix[len("model.") :]
            return f"{_COSMOS3_OMNI_LM_PREFIX}model.{suffix}"
    return None


def load_qwen3_vl_reasoner_from_cosmos3_dcp(model: torch.nn.Module, checkpoint_path: str) -> set[str]:
    """Load only the Cosmos3 Reasoner/language pathway into a pure Qwen3-VL.

    The Cosmos3-Nano DCP also contains Generator experts and action heads. They
    are deliberately absent from a standalone VLM and are ignored here. Vision
    weights are stored outside this DCP in the release's ``vision_encoder``
    safetensors and must be loaded separately.
    """

    resolved_path = _resolve_dcp_model_path(checkpoint_path)
    storage_reader = dcp.FileSystemReader(resolved_path)
    metadata = storage_reader.read_metadata()
    source_keys = set(metadata.state_dict_metadata)

    state_to_load: dict[str, torch.Tensor] = {}
    target_names_by_source: dict[str, str] = {}
    for target_name, tensor in model.state_dict().items():
        source_name = _omni_dcp_source_for_qwen3_vl_target(target_name)
        if source_name is None:
            continue
        if source_name not in source_keys:
            raise ValueError(
                f"Cosmos3 DCP {resolved_path!r} is missing Reasoner tensor {source_name!r} "
                f"required by pure-VLM tensor {target_name!r}."
            )
        state_to_load[source_name] = tensor
        target_names_by_source[source_name] = target_name

    if not state_to_load:
        raise ValueError("The target model exposes no Qwen3-VL Reasoner tensors to load from Cosmos3 DCP.")

    start_time = time.time()
    log.info(f"Loading pure-VLM Reasoner weights directly from Cosmos3 DCP: {resolved_path}")
    dcp.load(
        state_dict=state_to_load,
        storage_reader=storage_reader,
        planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
    )
    loaded = {target_names_by_source[source_name] for source_name in state_to_load}
    log.info(
        f"Loaded {len(loaded)} pure-VLM Reasoner tensors directly from Cosmos3 DCP in {time.time() - start_time:.1f}s"
    )
    return loaded


class RenameLoadPlanner(dcp.DefaultLoadPlanner):
    """
    RenameLoadPlanner that does two renaming operations during pretrained model load.
    1. Renames model parameters from "model." to "model.language_model." if the core model is a VLM.
    2. Renames model parameters from "_orig_mod." to "." since the pretrained model does not have torch compiled modules.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        state_dict = remove_orig_mode_from_state_dict(state_dict)

        # Renames model parameters from "model." to "model.language_model." if the core model is a VLM.
        # This is necessary because the checkpoint is saved with "model.language_model." for a VLM.
        # If the core model is an LLM, this renaming is not necessary.
        has_language_model = any("model.language_model." in key for key in metadata.state_dict_metadata)
        if has_language_model:
            state_dict = insert_language_model_in_state_dict(state_dict)

        # Perform more error checking to ensure the checkpoint is valid. If the keys are missing,
        # then the missing keys should be from the generation pathway. All keys from the
        # understanding pathway must be present in the checkpoint. Additionally, for 2B and 4B
        # dense Qwen VLMs, the `lm_head.weight` key is not present in the checkpoint. For these
        # models, the input embedding and generation layer share the same params due to
        # `tie_word_embeddings` being set to True in the configs. For the 0.6B LLM, 8B and 32B dense
        # VLMs, and the 30B and 235B MoE VLMs, the `lm_head.weight` key is present in the
        # checkpoint.
        for key in state_dict:
            if key not in metadata.state_dict_metadata and "_moe_gen" not in key and "lm_head.weight" not in key:
                raise ValueError(f"Missing key: {key} in checkpoint.")

        # If the keys are unexpected, then the unexpected keys should be from the visual part of the
        # VLM. All keys in the language part should be used by the Cosmos3 model.
        for key in metadata.state_dict_metadata:
            if key not in state_dict and "model.visual." not in key:
                raise ValueError(f"Unexpected key: {key} in checkpoint.")

        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )


def remove_orig_mode_from_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Renames model parameters from "_orig_mod." to "." since the pretrained model does not have torch compiled modules."""
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def insert_language_model_in_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Renames model parameters from "model." to "model.language_model." if the core model is a VLM."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key.replace("model.", "model.language_model.")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_language_model(
    model: torch.nn.Module, checkpoint_path: str, credential_path: str, enable_gcs_patch_in_boto3: bool = False
) -> None:
    """
    Universal language model loading function using DCP format.
    Handles key remapping for "model.language_model." -> "model." by default.

    Args:
        model: The language model to load weights into
        checkpoint_path: Path to checkpoint (must end with .safetensors or dcp)
        credential_path: Path to S3 credentials
        enable_gcs_patch_in_boto3: Whether to enable GCS patch in boto3 for DCP loading from GCS
    """
    start_time = time.time()
    if not checkpoint_path.strip("/").endswith("dcp"):
        raise ValueError(f"Checkpoint path {checkpoint_path} must end with dcp")

    log.info(f"Loading language model weights in DCP format from: {checkpoint_path}")
    state_dict = model.state_dict()

    # Choose storage reader
    if checkpoint_path.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=credential_path,
            path=checkpoint_path,
            enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
        )
    else:
        storage_reader = dcp.FileSystemReader(checkpoint_path)

    # Load checkpoint
    dcp.load(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=RenameLoadPlanner(allow_partial_load=False),
    )

    log.info(f"Successfully loaded language model from {checkpoint_path}")
    log.info(f"Time taken to load language model: {time.time() - start_time} seconds")
