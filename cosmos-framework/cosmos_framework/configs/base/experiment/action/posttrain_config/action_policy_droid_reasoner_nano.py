# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone DROID subgoal Reasoner LoRA SFT experiment.

This keeps only the ordinary assistant-token CE branch from the joint
Reasoner/WAM recipe.  It reuses that recipe's Reasoner dataflow verbatim, but
runs it through the standard single-forward trainer and a LoRA-only optimizer.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_reasoner_wam_nano import (
    reasoner_loader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


action_policy_droid_reasoner_nano = copy.deepcopy(action_policy_droid_nano)
action_policy_droid_reasoner_nano.job.name = "action_policy_droid_reasoner_nano"

# Ordinary VLM assistant-only CE; no WAM flow forward or soft-plan decoder.
action_policy_droid_reasoner_nano.model.config.reasoner_ce_enabled = True
action_policy_droid_reasoner_nano.model.config.reasoner_ce_loss_weight = 1.0
action_policy_droid_reasoner_nano.model.config.reasoner_wam_two_forward = False
action_policy_droid_reasoner_nano.model.config.soft_plan_num_tokens = 0
action_policy_droid_reasoner_nano.model.config.mid_soft = False
action_policy_droid_reasoner_nano.model.config.subtask_decoder = {"enabled": False}

# QKVO rank 128 is about 122.7M trainable parameters for the Nano Qwen3-VL
# understanding tower, exactly twice the joint recipe's rank-64 adapters.  Do
# not target generic MLP child names here: the generation MLP has the same
# child names under ``mlp_moe_gen`` and would receive unused adapters too.
action_policy_droid_reasoner_nano.model.config.lora_enabled = True
action_policy_droid_reasoner_nano.model.config.lora_rank = 128
action_policy_droid_reasoner_nano.model.config.lora_alpha = 256
action_policy_droid_reasoner_nano.model.config.lora_target_modules = "q_proj,k_proj,v_proj,o_proj"
action_policy_droid_reasoner_nano.model.config.lora_soft_plan_only = False
action_policy_droid_reasoner_nano.model.config.lora_freeze_non_lora = True
action_policy_droid_reasoner_nano.model.config.ema.enabled = False

# A Reasoner batch has no raw VAE image/video key.  Keep generic grad-norm
# tracking and disable callbacks that are specific to action/video batches.
action_policy_droid_reasoner_nano.trainer.callbacks.grad_clip.track_per_modality = False
action_policy_droid_reasoner_nano.trainer.callbacks.action_input_video_logger.enabled = False
action_policy_droid_reasoner_nano.trainer.callbacks.gpu_image_augmentation.enabled = False
action_policy_droid_reasoner_nano.trainer.callbacks.reasoner_dataloader_state = L(
    CosmosDataLoaderStateCallback
)(name="reasoner")

action_policy_droid_reasoner_nano.optimizer.keys_to_select = ["lora_"]
action_policy_droid_reasoner_nano.optimizer.lr = 5.0e-05
action_policy_droid_reasoner_nano.optimizer.lr_multipliers = {}
action_policy_droid_reasoner_nano.optimizer.weight_decay = 0.0

# The base DCP supplies the text/generation weights.  Its visual tower is
# absent or stale, so the launcher overlays the local HF visual checkpoint.
action_policy_droid_reasoner_nano.checkpoint.keys_to_skip_loading = [
    "net_ema.",
    "visual",
    "lora_",
]
action_policy_droid_reasoner_nano.checkpoint.strict_resume = False

action_policy_droid_reasoner_nano.dataloader_train = copy.deepcopy(reasoner_loader)
action_policy_droid_reasoner_nano.dataloader_val = None

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_reasoner_nano",
    node=action_policy_droid_reasoner_nano,
)


__all__ = ["action_policy_droid_reasoner_nano"]
