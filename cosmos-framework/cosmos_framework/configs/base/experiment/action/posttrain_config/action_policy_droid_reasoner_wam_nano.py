# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DROID ordinary-Reasoner CE + WAM flow joint SFT experiment.

Each logical step consumes one independently shuffled Reasoner batch and one
independently shuffled WAM batch.  The specialized trainer performs two
forwards/backwards and updates disjoint LoRA/WAM optimizers.

This experiment is opt-in and independent from both mid-soft recipes.  It uses
the ordinary VLM ``lm_head`` for Reasoner CE and never enables the soft-plan
``subtask_decoder``.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.reasoner_wam_dataloader_state import ReasonerWAMPairedDataLoaderStateCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.data.vfm.action.datasets.droid_reasoner_wam_streams import (
    get_droid_reasoner_stream_dataset,
    get_droid_wam_stream_dataset,
)
from cosmos_framework.data.vfm.action.droid_reasoner_wam_processors import (
    DROIDReasonerSupervisionProcessor,
    DROIDWAMStreamProcessor,
)
from cosmos_framework.data.vfm.action.reasoner_wam_paired_loader import ReasonerWAMPairedDataLoader
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    RankPartitionedDistributor,
    SequentialPackingBatcher,
    SimpleBatcher,
    VFMListCollator,
)
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.trainer.reasoner_wam import ReasonerWAMTrainer
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


def _common_raw_dataset_kwargs() -> dict:
    """Settings shared by Reasoner and WAM so their window sets are equal."""

    return dict(
        root="${oc.env:DROID_ROOT}",
        subtask_annotations_path="${oc.env:SUBTASK_ANNOTATIONS_PATH}",
        fps=15.0,
        chunk_length=32,
        mode="policy",
        viewpoint="concat_view",
        action_space="joint_pos",
        use_state=True,
        use_same_timestep_joint_action=False,
        action_normalization=None,
        use_image_augmentation=False,
        use_filter_dict=True,
        filter_dict_path="${oc.env:KEEP_RANGES_PATH}",
        prefilter_by_filter_dict=True,
        camera_order_path=None,
        cam2base_extrinsic_path="${oc.env:CAM2BASE_EXTRINSIC_PATH}",
        success_only=True,
        video_decode_backend="torchcodec",
        wrist_rotate_180=True,
        video_sample_stride=4,
        use_all_data_with_cam2cam_fallback=False,
        # Kept None in the persisted recipe; the launcher supplies the same
        # v2 JSON as SUBTASK_ANNOTATIONS_PATH when the subfilter is enabled.
        subtask_episode_filter_path=None,
        precise_subtask_annotations_path="${oc.env:PRECISE_SUBTASK_ANNOTATIONS_PATH}",
    )


reasoner_dataset = L(get_droid_reasoner_stream_dataset)(
    **_common_raw_dataset_kwargs(),
    rotation_aware_subgoal_degrees=None,
    max_subtasks_per_window=100,
    release_label_advance_frames=0,
    remove_post_completion_windows=False,
    include_post_completion_move_away_windows=False,
    object_position_augment_prob=0.0,
    object_position_augment_style="random",
    object_position_augment_seed=42,
    episode_shuffle_seed=43,
    # Mix one current image and a causal three-image history at equal weight.
    # The three-image choice uses a nominal eight-frame spacing: (-16, -8, 0).
    # Dataset boundary handling deterministically shrinks this to [0,3,6],
    # [0,1,1], or [0,0,0] when the full history is unavailable.
    reasoner_history_offsets=(0,),
    reasoner_history_boundary="repeat_first",
    reasoner_history_choices=((0,), (-16, -8, 0)),
    reasoner_history_choice_weights=(1.0, 1.0),
    reasoner_history_choice_seed=43,
    exact_action_sampling=False,
    task_variant_policy="first",
)

wam_dataset = L(get_droid_wam_stream_dataset)(
    **_common_raw_dataset_kwargs(),
    # Match the ordinary DROID action/mid-soft permutation.  The launcher
    # keeps this overrideable, while the Reasoner stream uses its independent
    # seed (43) so the two forwards do not consume the same windows.
    episode_shuffle_seed=42,
    use_exterior_plucker_condition=False,
    defer_plucker_generation=False,
    # Mode-2 WAM conditioning: gold subgoal vs. task at 8:2.  The factory's
    # source order is (task, gold_subgoal), hence weights (2, 8).
    condition_sources=("task", "gold_subgoal"),
    condition_weights=(2.0, 8.0),
    condition_seed=42,
    # Opt-in WAM-only augmentation; Reasoner assistant labels stay canonical.
    object_position_augment_prob=0.0,
    object_position_augment_style="random",
    object_position_augment_seed=42,
    # Legacy-safe defaults: the existing experiment still applies the six-rule
    # filter to the WAM index and does not repeat grip windows.  New full-WAM
    # experiments override these two fields from the launcher.
    apply_subtask_episode_filter_to_windows=True,
    primary_grip_oversample_factor=1.0,
    primary_grip_oversample_seed=42,
    resolution="544x640",
    max_action_dim="${model.config.max_action_dim}",
    tokenizer_config="${model.config.vlm_config.tokenizer}",
    cfg_dropout_rate=0.1,
    camera_cfg_dropout_rate=0.0,
    append_viewpoint_info=True,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    append_idle_frames=False,
    include_vit_first_frame=True,
    vit_processor_model_name="${oc.env:REASONER_PROCESSOR_PATH}",
    vit_processor_config_variant="hf",
    vit_processor_cache_dir=None,
    vit_first_frame_max_pixels=348160,
    vit_first_frame_min_pixels=None,
    defer_vit_pixel_values=False,
    vit_frame_offsets=(0,),
)

reasoner_loader = L(CosmosDataLoader)(
    distributor=L(RankPartitionedDistributor)(
        datasets={"reasoner": {"dataset": reasoner_dataset, "ratio": 1}},
    ),
    processor=L(DROIDReasonerSupervisionProcessor)(
        processor=L(VLMProcessor)(
            processor=L(build_processor)(
                tokenizer_type="${oc.env:REASONER_PROCESSOR_PATH}",
                config_variant="hf",
            ),
        ),
    ),
    # The paired trainer consumes one complete Reasoner batch and one complete
    # WAM batch per logical step.  This child loader has its own batch size;
    # there is no 3:7 ratio scheduler in the current mode.
    batcher=L(SimpleBatcher)(batch_size=16, drop_last=True),
    collator=L(VLMCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)

wam_loader = L(CosmosDataLoader)(
    distributor=L(RankPartitionedDistributor)(
        datasets={"wam": {"dataset": wam_dataset, "ratio": 1}},
    ),
    processor=L(DROIDWAMStreamProcessor)(),
    batcher=L(SequentialPackingBatcher)(
        max_sequence_length=None,
        max_samples_per_batch=16,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        lookahead_limit=20,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    ),
    collator=L(VFMListCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)


action_policy_droid_reasoner_wam_nano = copy.deepcopy(action_policy_droid_nano)
action_policy_droid_reasoner_wam_nano.job.name = "action_policy_droid_reasoner_wam_nano"
action_policy_droid_reasoner_wam_nano.trainer.type = ReasonerWAMTrainer

# Ordinary assistant-only VLM CE.  Keep the Reasoner coefficient at the standard
# unit weight; it remains independently tunable from the WAM flow scales.
action_policy_droid_reasoner_wam_nano.model.config.reasoner_ce_enabled = True
action_policy_droid_reasoner_wam_nano.model.config.reasoner_ce_loss_weight = 1.0
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_two_forward = True
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_vlm_max_steps = -1
# Keep the established behavior by default: WAM sees the current base+LoRA
# VLM features.  The launcher can override this to false for the frozen/base
# feature ablation without changing the Reasoner phase.
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_wam_use_lora = True
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_reasoner_keys = ["lora_"]
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_wam_keys = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
action_policy_droid_reasoner_wam_nano.model.config.reasoner_wam_wam_exclude_keys = ["lora_"]
action_policy_droid_reasoner_wam_nano.model.config.soft_plan_num_tokens = 0
action_policy_droid_reasoner_wam_nano.model.config.mid_soft = False
action_policy_droid_reasoner_wam_nano.model.config.subtask_decoder = {"enabled": False}

# Ordinary VLM QKVO LoRA plus the existing WAM/video/action modules.  Setting
# freeze_non_lora=False preserves generation-module trainability.
action_policy_droid_reasoner_wam_nano.model.config.lora_enabled = True
action_policy_droid_reasoner_wam_nano.model.config.lora_rank = 64
action_policy_droid_reasoner_wam_nano.model.config.lora_alpha = 128
action_policy_droid_reasoner_wam_nano.model.config.lora_target_modules = "q_proj,k_proj,v_proj,o_proj"
action_policy_droid_reasoner_wam_nano.model.config.lora_soft_plan_only = False
action_policy_droid_reasoner_wam_nano.model.config.lora_freeze_non_lora = False
action_policy_droid_reasoner_wam_nano.model.config.tokenizer.encode_exact_durations = [9]

# The inherited VFM grad-clip diagnostic calls ``model.is_image_batch`` at
# step start.  A standalone Reasoner batch has ``input_ids``/``pixel_values``
# but deliberately no raw VAE ``images``/``video`` key, so use the generic
# (unbucketed) norm logger for this heterogeneous recipe.  Clipping itself is
# unchanged and still applies to every parameter that has a gradient.
action_policy_droid_reasoner_wam_nano.trainer.callbacks.grad_clip.track_per_modality = False
action_policy_droid_reasoner_wam_nano.trainer.callbacks.reasoner_wam_dataloader_state = L(
    ReasonerWAMPairedDataLoaderStateCallback
)()
# Opt in to the bounded Reasoner image/task/gold-subgoal sample.  The base
# action recipe keeps this logger disabled for legacy/mid-soft runs.
action_policy_droid_reasoner_wam_nano.trainer.callbacks.action_input_video_logger.include_reasoner_vlm = True

action_policy_droid_reasoner_wam_nano.optimizer.keys_to_select = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "lora_",
]
action_policy_droid_reasoner_wam_nano.optimizer.lr_multipliers["lora_"] = 0.25
action_policy_droid_reasoner_wam_nano.checkpoint.keys_to_skip_loading = [
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
    "visual",
    "lora_",
]

# One pair per logical step; the child loaders retain independent worker/episode
# shuffles.  The paired loader deliberately does not use the legacy generic
# ratio scheduler, so the two forwards always have homogeneous schemas.
action_policy_droid_reasoner_wam_nano.dataloader_train = L(ReasonerWAMPairedDataLoader)(
    reasoner_loader=reasoner_loader,
    wam_loader=wam_loader,
)
action_policy_droid_reasoner_wam_nano.dataloader_val = None

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_reasoner_wam_nano",
    node=action_policy_droid_reasoner_wam_nano,
)


__all__ = ["action_policy_droid_reasoner_wam_nano"]
