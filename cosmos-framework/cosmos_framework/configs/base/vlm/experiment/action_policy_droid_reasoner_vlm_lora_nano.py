# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure-VLM DROID subgoal Reasoner with QKVO LoRA + visual-projector SFT."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.action_input_video_logger import ActionInputVideoLogger
from cosmos_framework.callbacks.cosmos_dataloader_state import (
    CosmosDataLoaderStateCallback,
    JointCosmosDataLoaderStateCallback,
)
from cosmos_framework.callbacks.reasoner_lora_export import ReasonerLoraExportCallback
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.data.vfm.action.datasets.droid_reasoner_wam_streams import (
    get_droid_reasoner_stream_dataset,
)
from cosmos_framework.data.vfm.action.droid_reasoner_wam_processors import DROIDReasonerSupervisionProcessor
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    JointCosmosDataLoader,
    RankPartitionedDistributor,
    SimpleBatcher,
)
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


reasoner_dataset = L(get_droid_reasoner_stream_dataset)(
    root="${oc.env:DROID_ROOT}",
    subtask_annotations_path="${oc.env:SUBTASK_ANNOTATIONS_PATH}",
    fps=15.0,
    chunk_length=32,
    mode="policy",
    viewpoint="concat_view",
    action_space="joint_pos",
    use_state=True,
    use_same_timestep_joint_action=True,
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
    use_all_data_with_cam2cam_fallback=True,
    subtask_episode_filter_path="${oc.env:SUBTASK_EPISODE_FILTER_PATH}",
    precise_subtask_annotations_path="${oc.env:PRECISE_SUBTASK_ANNOTATIONS_PATH}",
    include_idle_stationary_subtasks=False,
    remove_post_completion_windows=True,
    include_post_completion_move_away_windows=False,
    reasoner_history_offsets=(0,),
    reasoner_history_boundary="repeat_first",
    reasoner_history_choices=((0,), (-16, -8, 0)),
    reasoner_history_choice_weights=(1.0, 1.0),
    reasoner_history_choice_seed=43,
    rotation_aware_subgoal_degrees=60.0,
    max_subtasks_per_window=2,
    release_label_advance_frames=0,
    object_position_augment_prob=1.0,
    object_position_augment_style="random",
    object_position_augment_seed=42,
    execution_success_only=True,
    episode_split="train",
    validation_fraction=0.05,
    validation_split_seed=43,
    task_variant_policy="first",
    episode_shuffle_seed=43,
    resume_name="reasoner",
    exact_action_sampling=False,
    open_gripper_sampling_ratio=0.0,
    position_augment_priority=False,
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
    batcher=L(SimpleBatcher)(batch_size=16, drop_last=True),
    collator=L(VLMCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)

# Validation uses the same annotation source and preprocessing as training but
# a disjoint, stable episode-level split. The stream is intentionally restarted
# from the same order at every validation call; trainer.max_val_iter bounds it.
reasoner_val_dataset = copy.deepcopy(reasoner_dataset)
reasoner_val_dataset.episode_split = "val"
reasoner_val_dataset.resume_name = ""

reasoner_val_loader = copy.deepcopy(reasoner_loader)
reasoner_val_loader.distributor.datasets.reasoner.dataset = reasoner_val_dataset
reasoner_val_loader.persistent_workers = False


action_policy_droid_reasoner_vlm_lora_nano = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "local"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm"]},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action",
            group="action_sft_droid",
            name="droid_subgoal_reasoner_vlm_lora_qkvo_r128",
            wandb_mode="online",
        ),
        model=dict(
            config=dict(
                lora_enabled=True,
                lora_rank=128,
                lora_alpha=256,
                lora_target_modules="q_proj,k_proj,v_proj,o_proj",
                # Preserve the projector long enough for VLMModel to select it;
                # VLMModel then freezes every non-LoRA tensor except both Qwen
                # visual merger families before FSDP/optimizer construction.
                lora_freeze_non_lora=False,
                ema=dict(enabled=False),
                activation_checkpointing=dict(mode="full"),
                parallelism=dict(
                    data_parallel_shard_degree=8,
                    data_parallel_replicate_degree=-1,
                ),
                policy=dict(
                    backbone=dict(
                        model_name="cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json",
                        safetensors_path="${oc.env:REASONER_PROCESSOR_PATH}",
                        pretrained_weights=dict(
                            backbone_path="${oc.env:BASE_CHECKPOINT_PATH}",
                            checkpoint_format="cosmos3_dcp",
                        ),
                    ),
                ),
            ),
        ),
        trainer=dict(
            max_iter=10000,
            logging_iter=50,
            run_validation=True,
            validation_iter=500,
            max_val_iter=50,
            run_validation_on_start=False,
            grad_accum_iter=1,
            callbacks=dict(
                reasoner_dataloader_state=L(CosmosDataLoaderStateCallback)(name="reasoner"),
                reasoner_lora_export=L(ReasonerLoraExportCallback)(),
                reasoner_sample_logger=L(ActionInputVideoLogger)(
                    every_n=200,
                    log_first=True,
                    include_reasoner_vlm=True,
                    log_reasoner_immediately=True,
                    max_reasoner_vlm_samples=4,
                ),
                grad_clip=dict(track_per_modality=False),
            ),
        ),
        optimizer=dict(
            lr=5.0e-5,
            fused=True,
            weight_decay=0.0,
            betas=[0.9, 0.99],
            keys_to_select=["lora_", "visual.merger", "visual.deepstack_merger_list"],
            lr_multipliers={},
        ),
        scheduler=dict(
            warm_up_steps=[100],
            cycle_lengths=[10000],
            f_start=[0.0],
            f_max=[1.0],
            f_min=[0.0],
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            load_path="",
            keys_to_skip_loading=[],
            save_iter=100,
            max_to_keep=2,
        ),
        upload_reproducible_setup=False,
        dataloader_train=reasoner_loader,
        dataloader_val=reasoner_val_loader,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_reasoner_vlm_lora_nano",
    node=action_policy_droid_reasoner_vlm_lora_nano,
)


# Optional rationale replay keeps ordinary, family-stratified subgoal batches
# intact and alternates them with homogeneous start-frame rationale batches.
# Separate inner loaders are intentional: each source retains an independent
# episode-shuffled cursor, and no batch mixes direct and <think> targets.
rationale_dataset = copy.deepcopy(reasoner_dataset)
rationale_dataset.rationale_annotations_path = "${oc.env:REASONER_RATIONALE_ANNOTATIONS_PATH}"
rationale_dataset.rationale_only = True
rationale_dataset.rationale_strict_window_match = False
rationale_dataset.task_variant_policy = "object_match"
rationale_dataset.stratified_sampling = False
rationale_dataset.resume_name = "rationale"

rationale_loader = copy.deepcopy(reasoner_loader)
rationale_loader.distributor.datasets = {
    "rationale": {"dataset": rationale_dataset, "ratio": 1},
}

reasoner_rationale_loader = L(JointCosmosDataLoader)(
    dataloaders={
        "reasoner": {"dataloader": reasoner_loader, "ratio": 8.0},
        "rationale": {"dataloader": rationale_loader, "ratio": 2.0},
    },
    seed=43,
)

action_policy_droid_reasoner_rationale_vlm_lora_nano = copy.deepcopy(
    action_policy_droid_reasoner_vlm_lora_nano
)
action_policy_droid_reasoner_rationale_vlm_lora_nano.job.name = (
    "droid_subgoal_reasoner_rationale_vlm_lora_qkvo_r128"
)
action_policy_droid_reasoner_rationale_vlm_lora_nano.dataloader_train = reasoner_rationale_loader
action_policy_droid_reasoner_rationale_vlm_lora_nano.trainer.callbacks.reasoner_dataloader_state = L(
    JointCosmosDataLoaderStateCallback
)(dataset_names=("reasoner", "rationale"))

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_reasoner_rationale_vlm_lora_nano",
    node=action_policy_droid_reasoner_rationale_vlm_lora_nano,
)


__all__ = [
    "action_policy_droid_reasoner_rationale_vlm_lora_nano",
    "action_policy_droid_reasoner_vlm_lora_nano",
]
