# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any

import attrs

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.ema import EMAConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.vlm import VLMConfig
from cosmos_framework.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class DiffusionExpertConfig:
    # This determines the range of timesteps before the fourier feature embedding is applied.
    timestep_range: float = 1.0
    # Whether to load the generation pathway weights from pretrained LLM/VLM weights.
    load_weights_from_pretrained: bool = True

    patch_spatial: int = 2
    max_vae_latent_side_after_patchify: int = (
        20  # Max dimension (h or w) of the VAE latent after patchification (320/(8*2))
    )
    # Position embedding type for vision tokens:
    #   - "3d_rope": Additive 3D RoPE embeddings (VideoRopePosition3DEmb) + 1D position IDs for attention
    #   - "flattened_sin_cos": Additive flattened sin/cos embeddings + 1D position IDs for attention
    #   - "unified_3d_mrope": No additive embedding + 3D position IDs for Qwen3VL-style mRoPE attention
    position_embedding_type: str = "3d_rope"
    # When finetuning from lower resolution to higher resolution, the spatial resolution of videos increase.
    # So, we need to adjust the position embedding.
    # We use NTK based RoPE extrapolation to adjust the position embedding.
    # Reference: (https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
    # Design adapted from Cosmos2.5 (https://arxiv.org/pdf/2511.00062)
    # extrapolation_ratio here is how the base of the RoPE is scaled
    # b' = b * extrapolation_ratio^(dim / (dim - 2))
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    enable_fps_modulation: bool = False
    base_fps: int = 24
    # Base temporal compression factor for SOUND m-RoPE. None = current behavior
    # (sound advances at base_fps positions/sec). Set to the vision tcf (4) to put
    # sound on the same latent-frame temporal grid as vision/action.
    sound_base_temporal_compression_factor: int | None = None
    # Temporal coordinates used for unified_3d_mrope vision tokens.
    # - "latent_index": legacy behavior, positions are 0, 1, ..., T_latent-1.
    # - "uniae_source_right_edge": use UniAE padded-patch right-edge source-frame coordinates.
    vision_temporal_position_mode: str = "latent_index"
    # For unified_3d_mrope: whether spatial (H, W) indices reset to 0 for each vision segment
    unified_3d_mrope_reset_spatial_ids: bool = True
    # Setting the temporal gap on the boundary of the different modalities, default is 0, using a value greater than 0 will add an additional offset on the accumulated temporal offset.
    unified_3d_mrope_temporal_modality_margin: int = 0


@attrs.define(slots=False)
class LBLConfig:
    # For load balancing loss computation.
    # - "local": Use the fraction of tokens routed to each expert only for the local rank.
    # - "global": Use the fraction of tokens routed to each expert across all ranks.
    method: str = "local"

    # Coefficients for the load balancing loss.
    # - "und": Coefficient for the load balancing loss for the "und" pathway.
    # - "gen": Coefficient for the load balancing loss for the "gen" pathway.
    coeff_und: float | None = None
    coeff_gen: float | None = None


@attrs.define(slots=False)
class RectifiedFlowTrainingConfig:
    shift: Any = 5  # Training time shift. If dict, maps resolution (str) to shift value (int)
    use_dynamic_shift: bool = False  # Whether to use dynamic shifting
    train_time_image_distribution: str = "logitnormal"  # Training time distribution for images
    train_time_video_distribution: str = "logitnormal"  # Training time distribution for videos
    train_time_action_distribution: str = "logitnormal"  # Training time distribution for actions
    train_time_sound_distribution: str = "logitnormal"  # Training time distribution for sound
    train_time_weight: str = "uniform"  # Training time weight
    loss_scale: float = 1.0  # Loss scale
    image_loss_scale: float | None = None  # If set, overrides loss_scale for images
    sound_loss_scale: float | None = None  # If set, overrides loss_scale for sound
    use_discrete_rf: bool = False  # Whether to use discrete formulation of rectified flow

    # user: please adjust this value according to loss_scale to balance the action loss with the video loss.
    # default is 10.0 to align with previous training settings.
    action_loss_weight: float = 10.0

    # Independent noise schedule for action. When False (default), action shares the sigma
    # sampled from the vision RF on every step — legacy behavior. When True, action samples
    # its own sigma from `rectified_flow_action` using `shift_action`. Action always uses a
    # shared scalar sigma per sample ([B,1]), independent of vision's DF mode.
    independent_action_schedule: bool = False
    shift_action: int | None = None  # must be int; None → inherit `shift` (which must also be int)

    # Independent noise schedule for sound. When False (default), sound shares the vision
    # sigma schedule, reindexed to the dense audio-bearing subset. When True, sound samples
    # its own scalar sigma per sample ([B,1]) from `rectified_flow_sound` using `shift_sound`.
    independent_sound_schedule: bool = False
    shift_sound: int | None = None  # must be int; None → inherit `shift` (which must also be int)

    # When True, per-instance flow-matching loss is normalized by the count of
    # active (noisy) elements rather than all elements — preserves sum/active_count
    # semantics so conditioning-heavy samples (e.g. I2V, forward_dynamics, diffusion
    # forcing, AR rollout teacher-forcing) contribute gradient on par with K=0
    # samples. With .mean() the gradient of a K-conditioned sample is scaled by
    # (T-K)/T, which undertrains the attend-to-clean-history dynamics. Kept
    # False by default to preserve legacy loss magnitudes; enable for AR/DF training.
    normalize_loss_by_active: bool = False


@attrs.define(slots=False)
class RectifiedFlowInferenceConfig:
    scheduler_type: str = "unipc"  # Scheduler type
    num_train_timesteps: int = 1000
    shift: int = 1
    use_dynamic_shifting: bool = False


@attrs.define(slots=False)
class FixedStepSamplerConfig:
    """Config for the fixed-step sampler used by distilled models.

    Uses a fixed sigma schedule instead of a smooth multi-step solver.

    Mirrors the constructor args of ``FixedStepSampler``.
    """

    # Discrete noise-level schedule (descending, excluding the final 0.0 step).
    # Convention: exclude the final 0.0 step — FixedStepSampler appends it automatically.
    # Values must be descending. Using 0.999 instead of 1.0 avoids numeric edge cases at sigma=1.
    t_list: list[float] = [0.999, 0.75, 0.5, 0.25]
    # Integrator type: "ode" (deterministic Euler) or "sde" (stochastic re-noising at each step).
    sample_type: str = "ode"


@attrs.define(slots=False)
class SubtaskDecoderConfig:
    """Training-only SIM-CoT-style supervision for final soft-plan states."""

    enabled: bool = False
    loss_weight: float = attrs.field(default=1.0, validator=attrs.validators.ge(0.0))
    hidden_dim: int = attrs.field(default=512, validator=attrs.validators.gt(0))
    num_layers: int = attrs.field(default=2, validator=attrs.validators.gt(0))
    num_heads: int = attrs.field(default=8, validator=attrs.validators.gt(0))
    ffn_dim: int = attrs.field(default=2048, validator=attrs.validators.gt(0))
    dropout: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.lt(1.0)),
    )
    # Maximum number of gold subtask tokens per sample, including EOS.
    max_tokens: int = attrs.field(default=128, validator=attrs.validators.gt(0))

    def __attrs_post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "subtask_decoder.hidden_dim must be divisible by subtask_decoder.num_heads, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )


# Don't have any defaults and init only in config file.
@attrs.define(slots=False)
class OmniMoTModelConfig:
    """
    Config for Omni MoT model.
    """

    tokenizer: LazyDict = None
    net: LazyDict = None
    ema: EMAConfig = EMAConfig()

    # Parallelism (CP, CFGP, FSDP, DP) and FSDP reduce-dtype configuration.
    parallelism: ParallelismConfig = ParallelismConfig()

    # torch.compile knobs (enabled, compiled_region, dynamic, ...).
    compile: CompileConfig = CompileConfig()

    # Activation-checkpointing policy (trade-off between memory and speed).
    activation_checkpointing: ActivationCheckpointingConfig = ActivationCheckpointingConfig()

    # Model parameter / activation dtype (consumed by MixedPrecisionPolicy and
    # ``model.precision`` for LowPrecisionCallback). One of "bfloat16",
    # "float16", "float32".
    precision: str = "bfloat16"

    # LoRA (parameter-efficient fine-tuning). When `lora_enabled=True`,
    # `OmniMoTModel.build_net` injects custom LoRA adapters BEFORE FSDP wrap on
    # the meta-device network, then re-initializes lora_A/lora_B after
    # to_empty + init_weights. Pair with `optimizer.keys_to_select=["lora_"]`
    # and `checkpoint.keys_to_skip_loading=[..., "lora_"]`.
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    # When enabled, understanding-path LoRA deltas are routed only to soft-plan
    # token rows. Raw VLM/image/text rows use the frozen base projections.
    lora_soft_plan_only: bool = False
    # Keep legacy LoRA recipes adapter-only by default. Hybrid recipes can
    # preserve existing trainable flags and let optimizer selection train LoRA
    # together with full world/action modules.
    lora_freeze_non_lora: bool = True

    # Rectified flow configs
    rectified_flow_training_config: RectifiedFlowTrainingConfig = RectifiedFlowTrainingConfig()
    rectified_flow_inference_config: RectifiedFlowInferenceConfig = RectifiedFlowInferenceConfig()

    # Optional fixed-step sampler for distilled models (None for base models).
    fixed_step_sampler_config: FixedStepSamplerConfig | None = None

    # Model configs
    vlm_config: VLMConfig = VLMConfig()
    diffusion_expert_config: DiffusionExpertConfig = DiffusionExpertConfig()
    # Training data keys
    input_video_key: str = "video"
    input_image_key: str = "images"  # key to fetch input image from data_batch
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    # Opt-in DROID/RoboLab camera-ray residual. False preserves legacy model
    # structure and checkpoint behavior exactly.
    exterior_plucker_condition_enabled: bool = False
    plucker_condition_dim: int = attrs.field(default=6, validator=attrs.validators.gt(0))
    plucker_condition_hidden_dim: int = attrs.field(default=128, validator=attrs.validators.gt(0))

    # State and sequence shapes
    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    latent_downsample_factor: int = 8
    resolution: str = "512"
    max_num_tokens_after_packing: int = 13312  # Final num tokens after sequence packing

    # Attention implementation for joint understanding + generation
    # Note "two_way" and "three_way" disallow and remove "End-of-Vision" or other text token in the generation tower.
    # "three_way" must only be used when introducing sparsity
    joint_attn_implementation: str = "two_way"  # "two_way" or "three_way"

    # Opt-in semantic bottleneck inserted after SOG. Zero preserves the legacy
    # packed sequence and attention path exactly.
    soft_plan_num_tokens: int = attrs.field(default=0, validator=attrs.validators.ge(0))

    # Opt-in mid-sequence soft-plan layout.  ``False`` preserves the legacy
    # ``[text, SOG, P, generation]`` packing and checkpoint behavior.
    mid_soft: bool = False

    # Optional training-only decoder that reads the final-layer soft-plan
    # hidden states and reconstructs a gold textual subtask. Disabled by
    # default so legacy checkpoints/configs keep exactly the old model shape.
    subtask_decoder: SubtaskDecoderConfig = SubtaskDecoderConfig()

    # Opt-in ordinary VLM/reasoner CE branch used by the joint reasoner+WAM
    # recipe.  It consumes assistant-only labels from the VLM processor and
    # calls the language model's native ``lm_head``; it is independent of the
    # soft-plan ``subtask_decoder`` above.  Defaults preserve legacy flow-only
    # training and old persisted configs.
    reasoner_ce_enabled: bool = False
    reasoner_ce_loss_weight: float = attrs.field(default=1.0, validator=attrs.validators.ge(0.0))

    # Opt-in two-forward Reasoner/WAM training.  Defaults preserve the legacy
    # single-optimizer flow and are intentionally explicit for old persisted
    # configs/export paths.
    reasoner_wam_two_forward: bool = False
    # Number of logical paired steps that include the Reasoner CE phase.
    # ``-1`` keeps CE enabled for the whole run; a non-negative value freezes
    # the Reasoner phase after that step while WAM continues training.
    reasoner_wam_vlm_max_steps: int = attrs.field(default=-1, validator=attrs.validators.ge(-1))
    reasoner_wam_reasoner_lr: float | None = None
    reasoner_wam_wam_lr: float | None = None
    reasoner_wam_reasoner_keys: list[str] = attrs.field(factory=lambda: ["lora_"])
    reasoner_wam_wam_keys: list[str] = attrs.field(
        factory=lambda: [
            "moe_gen",
            "time_embedder",
            "vae2llm",
            "llm2vae",
            "action2llm",
            "llm2action",
            "action_modality_embed",
        ]
    )
    reasoner_wam_wam_exclude_keys: list[str] = attrs.field(factory=lambda: ["lora_"])
    # Whether the WAM forward should include the currently trained VLM LoRA
    # adapters in its understanding features.  ``True`` preserves the
    # original Reasoner-WAM behavior (base + LoRA features); setting it to
    # ``False`` is an opt-in ablation that evaluates WAM with base-VLM
    # features while keeping the Reasoner phase's LoRA update enabled.
    reasoner_wam_wam_use_lora: bool = True

    # "legacy_full": current behavior; action, future video, and first-frame
    # video tokens are in one bidirectional full split.
    # "fastwam_first_frame": opt-in FastWAM-compatible two-way mask where action
    # tokens can attend text + first-frame video + action tokens, but not future
    # video latents. First-frame video tokens are also prevented from attending
    # future video/action tokens so multi-layer action outputs stay equivalent to
    # the action-only path.
    # "fastwam_first_last_frame": same as fastwam_first_frame, but action tokens
    # may also attend the last video latent frame. The action-only inference pack
    # keeps the last-frame tokens with their original last-frame RoPE IDs.
    action_video_attention_mode: str = attrs.field(
        default="legacy_full",
        validator=attrs.validators.in_({"legacy_full", "fastwam_first_frame", "fastwam_first_last_frame"}),
    )

    # Per-layer NATTEN parameters
    # Must use "three_way" attention if used.
    # If None, all attention layers remain dense.
    # If not None, must be a list exactly the size of number of layers, and each layer can be either
    # None (dense) or a dictionary, with at least 'kernel_size' or 'kernel_size_float' keys
    # specifying sparsity. NATTEN parameters 'dilation' and 'stride' may also be specified either as
    # static integers, or as floating point values that will be mapped to their domain during
    # runtime. Integer parameters should never be mixed with floating point ones.
    #
    # Floating point parameters are highly recommended, unless the use case will have a fixed token
    # layout (input resolution).
    #
    # Examples:
    #   Interleaved sliding window layers, "GPT-OSS"-style, with static window size:
    #     natten_parameter_list = [None if layer_idx % 2 != 0 else {"kernel_size": (8, 8)}]
    #   Layers with odd indices ("None"s) will use dense attention, and layers with an even indices
    #   will use a static sliding window size of 8x8.
    #
    #   Interleaved sliding window layers, "GPT-OSS"-style, with input-dependent window size:
    #     natten_parameter_list = [None if layer_idx % 2 != 0 else {"kernel_size_float": (0.5, 0.5)}]
    #   Layers with odd indices ("None"s) will use dense attention, and layers with an even indices
    #   will use a dynamic window size that is 50% of the input along each of the two dimensions.
    #
    #   Interleaved sliding window and dilated layers, "DiNAT"-style:
    #     natten_parameter_list = [
    #       {
    #           "kernel_size_float": (0.5, 0.5),
    #           "dilation_float": (1.0, 1.0),
    #       } if layer_idx % 2 != 0 else {
    #           "kernel_size_float": (0.5, 0.5),
    #       }
    #     ]
    #   All layers will use a dynamic window size that is 50% of the input along each of the two
    #   dimensions. Layers with odd indices will also dilate to the maximum level possible.
    #
    natten_parameter_list: list | None = None

    # Temporal causality for training autoregressive video generation models.
    # When enabled, applies temporal causal attention to generation supertokens.
    # Each supertoken is num_action_tokens_per_supertoken action tokens followed
    # by H*W vision tokens; the value is stamped onto the packed sequence by the
    # temporal-causal packer and read by attention/KV-cache code unchanged.
    # Only supports image2video modes (with or without actions).
    # Requires joint_attn_implementation="three_way".
    video_temporal_causal: bool = False
    # "none":             standard joint denoising (shared σ, no clean context)
    # "teacher_forcing":  all frames noised with shared σ; clean history via cross-attention
    # "diffusion_forcing": each latent frame gets independent σ ~ Uniform[0,1]
    # "teacher_forcing_dcm": replayed teacher-forcing discrete-time consistency distillation
    causal_training_strategy: str = attrs.field(
        default="none",
        validator=attrs.validators.in_({"none", "teacher_forcing", "diffusion_forcing", "teacher_forcing_dcm"}),
    )

    # Load balancing loss config.
    lbl: LBLConfig = LBLConfig()

    # vision configs
    vision_gen: bool = True  # whether to use vision related parameters and condition/generate vision tokens

    # action configs
    action_gen: bool = False  # whether to use action related parameters and condition/generate action tokens
    max_action_dim: int = 32  # maximum dimension of the action space, we need to pad the data to this dimension.
    num_embodiment_domains: int = 32  # number of domains for the domain-aware linear layer

    # sound configs
    sound_gen: bool = False  # whether to use sound related parameters and condition/generate sound tokens
    sound_tokenizer: LazyDict | None = None  # Sound tokenizer config (e.g., AVAE)
    sound_dim: int | None = None  # Sound latent channel size (e.g., 64 for AVAE 48kHz)
    sound_latent_fps: int = 25  # Sound tokenizer's latent rate (e.g., 48kHz / 1920 hop = 25 Hz)

    log_enc_time_every_n: int = 100  # Frequency of logging encoding time to W&B

    # When True, ``OmniMoTModel.state_dict`` / ``load_state_dict`` skip the
    # reasoner (und) pathway weights under ``language_model`` — i.e. every key
    # WITHOUT a ``_moe_gen`` suffix (including ``visual`` / ``lm_head`` /
    # ``embed_tokens``).  These are not written to checkpoints and are left
    # untouched on load (typically already populated from the HF pretrained
    # backbone).  Generation-pathway (``_moe_gen``) and VFM heads are saved /
    # loaded as usual.
    exclude_reasoner_weights_from_checkpoint: bool = False
