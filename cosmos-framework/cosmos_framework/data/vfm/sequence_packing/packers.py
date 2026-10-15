# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Top-level input sequence packing orchestration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from cosmos_framework.data.vfm.sequence_packing.modalities import (
    pack_action_tokens,
    pack_soft_plan_tokens,
    pack_sound_tokens,
    pack_text_tokens,
    pack_text_tokens_mid_soft,
    pack_vision_tokens,
)
from cosmos_framework.data.vfm.sequence_packing.temporal_causal import pack_supertokens_temporal_causal
from cosmos_framework.data.vfm.sequence_packing.types import (
    ModalityData,
    ModalityTokenSpan,
    PackedSequence,
    SampleModalitySpans,
    SequencePlan,
)

if TYPE_CHECKING:
    from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean


def pack_input_sequence(
    sequence_plans: list[SequencePlan],
    input_text_indexes: list[list[int]],
    gen_data_clean: GenerationDataClean,
    input_timesteps: torch.Tensor,
    special_tokens: dict[str, int],
    max_num_tokens: int | None = None,
    latent_patch_size: int = 1,
    skip_text_tokens: bool = False,
    include_end_of_generation_token: bool = False,
    position_embedding_type: str = "3d_rope",
    unified_3d_mrope_reset_spatial_ids: bool = True,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_base_temporal_compression_factor: int | None = None,
    temporal_compression_factor: int = 4,
    vision_temporal_position_mode: str = "latent_index",
    video_temporal_causal: bool = False,
    action_dim: int = 32,
    initial_mrope_temporal_offset: int | float = 0,
    soft_plan_num_tokens: int = 0,
    mid_soft: bool = False,
    soft_plan_insert_indexes: list[int] | None = None,
) -> PackedSequence:
    """
    Pack a sequence of input strings and VAE latents into a packed tensor format.
    Uses SequencePlan to determine which modalities are present for each sample,
    and maintains separate indices for text, vision, action, and sound to handle variable modality presence.

    Args:
        sequence_plans: List of SequencePlan items describing which modalities are present.
        input_text_indexes: List of text token ID sequences (only for samples where has_text=True).
        gen_data_clean: GenerationDataClean containing vision, action, and sound tensors.
            - x0_tokens_vision: Vision tensors for samples where has_vision=True
            - x0_tokens_action: Action tensors for samples where has_action=True
            - x0_tokens_sound: Sound tensors (list of [C, T]) for samples where has_sound=True
        input_timesteps: Diffusion timesteps for each sample. Shape (B,) or (B, 1) for
            teacher_forcing/none (all frames share the same sigma), or (B, T_max) for
            diffusion_forcing (per-frame independent sigma). Entries are extracted per
            sample as a float (numel==1) or Tensor(T_max,) for per-frame indexing.
        special_tokens: Dictionary containing special token IDs (eos_token_id, start_of_generation, end_of_generation)
        max_num_tokens: Maximum number of tokens in the packed sequence
        latent_patch_size: Patch size used by the network to pack latents
        skip_text_tokens: If True, skip packing text tokens
        include_end_of_generation_token: If True, append end-of-generation token
        position_embedding_type: Position embedding type for vision tokens:
            - "3d_rope": Additive 3D RoPE embeddings + 1D position IDs for attention
            - "flattened_sin_cos": Additive flattened sin/cos embeddings + 1D position IDs
            - "unified_3d_mrope": No additive embedding + 3D position IDs for Qwen3VL-style mRoPE
        unified_3d_mrope_reset_spatial_ids: If True (default), spatial (H, W) indices
            start from 0 for each vision segment. If False, spatial indices are offset
            by the temporal offset (Qwen2VL-style). Only used when position_embedding_type="unified_3d_mrope".
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS
            to reflect real time. Requires fps_vision in gen_data_clean.
            Uses the same flag as diffusion_expert_config.enable_fps_modulation.
        base_fps: Base FPS for normalization (default 24.0).
            Uses the same value as diffusion_expert_config.base_fps.
        sound_base_temporal_compression_factor: Base temporal compression factor for sound FPS scaling.
            ``None`` preserves the current behavior where sound advances at ``base_fps`` positions/sec.
        temporal_compression_factor: VAE temporal compression factor (default 4).
            Obtained from the VAE tokenizer at runtime.
        vision_temporal_position_mode: Temporal coordinates used for unified_3d_mrope vision tokens.
            "latent_index" keeps legacy positions; "uniae_source_right_edge" uses
            per-latent positions from gen_data_clean.temporal_positions_vision.
    Returns:
        PackedSequence containing all packed tensors and metadata. See PackedSequence for field details.
    """
    del max_num_tokens

    if mid_soft and soft_plan_num_tokens <= 0:
        raise ValueError("mid_soft=True requires soft_plan_num_tokens > 0")
    if mid_soft and soft_plan_insert_indexes is None and not skip_text_tokens:
        raise ValueError("mid_soft=True requires soft_plan_insert_indexes")

    assert special_tokens is not None, "Special tokens must be provided"
    assert isinstance(input_timesteps, torch.Tensor), "input_timesteps must be a tensor"
    if input_timesteps.is_cuda:
        raise ValueError("input_timesteps must be on CPU, not CUDA")
    if isinstance(input_text_indexes, torch.Tensor):
        raise ValueError("input_text_tokens must be a list, not a tensor")

    supported_vision_temporal_position_modes = {"latent_index", "uniae_source_right_edge"}
    if vision_temporal_position_mode not in supported_vision_temporal_position_modes:
        raise ValueError(
            "Unsupported vision_temporal_position_mode: "
            f"{vision_temporal_position_mode}. Supported modes: {supported_vision_temporal_position_modes}."
        )
    has_any_vision = any(plan.has_vision for plan in sequence_plans)
    explicit_vision_temporal_positions_active = vision_temporal_position_mode != "latent_index" and has_any_vision
    if explicit_vision_temporal_positions_active:
        if position_embedding_type != "unified_3d_mrope":
            raise NotImplementedError(
                "Explicit vision temporal positions are only supported with position_embedding_type='unified_3d_mrope'."
            )
        if gen_data_clean.temporal_positions_vision is None:
            raise ValueError(
                f"vision_temporal_position_mode={vision_temporal_position_mode} requires "
                "gen_data_clean.temporal_positions_vision."
            )
        if gen_data_clean.x0_tokens_vision is not None and len(gen_data_clean.temporal_positions_vision) != len(
            gen_data_clean.x0_tokens_vision
        ):
            raise ValueError(
                "temporal_positions_vision must have one entry per x0_tokens_vision item, "
                f"got {len(gen_data_clean.temporal_positions_vision)} positions for "
                f"{len(gen_data_clean.x0_tokens_vision)} vision items."
            )
        if video_temporal_causal:
            raise NotImplementedError(
                "video_temporal_causal=True is not wired for explicit UniAE vision temporal positions yet."
            )
        if any(plan.has_action for plan in sequence_plans):
            raise NotImplementedError("Action packing is not wired for explicit UniAE vision temporal positions yet.")
        if initial_mrope_temporal_offset != 0:
            raise NotImplementedError(
                "Autoregressive mRoPE temporal offsets are not wired for explicit UniAE vision temporal positions yet."
            )
    use_float_mrope_positions = enable_fps_modulation or explicit_vision_temporal_positions_active

    # Initialize packed sequence (acts as builder during packing)
    packed_seq = PackedSequence()
    if not skip_text_tokens and getattr(gen_data_clean, "pixel_values", None) is not None:
        pixel_values = gen_data_clean.pixel_values
        image_grid_thw = gen_data_clean.image_grid_thw
        if image_grid_thw is None:
            raise ValueError("gen_data_clean.image_grid_thw is required when pixel_values is provided.")
        if isinstance(pixel_values, list):
            packed_seq.pixel_values = torch.cat(pixel_values, dim=0)
        else:
            packed_seq.pixel_values = pixel_values
        if isinstance(image_grid_thw, list):
            packed_seq.image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            packed_seq.image_grid_thw = image_grid_thw

    native_text_position_ids = getattr(gen_data_clean, "text_position_ids", None)
    if native_text_position_ids is not None:
        if skip_text_tokens:
            raise ValueError("text_position_ids were provided while skip_text_tokens=True.")
        expected_text_samples = sum(plan.has_text for plan in sequence_plans)
        if len(native_text_position_ids) != expected_text_samples:
            raise ValueError(
                f"Expected {expected_text_samples} text_position_ids tensors, got {len(native_text_position_ids)}."
            )

    # Configure 3D mRoPE on the builder (enabled when position_embedding_type is unified_3d_mrope)
    packed_seq._use_mrope = position_embedding_type == "unified_3d_mrope"
    packed_seq._mrope_reset_spatial = unified_3d_mrope_reset_spatial_ids

    # Maintain separate indices for each modality
    idx_text = 0
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    null_action_flags: list[bool] = []  # collected from TC path; asserted consistent after the loop

    # Validate: all samples must have text (causal split is always required for two-way attention).
    # CFG dropout only drops text *content*, not the structural text split.
    if not skip_text_tokens:
        for plan in sequence_plans:
            assert plan.has_text, "All sequence plans must have has_text=True when skip_text_tokens=False"

    # Pack each sample based on its sequence plan
    for sample_idx, sequence_plan in enumerate(sequence_plans):
        curr_rope_id = 0
        sample_len = 0
        sample_spans = SampleModalitySpans()

        # mRoPE temporal offset resets per sample.
        # initial_mrope_temporal_offset is non-zero only for AR inference (frame N seeds at N*tcf).
        packed_seq._mrope_temporal_offset = initial_mrope_temporal_offset

        _ts = input_timesteps[sample_idx]
        input_timestep = _ts.item() if _ts.numel() == 1 else _ts  # float (TF) or Tensor(T_max,) (DF)

        # Pack text tokens if has_text=True and not skipped
        if sequence_plan.has_text and not skip_text_tokens:
            text_ids = input_text_indexes[idx_text]
            sample_text_position_ids = None
            if native_text_position_ids is not None:
                sample_text_position_ids = native_text_position_ids[idx_text]
                if sample_text_position_ids.ndim == 3 and sample_text_position_ids.shape[0] == 1:
                    sample_text_position_ids = sample_text_position_ids[0]
            idx_text += 1

            has_generation_for_sample = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
            text_start = packed_seq.curr
            if mid_soft:
                if soft_plan_insert_indexes is None or sample_idx >= len(soft_plan_insert_indexes):
                    raise ValueError("mid_soft packing requires one soft_plan_insert_index per sample.")
                curr_rope_id, _, text_sample_len, _ = pack_text_tokens_mid_soft(
                    packed_seq,
                    text_ids,
                    insert_index=int(soft_plan_insert_indexes[sample_idx]),
                    special_tokens=special_tokens,
                    curr_rope_id=curr_rope_id,
                    has_generation=has_generation_for_sample,
                    use_float_positions=use_float_mrope_positions,
                    text_position_ids=sample_text_position_ids,
                    num_soft_plan_tokens=soft_plan_num_tokens,
                )
            else:
                curr_rope_id, _, text_sample_len = pack_text_tokens(
                    packed_seq,
                    text_ids,
                    special_tokens,
                    curr_rope_id,
                    has_generation=has_generation_for_sample,
                    use_float_positions=use_float_mrope_positions,
                    text_position_ids=sample_text_position_ids,
                )
            sample_spans.text.append(ModalityTokenSpan(start=text_start, end=packed_seq.curr))
            sample_len += text_sample_len

            if soft_plan_num_tokens > 0 and not mid_soft:
                if not has_generation_for_sample:
                    raise ValueError("soft_plan_num_tokens requires a generation modality and SOG token.")
                curr_rope_id, soft_plan_len = pack_soft_plan_tokens(
                    packed_seq,
                    num_tokens=soft_plan_num_tokens,
                    curr_rope_id=curr_rope_id,
                    use_float_positions=use_float_mrope_positions,
                )
                sample_len += soft_plan_len

            # End of text modality, add an offset as the boundary between text and vision.
            packed_seq._mrope_temporal_offset += unified_3d_mrope_temporal_modality_margin

        # Save temporal offset before vision for action tokens (action uses same offset as vision start)
        vision_start_temporal_offset = packed_seq._mrope_temporal_offset

        # Pack vision (and optionally action) tokens
        if video_temporal_causal and sequence_plan.has_vision:
            # Temporal causal path: when sequence_plan.has_action=True, interleaved supertokens
            # [action_t, vision_t]; when False, supertokens are just vision patches.
            assert position_embedding_type == "unified_3d_mrope", (
                "video_temporal_causal=True requires position_embedding_type='unified_3d_mrope'"
            )
            input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]
            idx_vision += 1

            vision_fps = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_vision is not None
                and idx_vision - 1 < len(gen_data_clean.fps_vision)
            ):
                vision_fps = float(gen_data_clean.fps_vision[idx_vision - 1].item())

            input_action_tokens_tc: torch.Tensor | None = None
            action_fps_tc: float | None = None
            if sequence_plan.has_action:
                input_action_tokens_tc = gen_data_clean.x0_tokens_action[idx_action]
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps_tc = float(gen_data_clean.fps_action[idx_action].item())
                idx_action += 1

            supertoken_split_len, null_flag = pack_supertokens_temporal_causal(
                packed_seq=packed_seq,
                input_vision_tokens=input_vision_tokens,
                input_action_tokens=input_action_tokens_tc,
                condition_frame_indexes_vision=sequence_plan.condition_frame_indexes_vision,
                input_timestep=input_timestep,
                curr_rope_id=curr_rope_id,
                latent_patch_size=latent_patch_size,
                temporal_compression_factor=temporal_compression_factor,
                action_dim=action_dim,
                vision_fps=vision_fps,
                action_fps=action_fps_tc,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                pack_action_tokens=sequence_plan.has_action,
            )
            null_action_flags.append(null_flag)
            # We assume all samples in a batch share the same has_action layout, so
            # stamp the supertoken layout constant directly here. This is the
            # single source of truth read by downstream attention / KV-cache
            # code (no recomputation in the network).
            packed_seq.num_action_tokens_per_supertoken = temporal_compression_factor if sequence_plan.has_action else 0
            sample_len += supertoken_split_len
            vision_split_len = supertoken_split_len
            action_split_len = 0  # Already absorbed into supertoken_split_len

        else:
            # Standard path: vision and action packed separately
            if sequence_plan.has_vision:
                # Determine how many vision items this sample owns.
                # For multi-item samples (e.g. image editing), num_vision_items_per_sample
                # records [2, 2, ...]; for standard T2I/T2V it is None (1 item per sample).
                num_vis = (
                    gen_data_clean.num_vision_items_per_sample[sample_idx]
                    if gen_data_clean.num_vision_items_per_sample is not None
                    else 1
                )

                vision_split_len = 0
                # Per-item split lengths for multi-control attention routing.
                # Only tracked when control_weights are present (inference-only);
                # skipped during training to avoid unnecessary side effects.
                track_item_split_lens = gen_data_clean.control_weights is not None
                sample_item_split_lens: list[int] = []
                # Controlnet-style transfer: when set, all vision items share the same
                # temporal mRoPE grid. We snapshot the offset before the loop and
                # rewind to it before each item, so every item produces identical
                # temporal IDs. Each pack_vision_tokens call still advances the
                # offset by latent_t internally; in shared-grid mode the post-loop
                # offset equals snapshot + latent_t (single-clip semantics for
                # downstream EOV / next-modality tokens).
                shared_grid = sequence_plan.share_vision_temporal_positions and num_vis > 1
                items_temporal_offset_snapshot = packed_seq._mrope_temporal_offset
                shared_latent_t: int | None = None
                shared_patch_h: int | None = None
                shared_patch_w: int | None = None
                shared_temporal_positions: torch.Tensor | None = None
                # FPS is recorded per-sample (shape [B]); for multi-item samples
                # (transfer / image-edit) every vision item in this sample shares
                # the same conditioning FPS, so we read by sample_idx, not by the
                # flat idx_vision counter (which would alias to a neighbor sample's
                # fps and corrupt RoPE FPS modulation).
                sample_vision_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_vision is not None
                    and sample_idx < len(gen_data_clean.fps_vision)
                ):
                    sample_vision_fps = float(gen_data_clean.fps_vision[sample_idx].item())

                for item_idx in range(num_vis):
                    flat_vision_idx = idx_vision
                    input_vision_tokens = gen_data_clean.x0_tokens_vision[flat_vision_idx]
                    _, _, item_latent_t, item_latent_h, item_latent_w = input_vision_tokens.shape
                    item_patch_h = math.ceil(item_latent_h / latent_patch_size)
                    item_patch_w = math.ceil(item_latent_w / latent_patch_size)
                    vision_temporal_positions: torch.Tensor | None = None
                    if explicit_vision_temporal_positions_active:
                        assert gen_data_clean.temporal_positions_vision is not None
                        vision_temporal_positions = gen_data_clean.temporal_positions_vision[flat_vision_idx]
                        if vision_temporal_positions.shape[0] != input_vision_tokens.shape[2]:
                            raise ValueError(
                                "vision_temporal_positions must match latent_t for each vision item, "
                                f"got {vision_temporal_positions.shape[0]} positions and "
                                f"latent_t={input_vision_tokens.shape[2]} for item {flat_vision_idx}."
                            )
                    vision_fps = sample_vision_fps
                    idx_vision += 1

                    # Determine conditioning for this vision item.
                    # For multi-item mode: all items except the last are fully conditioned
                    # (all frames are clean); the last item uses the SequencePlan's
                    # condition_frame_indexes_vision (typically [] = fully generated).
                    if num_vis > 1 and item_idx < num_vis - 1:
                        # Conditioning item (e.g. source image): mark all frames as clean
                        latent_t = input_vision_tokens.shape[2]
                        item_condition_frames = list(range(latent_t))
                    else:
                        # Generation item (single-item mode or last item in multi-item)
                        item_condition_frames = sequence_plan.condition_frame_indexes_vision

                    if shared_grid:
                        if shared_latent_t is None:
                            shared_latent_t = item_latent_t
                            shared_patch_h = item_latent_h
                            shared_patch_w = item_latent_w
                        else:
                            assert item_latent_t == shared_latent_t, (
                                f"share_vision_temporal_positions requires equal latent_t across items, "
                                f"got item {item_idx} latent_t={item_latent_t} vs first={shared_latent_t}"
                            )
                            assert item_latent_h == shared_patch_h and item_latent_w == shared_patch_w, (
                                f"share_vision_temporal_positions requires equal spatial grid across items, "
                                f"got item {item_idx} (H,W)=({item_latent_h},{item_latent_w}) "
                                f"vs first=({shared_patch_h},{shared_patch_w})"
                            )
                        if vision_temporal_positions is not None:
                            if shared_temporal_positions is None:
                                shared_temporal_positions = vision_temporal_positions
                            else:
                                comparison_temporal_positions = vision_temporal_positions.to(
                                    device=shared_temporal_positions.device
                                )  # [T]
                                assert torch.allclose(comparison_temporal_positions, shared_temporal_positions), (
                                    "share_vision_temporal_positions requires equal explicit temporal positions "
                                    f"across vision items, got item {item_idx} positions "
                                    f"{vision_temporal_positions.tolist()} vs first "
                                    f"{shared_temporal_positions.tolist()}."
                                )
                        # Rewind so this item starts at the same temporal offset as item 0.
                        packed_seq._mrope_temporal_offset = items_temporal_offset_snapshot

                    vision_start = packed_seq.curr
                    item_split_len = pack_vision_tokens(
                        packed_seq=packed_seq,
                        input_vision_tokens=input_vision_tokens,
                        condition_frame_indexes_vision=item_condition_frames,
                        input_timestep=input_timestep,
                        curr_rope_id=curr_rope_id,
                        latent_patch_size=latent_patch_size,
                        vision_fps=vision_fps,
                        enable_fps_modulation=enable_fps_modulation,
                        base_fps=base_fps,
                        temporal_compression_factor=temporal_compression_factor,
                        vision_temporal_positions=vision_temporal_positions,
                    )
                    sample_spans.vision.append(
                        ModalityTokenSpan(
                            start=vision_start,
                            end=packed_seq.curr,
                            token_shape=(item_latent_t, item_patch_h, item_patch_w),
                        )
                    )
                    vision_split_len += item_split_len
                    if track_item_split_lens:
                        sample_item_split_lens.append(item_split_len)
                if track_item_split_lens:
                    packed_seq.vision_item_split_lens.append(sample_item_split_lens)
                sample_len += vision_split_len

            else:
                vision_split_len = 0

            # Pack action tokens if has_action=True
            if sequence_plan.has_action:
                input_action_tokens = gen_data_clean.x0_tokens_action[idx_action]

                # Get FPS for action (action may have its own FPS independent of vision)
                action_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps = float(gen_data_clean.fps_action[idx_action].item())

                idx_action += 1

                action_start = packed_seq.curr
                action_split_len = pack_action_tokens(
                    packed_seq=packed_seq,
                    input_action_tokens=input_action_tokens,
                    condition_frame_indexes_action=sequence_plan.condition_frame_indexes_action,
                    input_timestep=input_timestep,
                    curr_rope_id=curr_rope_id,
                    action_temporal_offset=vision_start_temporal_offset,
                    enable_fps_modulation=enable_fps_modulation,
                    base_fps=base_fps,
                    action_fps=action_fps,
                    base_temporal_compression_factor=temporal_compression_factor,
                    action_start_frame_offset=sequence_plan.action_start_frame_offset,
                )
                sample_spans.action.append(
                    ModalityTokenSpan(start=action_start, end=packed_seq.curr, token_shape=(action_split_len,))
                )
                sample_len += action_split_len
            else:
                action_split_len = 0

        # Pack sound tokens if has_sound=True
        if sequence_plan.has_sound:
            input_sound_tokens = gen_data_clean.x0_tokens_sound[idx_sound]

            # Get FPS for sound (from gen_data_clean, like vision and action)
            sound_fps: float | None = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_sound is not None
                and idx_sound < len(gen_data_clean.fps_sound)
            ):
                sound_fps = float(gen_data_clean.fps_sound[idx_sound].item())

            idx_sound += 1

            sound_start = packed_seq.curr
            sound_split_len = pack_sound_tokens(
                packed_seq=packed_seq,
                input_sound_tokens=input_sound_tokens,
                condition_frame_indexes_sound=sequence_plan.condition_frame_indexes_sound,
                input_timestep=input_timestep,
                curr_rope_id=curr_rope_id,
                sound_temporal_offset=vision_start_temporal_offset,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                sound_fps=sound_fps,
                sound_base_temporal_compression_factor=sound_base_temporal_compression_factor,
            )
            sample_spans.sound.append(
                ModalityTokenSpan(start=sound_start, end=packed_seq.curr, token_shape=(sound_split_len, 1, 1))
            )
            sample_len += sound_split_len
        else:
            sound_split_len = 0

        # Add end-of-generation token if needed
        eov_len = 0
        has_any_generation = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
        if include_end_of_generation_token and has_any_generation:
            # Type narrowing: we're in build mode, fields are lists
            assert isinstance(packed_seq.text_ids, list)
            assert isinstance(packed_seq.text_indexes, list)
            assert isinstance(packed_seq.position_ids, list)

            packed_seq.text_ids.append(special_tokens["end_of_generation"])
            packed_seq.text_indexes.append(packed_seq.curr)

            # EOV position IDs: 3D mRoPE or 1D RoPE
            if packed_seq._use_mrope:
                # Use float dtype when any vision mRoPE positions are fractional.
                eov_dtype = torch.float32 if use_float_mrope_positions else torch.long
                eov_mrope_ids = torch.full((3, 1), packed_seq._mrope_temporal_offset, dtype=eov_dtype)  # [3,1]
                packed_seq.position_ids.append(eov_mrope_ids)  # type: ignore[arg-type]
                packed_seq._mrope_temporal_offset += 1
            else:
                packed_seq.position_ids.append(curr_rope_id)  # type: ignore[arg-type]

            packed_seq.curr += 1
            eov_len = 1
            sample_len += 1

        combined_split_len = vision_split_len + action_split_len + sound_split_len + eov_len
        packed_seq.attn_modes.append("full")
        packed_seq.split_lens.append(combined_split_len)
        packed_seq.sample_lens.append(sample_len)
        packed_seq.sample_modality_spans.append(sample_spans)

    # Assert consistent null_action_supertokens across all TC samples, then set once
    if null_action_flags:
        assert len(set(null_action_flags)) == 1, (
            f"Inconsistent null_action_supertokens across samples: {null_action_flags}. "
            "All samples in a batch must have the same structure (all training or all AR inference)."
        )
        packed_seq.null_action_supertokens = null_action_flags[0]

    # Finalize and return packed data
    return packed_seq.finalize(
        gen_data_clean=gen_data_clean,
    )


def _fastwam_visible_frame_indexes(latent_t: int, action_video_attention_mode: str) -> list[int]:
    if latent_t <= 0:
        raise ValueError(f"Vision token_shape T must be positive, got {latent_t}.")
    if action_video_attention_mode == "fastwam_first_frame" or latent_t == 1:
        return [0]
    if action_video_attention_mode == "fastwam_first_last_frame":
        return [0, latent_t - 1]
    raise ValueError(f"Unsupported FastWAM action/video attention mode: {action_video_attention_mode!r}.")


def build_fastwam_action_only_packed_sequence(
    packed_seq: PackedSequence,
    *,
    action_video_attention_mode: str = "fastwam_first_frame",
) -> PackedSequence:
    """Build an action-only sequence from a joint vision/action sequence.

    The returned sequence keeps the causal text split, the FastWAM-visible
    frames from every vision span, and all action tokens. It drops other video
    tokens so an action-only forward can be compared directly against a joint
    forward that uses FastWAM-style visibility.

    Output shape conceptually becomes:
        text tokens + visible vision tokens + full action chunk tokens.
    For a 32-step policy chunk, the action side stays length 32; only video is
    reduced according to ``action_video_attention_mode``.
    """
    if packed_seq.vision is None or packed_seq.action is None:
        raise ValueError("FastWAM action-only packing requires both vision and action modalities.")
    if not packed_seq.sample_modality_spans:
        raise ValueError("FastWAM action-only packing requires PackedSequence.sample_modality_spans metadata.")
    if packed_seq.sound is not None:
        raise NotImplementedError("FastWAM action-only packing does not support sound tokens yet.")
    if isinstance(packed_seq.soft_plan_indexes, torch.Tensor) and packed_seq.soft_plan_indexes.numel() > 0:
        raise ValueError("FastWAM action-only packing is incompatible with soft-plan bottleneck sequences.")

    device = packed_seq.text_indexes.device
    # Absolute indexes in the original joint packed sequence that survive in
    # the action-only sequence. They are remapped through old_to_new below.
    kept_indices: list[int] = []
    new_sample_lens: list[int] = []
    new_split_lens: list[int] = []
    new_attn_modes: list[str] = []
    new_spans: list[SampleModalitySpans] = []

    for sample_spans in packed_seq.sample_modality_spans:
        sample_new_start = len(kept_indices)
        text_indices: list[int] = []
        for span in sample_spans.text:
            text_indices.extend(range(span.start, span.end))
        kept_indices.extend(text_indices)
        text_len = len(text_indices)

        sample_new_spans = SampleModalitySpans()
        if text_len > 0:
            sample_new_spans.text.append(ModalityTokenSpan(start=sample_new_start, end=sample_new_start + text_len))

        full_start_new = len(kept_indices)
        for span in sample_spans.vision:
            if span.token_shape is None or len(span.token_shape) != 3:
                raise ValueError(f"Vision span requires token_shape=(T,H,W), got {span.token_shape}.")
            latent_t, patch_h, patch_w = span.token_shape
            frame_len = patch_h * patch_w
            frame_indexes = _fastwam_visible_frame_indexes(latent_t, action_video_attention_mode)
            vision_start_new = len(kept_indices)
            for frame_idx in frame_indexes:
                frame_start = span.start + frame_idx * frame_len
                kept_indices.extend(range(frame_start, frame_start + frame_len))
            sample_new_spans.vision.append(
                ModalityTokenSpan(
                    start=vision_start_new,
                    end=vision_start_new + len(frame_indexes) * frame_len,
                    token_shape=(len(frame_indexes), patch_h, patch_w),
                )
            )

        for span in sample_spans.action:
            action_start_new = len(kept_indices)
            kept_indices.extend(range(span.start, span.end))
            sample_new_spans.action.append(
                ModalityTokenSpan(start=action_start_new, end=len(kept_indices), token_shape=span.token_shape)
            )

        full_len = len(kept_indices) - full_start_new
        sample_len = len(kept_indices) - sample_new_start
        if full_len <= 0:
            raise ValueError("FastWAM action-only packing requires at least one generation token per sample.")
        new_sample_lens.append(sample_len)
        new_split_lens.extend([text_len, full_len])
        new_attn_modes.extend(["causal", "full"])
        new_spans.append(sample_new_spans)

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
    kept_index_tensor = torch.tensor(kept_indices, dtype=torch.long, device=packed_seq.position_ids.device)

    if packed_seq.position_ids.ndim == 2:
        new_position_ids = packed_seq.position_ids.index_select(1, kept_index_tensor)
    else:
        new_position_ids = packed_seq.position_ids.index_select(0, kept_index_tensor)

    text_indexes: list[int] = []
    text_ids: list[torch.Tensor] = []
    assert isinstance(packed_seq.text_indexes, torch.Tensor)
    assert isinstance(packed_seq.text_ids, torch.Tensor)
    for text_order, old_text_idx in enumerate(packed_seq.text_indexes.tolist()):
        if old_text_idx in old_to_new:
            text_indexes.append(old_to_new[old_text_idx])
            text_ids.append(packed_seq.text_ids[text_order])
    new_text_ids = (
        torch.stack(text_ids).to(device=device)
        if text_ids
        else torch.empty((0,), dtype=packed_seq.text_ids.dtype, device=device)
    )
    new_text_indexes = torch.tensor(text_indexes, dtype=torch.long, device=device)

    new_label_ids: torch.Tensor | None = None
    new_ce_loss_indexes: torch.Tensor | None = None
    new_ce_loss_weights: torch.Tensor | None = None
    if (
        isinstance(packed_seq.label_ids, torch.Tensor)
        and isinstance(packed_seq.ce_loss_indexes, torch.Tensor)
        and isinstance(packed_seq.ce_loss_weights, torch.Tensor)
    ):
        ce_labels: list[torch.Tensor] = []
        ce_indexes: list[int] = []
        ce_weights: list[torch.Tensor] = []
        for ce_order, old_ce_idx in enumerate(packed_seq.ce_loss_indexes.tolist()):
            if old_ce_idx in old_to_new:
                ce_indexes.append(old_to_new[old_ce_idx])
                ce_labels.append(packed_seq.label_ids[ce_order])
                ce_weights.append(packed_seq.ce_loss_weights[ce_order])
        if ce_indexes:
            new_label_ids = torch.stack(ce_labels).to(device=device)
            new_ce_loss_indexes = torch.tensor(ce_indexes, dtype=torch.long, device=device)
            new_ce_loss_weights = torch.stack(ce_weights).to(device=device)

    assert packed_seq.vision.tokens is not None
    new_vision_tokens: list[torch.Tensor] = []
    new_vision_sequence_indexes: list[int] = []
    new_vision_mse_loss_indexes: list[int] = []
    new_vision_timesteps: list[torch.Tensor] = []
    new_vision_token_shapes: list[tuple[int, int, int]] = []
    new_vision_condition_mask: list[torch.Tensor] = []
    new_vision_noisy_frame_indexes: list[torch.Tensor] = []
    vision_timestep_offset = 0
    for vision_idx, span in enumerate(span for spans in packed_seq.sample_modality_spans for span in spans.vision):
        if span.token_shape is None:
            raise ValueError("Vision span token_shape is required.")
        latent_t, patch_h, patch_w = span.token_shape
        frame_len = patch_h * patch_w
        frame_indexes = _fastwam_visible_frame_indexes(latent_t, action_video_attention_mode)
        frame_index_tensor = torch.tensor(
            frame_indexes, dtype=torch.long, device=packed_seq.vision.tokens[vision_idx].device
        )
        new_vision_tokens.append(packed_seq.vision.tokens[vision_idx].index_select(2, frame_index_tensor).contiguous())
        new_vision_token_shapes.append((len(frame_indexes), patch_h, patch_w))
        for frame_idx in frame_indexes:
            frame_start = span.start + frame_idx * frame_len
            new_vision_sequence_indexes.extend(old_to_new[i] for i in range(frame_start, frame_start + frame_len))
        old_mask = packed_seq.vision.condition_mask[vision_idx]
        new_vision_condition_mask.append(old_mask.index_select(0, frame_index_tensor.to(old_mask.device)).clone())
        old_noisy_frame_indexes = packed_seq.vision.noisy_frame_indexes[vision_idx]
        visible_noisy_frame_indexes: list[int] = []
        old_noisy_frame_to_order = {
            int(frame_idx): noisy_order for noisy_order, frame_idx in enumerate(old_noisy_frame_indexes.tolist())
        }
        for new_frame_idx, old_frame_idx in enumerate(frame_indexes):
            old_noisy_order = old_noisy_frame_to_order.get(int(old_frame_idx))
            if old_noisy_order is None:
                continue
            visible_noisy_frame_indexes.append(new_frame_idx)
            frame_start = span.start + old_frame_idx * frame_len
            new_vision_mse_loss_indexes.extend(old_to_new[i] for i in range(frame_start, frame_start + frame_len))
            assert isinstance(packed_seq.vision.timesteps, torch.Tensor)
            old_timestep_start = vision_timestep_offset + old_noisy_order * frame_len
            new_vision_timesteps.extend(
                packed_seq.vision.timesteps[old_timestep_start : old_timestep_start + frame_len]
            )
        new_vision_noisy_frame_indexes.append(
            torch.tensor(visible_noisy_frame_indexes, dtype=torch.long, device=old_mask.device)
        )
        vision_timestep_offset += int(old_noisy_frame_indexes.numel()) * frame_len

    vision = ModalityData(
        sequence_indexes=torch.tensor(new_vision_sequence_indexes, dtype=torch.long, device=device),
        timesteps=(
            torch.stack(new_vision_timesteps).to(device=device)
            if new_vision_timesteps
            else torch.empty((0,), dtype=packed_seq.vision.timesteps.dtype, device=device)
        ),
        mse_loss_indexes=torch.tensor(new_vision_mse_loss_indexes, dtype=torch.long, device=device),
        token_shapes=new_vision_token_shapes,
        tokens=new_vision_tokens,
        condition_mask=new_vision_condition_mask,
        noisy_frame_indexes=new_vision_noisy_frame_indexes,
    )

    assert packed_seq.action.tokens is not None
    new_action_sequence_indexes: list[int] = []
    new_action_mse_loss_indexes: list[int] = []
    for action_idx, span in enumerate(span for spans in packed_seq.sample_modality_spans for span in spans.action):
        # Keep the full action chunk. This is what makes infer_action predict
        # the same action horizon as infer_joint while omitting future video.
        new_action_sequence_indexes.extend(old_to_new[i] for i in range(span.start, span.end))
        for noisy_frame_idx in packed_seq.action.noisy_frame_indexes[action_idx].tolist():
            old_idx = span.start + noisy_frame_idx
            new_action_mse_loss_indexes.append(old_to_new[old_idx])

    action = ModalityData(
        sequence_indexes=torch.tensor(new_action_sequence_indexes, dtype=torch.long, device=device),
        timesteps=packed_seq.action.timesteps,
        mse_loss_indexes=torch.tensor(new_action_mse_loss_indexes, dtype=torch.long, device=device),
        token_shapes=list(packed_seq.action.token_shapes),
        tokens=list(packed_seq.action.tokens),
        condition_mask=list(packed_seq.action.condition_mask),
        noisy_frame_indexes=list(packed_seq.action.noisy_frame_indexes),
        domain_id=list(packed_seq.action.domain_id),
        raw_action_dim=packed_seq.action.raw_action_dim,
    )

    return PackedSequence(
        sequence_length=len(kept_indices),
        sample_lens=new_sample_lens,
        split_lens=new_split_lens,
        attn_modes=new_attn_modes,
        sample_modality_spans=new_spans,
        is_image_batch=packed_seq.is_image_batch,
        text_ids=new_text_ids,
        text_indexes=new_text_indexes,
        soft_plan_indexes=torch.empty((0,), dtype=torch.long, device=device),
        soft_plan_ranges=[],
        position_ids=new_position_ids,
        label_ids=new_label_ids,
        ce_loss_indexes=new_ce_loss_indexes,
        ce_loss_weights=new_ce_loss_weights,
        vision=vision,
        action=action,
        sound=None,
        null_action_supertokens=False,
        num_action_tokens_per_supertoken=0,
        vision_item_split_lens=[],
        control_weights=None,
    )
