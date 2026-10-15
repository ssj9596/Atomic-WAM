# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Modality-specific append helpers for VFM sequence packing."""

import math
from typing import Dict, List, Tuple

import torch

from cosmos_framework.data.vfm.sequence_packing.mrope import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from cosmos_framework.data.vfm.sequence_packing.types import ModalityData, PackedSequence


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """Prepare dense attention mask for a single sample with multiple splits.

    Args:
        split_lens: List of integers indicating length of each split within the sample
        attn_modes: List of attention modes for each split ('causal', 'full', or 'noise')
        device: Device to place the attention mask tensor on

    Returns:
        Attention mask tensor of shape (sample_len, sample_len) with -inf for masked positions
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)  # [sample_len,sample_len]

    # First pass: Set up basic attention patterns for each split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"], f"Invalid attention mode: {attn_mode}"

        split_start = current_pos
        split_end = current_pos + split_len

        if attn_mode == "causal":
            # Causal: lower triangular within split + full attention to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            ).tril()  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1
        else:  # "full" or "noise"
            # Full attention within split and to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            )  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1

        current_pos += split_len

    # Second pass: Handle noise mode - mask out noise columns except within same split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            split_start = current_pos
            split_end = current_pos + split_len

            # Zero out the entire column for noise tokens
            attention_mask[:, split_start:split_end] = 0
            # But allow self-attention within the noise split
            attention_mask[split_start:split_end, split_start:split_end] = 1

        current_pos += split_len

    # Convert boolean mask to float with -inf for masked positions
    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )  # [sample_len,sample_len]

    return attention_mask


# ============================================================================
# Tokenizer utilities
# ============================================================================


def add_special_tokens(tokenizer):
    """Add image-related special tokens to tokenizer if not already present.

    Args:
        tokenizer: Tokenizer to add special tokens to

    Returns:
        Tuple of (modified tokenizer, dict of new token IDs)
    """
    # Collect existing special tokens
    existing_special_tokens = []
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            existing_special_tokens.append(value)
        elif isinstance(value, list):
            existing_special_tokens.extend(value)

    # Define image boundary tokens to add if missing
    tokens_to_add = []
    if "<|vision_start|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_start|>")
    if "<|vision_end|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_end|>")

    # Add new tokens to tokenizer vocabulary
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)

    # Get token IDs for image boundary tokens
    new_token_ids = {
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }

    return tokenizer, new_token_ids


def compute_text_split_length(
    num_caption_tokens: int,
    special_tokens: Dict[str, int],
    has_generation: bool = True,
) -> int:
    """Compute the total text split length without mutating any state.

    This is the number of token positions occupied by the text split in a
    packed sequence: caption tokens + optional BOS + EOS + optional BOV.

    Args:
        num_caption_tokens: Number of raw caption token IDs (before special tokens).
        special_tokens: Dictionary of special token IDs (checked for ``"bos_token_id"``).
        has_generation: Whether a start-of-generation (BOV) token follows text.

    Returns:
        Total text split length (positions consumed in the packed sequence).
    """
    n = num_caption_tokens
    if "bos_token_id" in special_tokens:
        n += 1
    n += 1  # EOS
    if has_generation:
        n += 1  # start-of-generation / BOV
    return n


def pack_text_tokens(
    packed_seq: PackedSequence,
    text_ids: List[int],
    special_tokens: Dict[str, int],
    curr_rope_id: int,
    has_generation: bool,
    use_float_positions: bool = False,
    text_position_ids: torch.Tensor | None = None,
) -> Tuple[int, int, int]:
    """Pack text tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        text_ids: List of text token IDs (integers).
        special_tokens: Dictionary of special token IDs.
        curr_rope_id: Current RoPE position ID.
        has_generation: Whether there's media/action after text.
        use_float_positions: If True, generate float position IDs for 3D mRoPE
            (for consistency with FPS-modulated vision tokens).
        text_position_ids: Optional native multimodal mRoPE IDs with shape
            ``[3, len(text_ids)]``. Only used by the opt-in Qwen3-VL image path.

    Returns:
        Tuple of (updated curr_rope_id, split_length, sample_length).
    """
    # Ensure we're in build mode (fields are lists, not tensors)
    assert isinstance(packed_seq.text_ids, list), "PackedSequence must be in build mode"
    assert isinstance(packed_seq.text_indexes, list)
    assert isinstance(packed_seq.position_ids, list)
    assert isinstance(packed_seq.label_ids, list)
    assert isinstance(packed_seq.ce_loss_indexes, list)
    assert isinstance(packed_seq.ce_loss_weights, list)

    curr = packed_seq.curr

    # Prepend BOS token if available
    if "bos_token_id" in special_tokens:
        shifted_text_ids = [special_tokens["bos_token_id"]] + text_ids
    else:
        shifted_text_ids = text_ids

    split_len = 0

    # Add text tokens to sequence
    packed_seq.text_ids.extend(shifted_text_ids)
    packed_seq.text_indexes.extend(range(curr, curr + len(shifted_text_ids)))

    # Configure loss computation for text tokens
    packed_seq.ce_loss_indexes.extend(range(curr, curr + len(shifted_text_ids)))
    packed_seq.ce_loss_weights.extend([1.0] * len(shifted_text_ids))
    packed_seq.label_ids.extend(text_ids[1:] + [special_tokens["eos_token_id"]])

    curr += len(shifted_text_ids)
    split_len += len(shifted_text_ids)

    # Add EOS token
    packed_seq.text_ids.append(special_tokens["eos_token_id"])
    packed_seq.text_indexes.append(curr)
    curr += 1
    split_len += 1

    # Add start-of-generation token, but only if there's media/action present.
    if has_generation:
        packed_seq.text_ids.append(special_tokens["start_of_generation"])
        packed_seq.text_indexes.append(curr)
        curr += 1
        split_len += 1

    # Sanity check -- compute_text_split_length() is called elsewhere.
    assert split_len == compute_text_split_length(len(text_ids), special_tokens, has_generation)

    # Update position IDs and attention mode for text split
    if text_position_ids is not None and not packed_seq._use_mrope:
        raise ValueError("Native text_position_ids require position_embedding_type='unified_3d_mrope'.")
    if packed_seq._use_mrope:
        if text_position_ids is None:
            text_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_text_tokens(
                num_tokens=split_len,
                temporal_offset=packed_seq._mrope_temporal_offset,
                use_float_positions=use_float_positions,
            )  # text_mrope_ids: [3,split_len]
        else:
            if text_position_ids.shape != (3, len(text_ids)):
                raise ValueError(
                    f"Expected text_position_ids shape (3, {len(text_ids)}), got {tuple(text_position_ids.shape)}"
                )
            # Position metadata is assembled on CPU throughout this packer and
            # moved to CUDA once by PackedSequence.to_cuda() after finalization.
            # Qwen native mRoPE IDs may already be on CUDA with the input batch,
            # so normalize them here before concatenating generated positions.
            native_positions = text_position_ids.to(
                device="cpu", dtype=torch.float32 if use_float_positions else torch.long
            )
            if "bos_token_id" in special_tokens:
                native_positions = native_positions + 1
                native_positions = torch.cat(
                    (
                        torch.zeros((3, 1), dtype=native_positions.dtype, device=native_positions.device),
                        native_positions,
                    ),
                    dim=1,
                )
            native_positions = native_positions + packed_seq._mrope_temporal_offset
            num_appended_tokens = split_len - native_positions.shape[1]
            next_position = native_positions.max() + 1
            appended_positions = (
                torch.arange(
                    num_appended_tokens,
                    dtype=native_positions.dtype,
                    device=native_positions.device,
                )
                .unsqueeze(0)
                .expand(3, -1)
                + next_position
            )
            text_mrope_ids = torch.cat((native_positions, appended_positions), dim=1)
            packed_seq._mrope_temporal_offset = text_mrope_ids.max().item() + 1
        packed_seq.position_ids.append(text_mrope_ids)
    else:
        packed_seq.position_ids.extend(range(curr_rope_id, curr_rope_id + split_len))
    packed_seq.attn_modes.append("causal")
    packed_seq.split_lens.append(split_len)

    packed_seq.curr = curr
    return curr_rope_id + split_len, split_len, split_len


def pack_text_tokens_mid_soft(
    packed_seq: PackedSequence,
    text_ids: List[int],
    insert_index: int,
    special_tokens: Dict[str, int],
    curr_rope_id: int,
    has_generation: bool,
    use_float_positions: bool = False,
    text_position_ids: torch.Tensor | None = None,
    num_soft_plan_tokens: int = 0,
) -> Tuple[int, int, int, tuple[int, int]]:
    """Pack ``[T_pre][P][T_post][EOS][SOG]`` for the opt-in mid-soft layout.

    The whole text-side sequence remains one causal split.  This is important:
    ordinary causal attention then gives P access to the instruction prefix and
    gives trailing metadata access to P without allowing P to read metadata.
    ``soft_plan_ranges`` records the absolute P range for the custom generation
    KV selection performed by the attention kernel.
    """
    if num_soft_plan_tokens <= 0:
        raise ValueError("mid-soft packing requires num_soft_plan_tokens > 0")
    if not 0 <= insert_index <= len(text_ids):
        raise ValueError(f"soft_plan insert_index={insert_index} outside text token range [0, {len(text_ids)}]")
    if has_generation is False:
        raise ValueError("mid-soft packing requires a generation modality")

    assert isinstance(packed_seq.text_ids, list)
    assert isinstance(packed_seq.text_indexes, list)
    assert isinstance(packed_seq.position_ids, list)
    assert isinstance(packed_seq.label_ids, list)
    assert isinstance(packed_seq.ce_loss_indexes, list)
    assert isinstance(packed_seq.ce_loss_weights, list)

    raw_start = packed_seq.curr
    shifted = ([special_tokens["bos_token_id"]] if "bos_token_id" in special_tokens else []) + list(text_ids)
    bos_count = 1 if "bos_token_id" in special_tokens else 0
    split_len = len(shifted) + num_soft_plan_tokens + 1 + 1  # EOS + SOG

    # Structural/text token positions, with P omitted from text_indexes.
    pre = shifted[: bos_count + insert_index]
    post = shifted[bos_count + insert_index :]
    packed_seq.text_ids.extend(pre)
    packed_seq.text_indexes.extend(range(packed_seq.curr, packed_seq.curr + len(pre)))
    pre_positions = list(range(packed_seq.curr, packed_seq.curr + len(pre)))
    packed_seq.curr += len(pre)

    plan_start = packed_seq.curr
    packed_seq.soft_plan_indexes.extend(range(plan_start, plan_start + num_soft_plan_tokens))
    packed_seq.soft_plan_ranges.append((plan_start, plan_start + num_soft_plan_tokens))
    packed_seq.curr += num_soft_plan_tokens

    post_start = packed_seq.curr
    packed_seq.text_ids.extend(post)
    packed_seq.text_indexes.extend(range(post_start, post_start + len(post)))
    packed_seq.curr += len(post)

    eos_pos = packed_seq.curr
    packed_seq.text_ids.append(special_tokens["eos_token_id"])
    packed_seq.text_indexes.append(eos_pos)
    packed_seq.curr += 1
    if has_generation:
        packed_seq.text_ids.append(special_tokens["start_of_generation"])
        packed_seq.text_indexes.append(packed_seq.curr)
        packed_seq.curr += 1

    # CE remains defined only on real token positions.  The target sequence is
    # the original text shifted by one, so P itself never receives a CE target.
    ce_positions = pre_positions[bos_count:] + list(range(post_start, post_start + len(post)))
    packed_seq.ce_loss_indexes.extend(ce_positions)
    packed_seq.ce_loss_weights.extend([1.0] * len(ce_positions))
    packed_seq.label_ids.extend(list(text_ids[1:]) + [special_tokens["eos_token_id"]])

    if text_position_ids is not None and not packed_seq._use_mrope:
        raise ValueError("Native text_position_ids require position_embedding_type='unified_3d_mrope'.")
    if packed_seq._use_mrope:
        dtype = torch.float32 if use_float_positions else torch.long
        if text_position_ids is None:
            positions = torch.arange(split_len, dtype=dtype) + packed_seq._mrope_temporal_offset
        else:
            if text_position_ids.shape != (3, len(text_ids)):
                raise ValueError(
                    f"Expected text_position_ids shape (3, {len(text_ids)}), got {tuple(text_position_ids.shape)}"
                )
            native = text_position_ids.to(device="cpu", dtype=dtype)
            if bos_count:
                native = native + 1
                native = torch.cat((torch.zeros((3, 1), dtype=dtype), native), dim=1)
            native = native + packed_seq._mrope_temporal_offset
            cut = bos_count + insert_index
            prefix_native = native[:, :cut]
            suffix_native = native[:, cut:] + num_soft_plan_tokens
            prefix_max = (
                prefix_native.max()
                if prefix_native.numel() > 0
                else torch.tensor(packed_seq._mrope_temporal_offset - 1, dtype=dtype)
            )
            p_positions = (
                torch.arange(num_soft_plan_tokens, dtype=dtype).unsqueeze(0).expand(3, -1)
                + prefix_max
                + 1
            )
            suffix_max = suffix_native.max() if suffix_native.numel() > 0 else p_positions.max()
            tail_start = torch.maximum(p_positions.max(), suffix_max) + 1
            tail_positions = (
                torch.arange(2, dtype=dtype).unsqueeze(0).expand(3, -1) + tail_start
            )
            positions = torch.cat(
                (
                    prefix_native,
                    p_positions,
                    suffix_native,
                    tail_positions,
                ),
                dim=1,
            )
        packed_seq.position_ids.append(positions)
        packed_seq._mrope_temporal_offset = positions.max().item() + 1
    else:
        packed_seq.position_ids.extend(range(curr_rope_id, curr_rope_id + split_len))

    packed_seq.attn_modes.append("causal")
    packed_seq.split_lens.append(split_len)
    packed_seq.curr = raw_start + split_len
    return curr_rope_id + split_len, split_len, split_len, (plan_start, plan_start + num_soft_plan_tokens)


def pack_soft_plan_tokens(
    packed_seq: PackedSequence,
    num_tokens: int,
    curr_rope_id: int,
    use_float_positions: bool = False,
) -> tuple[int, int]:
    """Append learned soft-plan positions to the current causal split.

    Soft-plan tokens have no tokenizer IDs and no CE targets. They are routed
    through the understanding projections by extending the preceding causal
    split, while their embeddings are scattered later by the model.
    """
    if num_tokens <= 0:
        return curr_rope_id, 0
    assert isinstance(packed_seq.soft_plan_indexes, list)
    assert isinstance(packed_seq.position_ids, list)
    if not packed_seq.attn_modes or packed_seq.attn_modes[-1] != "causal":
        raise ValueError("Soft-plan tokens must immediately follow a causal text/SOG split.")

    start = packed_seq.curr
    end = start + num_tokens
    packed_seq.soft_plan_indexes.extend(range(start, end))
    packed_seq.soft_plan_ranges.append((start, end))

    if packed_seq._use_mrope:
        dtype = torch.float32 if use_float_positions else torch.long
        positions = torch.arange(num_tokens, dtype=dtype) + packed_seq._mrope_temporal_offset
        packed_seq.position_ids.append(positions.unsqueeze(0).expand(3, -1))
        packed_seq._mrope_temporal_offset += num_tokens
    else:
        packed_seq.position_ids.extend(range(curr_rope_id, curr_rope_id + num_tokens))

    packed_seq.split_lens[-1] += num_tokens
    packed_seq.curr = end
    return curr_rope_id + num_tokens, num_tokens


def pack_vision_tokens(
    packed_seq: PackedSequence,
    input_vision_tokens: torch.Tensor,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    curr_rope_id: int,
    latent_patch_size: int = 1,
    vision_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    vision_temporal_positions: torch.Tensor | None = None,
) -> int:
    """Pack vision tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_vision_tokens: Vision latent tokens (C, T, H, W).
        condition_frame_indexes_vision: Indexes of conditioning frames.
        input_timestep: Diffusion timestep. Either a float (teacher_forcing/none — all frames
            share the same sigma) or a Tensor(T_max,) (diffusion_forcing — per-frame sigma;
            indexed as input_timestep[frame_idx] for each noisy frame).
        curr_rope_id: Current RoPE position ID.
        latent_patch_size: Patch size for latent patchification.
        vision_fps: Frames per second of the video. Used when enable_fps_modulation=True.
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS.
        base_fps: Base FPS for normalization (default 24.0).
        temporal_compression_factor: VAE temporal compression factor (default 4).
        vision_temporal_positions: Optional explicit temporal coordinate per latent
            frame, shape ``(T,)``. Used by UniAE to account for kept boundary latents.
    Returns:
        Vision split length.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    vision_split_len = 0

    # Initialize vision modality if not present.
    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()

    # Ensure vision modality is in build mode
    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)

    # Compute position IDs for image patches
    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    if latent_patch_size < 1:
        raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}")
    # Use ceil to support latent dims not divisible by patch size (padding handled in network)
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    # Add image token indexes and loss information
    num_vision_tokens = latent_t * patch_h * patch_w
    packed_seq.vision.sequence_indexes.extend(range(curr, curr + num_vision_tokens))

    # Supervise vision tokens based on conditioning frames
    condition_set = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    assert isinstance(packed_seq.vision.condition_mask, list)

    vision_condition_mask = torch.zeros(
        (latent_t, 1, 1), device=input_vision_tokens.device, dtype=input_vision_tokens.dtype
    )  # [T,1,1]
    for frame_idx in condition_set:
        vision_condition_mask[frame_idx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set],
        device=input_vision_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_frames]
    assert isinstance(packed_seq.vision.noisy_frame_indexes, list)
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    frame_token_stride = patch_h * patch_w
    for frame_idx in range(latent_t):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.vision.mse_loss_indexes.extend(range(frame_start, frame_end))
        if isinstance(input_timestep, torch.Tensor):
            frame_ts = input_timestep[frame_idx].item()
        else:
            frame_ts = input_timestep
        packed_seq.vision.timesteps.extend([frame_ts] * frame_token_stride)

    curr += num_vision_tokens
    vision_split_len += num_vision_tokens

    # Update position IDs for image split
    if packed_seq._use_mrope:
        # Determine FPS for this vision segment (None disables FPS modulation)
        effective_fps = vision_fps if enable_fps_modulation else None
        if vision_temporal_positions is not None:
            vision_temporal_positions = vision_temporal_positions.to(device="cpu", dtype=torch.float32)  # [T]

        vision_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=packed_seq._mrope_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
            temporal_positions=vision_temporal_positions,
            actual_temporal_compression_factor=temporal_compression_factor,
        )  # vision_mrope_ids: [3,N_vision_tokens]
        packed_seq.position_ids.append(vision_mrope_ids)
    else:
        # All image tokens share the same RoPE position ID
        packed_seq.position_ids.extend([curr_rope_id] * vision_split_len)

    packed_seq.curr = curr
    return vision_split_len


def pack_action_tokens(
    packed_seq: PackedSequence,
    input_action_tokens: torch.Tensor,
    condition_frame_indexes_action: list[int],
    input_timestep: float,
    curr_rope_id: int,
    action_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    action_fps: float | None = None,
    base_temporal_compression_factor: int | None = None,
    action_start_frame_offset: int = 1,
) -> int:
    """Pack action tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_action_tokens: Action latent tokens (T, D).
        condition_frame_indexes_action: Indexes of conditioning action steps.
        input_timestep: Diffusion timestep.
        curr_rope_id: Current RoPE position ID.
        action_temporal_offset: Temporal offset for action mRoPE IDs (typically
            the vision start offset so action aligns temporally with vision).
        enable_fps_modulation: If True, scale temporal position IDs based on FPS.
        base_fps: Base FPS for normalization (default 24.0).
        action_fps: Frames per second of the action data. Used when enable_fps_modulation=True.
        base_temporal_compression_factor: Base temporal compression factor for FPS scaling.
            Should be set to the vision temporal compression factor (e.g. 4) so that action
            tokens advance at frame rate (4x finer) relative to vision latent frames.
            Only affects behavior when FPS modulation is enabled.
        action_start_frame_offset: Frame offset for aligning action[0] with the
            corresponding vision frame. Default 1 aligns action[0] with vision frame 1.
    Returns:
        Number of action tokens added.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    action_split_len = input_action_tokens.shape[0]

    # Initialize action modality if not present
    if packed_seq.action is None:
        packed_seq.action = ModalityData()

    # Ensure action modality is in build mode
    assert isinstance(packed_seq.action.sequence_indexes, list)
    assert isinstance(packed_seq.action.mse_loss_indexes, list)
    assert isinstance(packed_seq.action.timesteps, list)
    assert isinstance(packed_seq.action.tokens, list)

    # Add token indexes and loss information
    action_indexes = list(range(curr, curr + action_split_len))
    packed_seq.action.sequence_indexes.extend(action_indexes)
    packed_seq.action.token_shapes.append((action_split_len,))
    packed_seq.action.tokens.append(input_action_tokens)

    condition_set = {idx for idx in condition_frame_indexes_action if 0 <= idx < action_split_len}
    assert isinstance(packed_seq.action.condition_mask, list)

    action_condition_mask = torch.zeros(
        (action_split_len, 1), device=input_action_tokens.device, dtype=input_action_tokens.dtype
    )  # [T_action,1]
    for frame_idx in condition_set:
        action_condition_mask[frame_idx, 0] = 1.0
    packed_seq.action.condition_mask.append(action_condition_mask)

    action_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(action_split_len) if idx not in condition_set],
        device=input_action_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_action_frames]
    assert isinstance(packed_seq.action.noisy_frame_indexes, list)
    packed_seq.action.noisy_frame_indexes.append(action_noisy_frame_indexes)

    frame_token_stride = 1  # Action has 1 token per frame (no spatial dimension)
    for frame_idx in range(action_split_len):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.action.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.action.timesteps.extend([input_timestep] * frame_token_stride)

    # Update RoPE position IDs for action tokens.
    if packed_seq._use_mrope:
        # 3D mRoPE: action tokens use a 1x1 spatial grid with start_frame_offset=1
        # so action[0] (null token) aligns with vision frame 1, not frame 0.
        effective_fps = action_fps if enable_fps_modulation else None

        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=action_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Action is at frame rate (no temporal compression)
            base_temporal_compression_factor=base_temporal_compression_factor,
            start_frame_offset=action_start_frame_offset,  # Align action[0] with vision frame action_start_frame_offset
        )  # action_mrope_ids: [3,N_action_tokens]
        packed_seq.position_ids.append(action_mrope_ids)
        # Note: we don't update _mrope_temporal_offset here because action tokens
        # share the temporal space with vision tokens (they run in parallel).
    else:
        # All action tokens share the SAME RoPE position as vision tokens (see docs/sequence_packing.md).
        packed_seq.position_ids.extend([curr_rope_id] * action_split_len)

    packed_seq.curr = curr + action_split_len
    return action_split_len


def pack_sound_tokens(
    packed_seq: PackedSequence,
    input_sound_tokens: torch.Tensor,
    condition_frame_indexes_sound: list[int],
    input_timestep: float,
    curr_rope_id: int,
    sound_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_fps: float | None = None,
    sound_base_temporal_compression_factor: int | None = None,
) -> int:
    """Pack sound/audio tokens into the sequence.

    Sound latents have shape [C, T] where C is channels and T is temporal frames.
    Sound tokens are added to the unified generation split to maintain SequencePack's
    2-split invariant (causal + full).

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_sound_tokens: Sound latent tokens (C, T).
        condition_frame_indexes_sound: Indexes of conditioning frames.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
        input_timestep: Diffusion timestep.
        curr_rope_id: Current RoPE position ID.
        sound_temporal_offset: Temporal offset for m-RoPE position IDs (aligned with vision start).
        enable_fps_modulation: If True, scale temporal positions by FPS ratio.
        base_fps: Base FPS for normalization (default 24.0).
        sound_fps: Sound latent FPS (e.g., 25.0). Used for FPS-aware m-RoPE positions.
        sound_base_temporal_compression_factor: Base temporal compression factor for sound FPS scaling.
            ``None`` preserves the current behavior where sound advances at ``base_fps`` positions/sec.

    Returns:
        Number of sound tokens added.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr

    # Sound latent shape: [C, T] → T tokens
    _, sound_split_len = input_sound_tokens.shape

    # Initialize sound modality if not present
    if packed_seq.sound is None:
        packed_seq.sound = ModalityData()

    # Ensure sound modality is in build mode
    assert isinstance(packed_seq.sound.sequence_indexes, list)
    assert isinstance(packed_seq.sound.mse_loss_indexes, list)
    assert isinstance(packed_seq.sound.timesteps, list)
    assert isinstance(packed_seq.sound.tokens, list)

    # Add token indexes - sound uses (T, 1, 1) shape for compatibility with 3D RoPE
    packed_seq.sound.token_shapes.append((sound_split_len, 1, 1))
    packed_seq.sound.sequence_indexes.extend(range(curr, curr + sound_split_len))
    packed_seq.sound.tokens.append(input_sound_tokens)

    # Supervise sound tokens based on conditioning frames
    condition_set = {idx for idx in condition_frame_indexes_sound if 0 <= idx < sound_split_len}
    assert isinstance(packed_seq.sound.condition_mask, list)

    # Condition mask: shape (T, 1) — 1 = clean/conditioning, 0 = noised/supervised
    sound_condition_mask = torch.zeros(
        (sound_split_len, 1), device=input_sound_tokens.device, dtype=input_sound_tokens.dtype
    )  # [T_sound,1]
    for frame_idx in condition_set:
        sound_condition_mask[frame_idx, 0] = 1.0
    packed_seq.sound.condition_mask.append(sound_condition_mask)

    sound_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(sound_split_len) if idx not in condition_set],
        device=input_sound_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_sound_frames]
    assert isinstance(packed_seq.sound.noisy_frame_indexes, list)
    packed_seq.sound.noisy_frame_indexes.append(sound_noisy_frame_indexes)

    # Add to MSE loss indexes and timesteps for non-conditioning frames
    for frame_idx in range(sound_split_len):
        if frame_idx in condition_set:
            continue
        # Sound has 1 token per frame (no spatial dimension)
        frame_start = curr + frame_idx
        frame_end = frame_start + 1
        packed_seq.sound.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.sound.timesteps.extend([input_timestep])

    # Update RoPE position IDs for sound tokens.
    if packed_seq._use_mrope:
        # 3D mRoPE: sound tokens use a 1x1 spatial grid, aligned with vision temporal positions.
        # sound[0] aligns with vision frame 0 (start_frame_offset=0, unlike action which offsets by 1).
        effective_fps = sound_fps if enable_fps_modulation else None

        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=sound_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Sound latent is already at sound_latent_fps (no further compression)
            base_temporal_compression_factor=sound_base_temporal_compression_factor,
            start_frame_offset=0,  # Sound[0] aligns with vision frame 0
        )  # sound_mrope_ids: [3,N_sound_tokens]
        packed_seq.position_ids.append(sound_mrope_ids)
        # Note: we don't update _mrope_temporal_offset here because sound tokens
        # share the temporal space with vision tokens (they run in parallel).
    else:
        # All sound tokens share the SAME RoPE position as vision/action tokens (unified generation split).
        packed_seq.position_ids.extend([curr_rope_id] * sound_split_len)

    packed_seq.curr = curr + sound_split_len
    return sound_split_len
