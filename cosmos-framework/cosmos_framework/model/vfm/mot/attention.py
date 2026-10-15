# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from cosmos_framework.data.vfm.sequence_packing import SampleModalitySpans
from cosmos_framework.data.vfm.sequence_packing.natten import (
    generate_natten_metadata,
    generate_temporal_causal_natten_metadata,
)
from cosmos_framework.data.vfm.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.attention import (
    attention,
    merge_attentions,
    multi_dimensional_attention_varlen,
)
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryValue

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    try:  # pragma: no cover - depends on the installed torch._dynamo surface.
        import torch._dynamo as _dynamo

        # Same guard as NVlabs/FastGen fastgen/networks/Wan/network_causal.py
        # FlexAttention setup: higher-order ops can make Dynamo's DDP optimizer
        # raise NotImplemented under distributed training.
        _dynamo.config.optimize_ddp = False
    except Exception:
        pass

    flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")
except ImportError:  # pragma: no cover - depends on the installed PyTorch build.
    create_block_mask = None
    flex_attention = None


class SplitInfo:
    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
        sample_modality_spans: list[SampleModalitySpans] | None = None,
        action_video_attention_mode: str = "legacy_full",
        soft_plan_num_tokens: int = 0,
        mid_soft: bool = False,
        soft_plan_ranges: list[tuple[int, int]] | None = None,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which may
        be padded to max sequence length by upstream packers.
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)

        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens

        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens
        self.sample_modality_spans = sample_modality_spans
        self.action_video_attention_mode = action_video_attention_mode
        self.soft_plan_num_tokens = soft_plan_num_tokens
        self.mid_soft = bool(mid_soft)
        self.soft_plan_ranges = soft_plan_ranges

        # Multi-control transfer fields (set post-construction in cosmos3_vfm_network.py).
        # Gen-relative token ranges for each control stream, one tuple (start, end) per control.
        self.control_stream_token_ranges: list[tuple[int, int]] | None = None
        # Gen-relative token range (start, end) for the noisy target tokens.
        self.noisy_token_range: tuple[int, int] | None = None
        # Per-control scalar weights; parallel to control_stream_token_ranges.
        self.control_weights: list[float] | None = None

        # FastWAM FlexAttention BlockMask cache. SplitInfo is created once per
        # packed forward and shared by all decoder layers, so this cache avoids
        # rebuilding the same block-sparse metadata in every layer.
        self._fastwam_flex_block_mask_cache: dict[tuple, object] = {}


AttentionMaskType = SplitInfo


_dotproduct_attention_cache = {}
_FASTWAM_FLEX_BLOCK_SIZE = 128
_FASTWAM_FLEX_LENGTH_ALIGNMENT = 128
_FASTWAM_ACTION_VIDEO_ATTENTION_MODES = {"fastwam_first_frame", "fastwam_first_last_frame"}


def two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_meta: SplitInfo | None = None,
) -> SequencePack:
    """
    Performs two-way attention with causal and full attention.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)

    sample_offsets = packed_query_states["sample_offsets"]

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # NOTE: cosmos_framework attention is BSHD in, BSHD out
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]

    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    if (
        attention_meta is not None
        and attention_meta.action_video_attention_mode in _FASTWAM_ACTION_VIDEO_ATTENTION_MODES
    ):
        full_res = _fastwam_first_frame_full_attention(
            full_q=full_q,
            full_q_offsets=full_q_offsets,
            all_k=get_all_seq(packed_key_states),
            all_v=get_all_seq(packed_value_states),
            sample_offsets=sample_offsets,
            attention_meta=attention_meta,
        )
    else:
        full_res = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            get_all_seq(packed_key_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
            get_all_seq(packed_value_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=sample_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_sample_len"],
        )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def _offsets_from_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def soft_plan_bottleneck_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_meta: SplitInfo,
) -> SequencePack:
    """Run causal soft-plan bottleneck attention with two varlen kernels.

    In the legacy layout, ``T`` is the causal prompt including
    ViT/caption/EOS/SOG and ``P`` is its learned suffix. Visibility is::

        [T, P] -> [T, P] (causal)
        G -> P + G (full)

    Thus each ``P_i`` sees all of ``T`` and only ``P_0..P_i``. Generation K/V
    never contains ``T``.

    With ``mid_soft=True`` the causal split is ``[T_pre, P, T_post]``. Its
    ordinary causal pass gives P only ``T_pre`` and earlier P tokens, while G
    receives exactly ``[P, T_post, G]`` as K/V.
    """
    num_soft = attention_meta.soft_plan_num_tokens
    if num_soft <= 0:
        raise ValueError("soft_plan_bottleneck_attention requires soft_plan_num_tokens > 0.")

    causal_q, causal_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, causal_v_offsets = get_causal_seq(packed_value_states)
    full_q, full_offsets = get_full_only_seq(packed_query_states)
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)
    full_v, full_v_offsets = get_full_only_seq(packed_value_states)

    causal_lengths = [
        int(split_len)
        for split_len, attn_mode in zip(attention_meta.split_lens, attention_meta.attn_modes)
        if attn_mode == "causal"
    ]
    gen_lengths = [
        int(split_len)
        for split_len, attn_mode in zip(attention_meta.split_lens, attention_meta.attn_modes)
        if attn_mode == "full"
    ]
    if not (
        causal_offsets.numel()
        == causal_k_offsets.numel()
        == causal_v_offsets.numel()
        == full_offsets.numel()
        == full_k_offsets.numel()
        == full_v_offsets.numel()
        == len(causal_lengths) + 1
        == len(gen_lengths) + 1
    ):
        raise ValueError("Soft-plan attention requires one causal and one generation split per sample.")

    num_samples = len(causal_lengths)
    causal_starts = [0]
    full_starts = [0]
    for causal_len, gen_len in zip(causal_lengths, gen_lengths):
        causal_starts.append(causal_starts[-1] + causal_len)
        full_starts.append(full_starts[-1] + gen_len)

    plan_gen_k_chunks: list[torch.Tensor] = []
    plan_gen_v_chunks: list[torch.Tensor] = []

    if attention_meta.mid_soft:
        ranges = attention_meta.soft_plan_ranges
        if ranges is None or len(ranges) != num_samples:
            raise ValueError("mid-soft attention requires one absolute soft_plan range per sample.")
        sample_abs_starts = [0]
        for length in attention_meta.sample_lens:
            sample_abs_starts.append(sample_abs_starts[-1] + int(length))
        for sample_idx in range(num_samples):
            causal_start, causal_end = causal_starts[sample_idx : sample_idx + 2]
            full_start, full_end = full_starts[sample_idx : sample_idx + 2]
            plan_abs_start, plan_abs_end = ranges[sample_idx]
            sample_abs_start = sample_abs_starts[sample_idx]
            local_plan_start = int(plan_abs_start) - sample_abs_start
            local_plan_end = int(plan_abs_end) - sample_abs_start
            if local_plan_start < 0 or local_plan_end > causal_lengths[sample_idx] or local_plan_end <= local_plan_start:
                raise ValueError(
                    f"Invalid mid-soft range {ranges[sample_idx]} for sample {sample_idx} "
                    f"with causal length {causal_lengths[sample_idx]}"
                )
            if local_plan_end - local_plan_start != num_soft:
                raise ValueError("mid-soft range length must equal soft_plan_num_tokens.")
            # G sees P and all trailing metadata, but never the instruction
            # prefix before P.  Its own tokens remain visible as usual.
            plan_gen_k_chunks.append(
                torch.cat(
                    (
                        causal_k[causal_start + local_plan_start : causal_start + local_plan_end],
                        causal_k[causal_start + local_plan_end : causal_end],
                        full_k[full_start:full_end],
                    ),
                    dim=0,
                )
            )
            plan_gen_v_chunks.append(
                torch.cat(
                    (
                        causal_v[causal_start + local_plan_start : causal_start + local_plan_end],
                        causal_v[causal_start + local_plan_end : causal_end],
                        full_v[full_start:full_end],
                    ),
                    dim=0,
                )
            )
    else:
        for sample_idx in range(num_samples):
            causal_start, causal_end = causal_starts[sample_idx : sample_idx + 2]
            full_start, full_end = full_starts[sample_idx : sample_idx + 2]
            causal_len = causal_lengths[sample_idx]
            if causal_len <= num_soft:
                raise ValueError(
                    f"Sample {sample_idx} causal length {causal_len} must exceed soft_plan_num_tokens={num_soft}."
                )
            plan_start = causal_end - num_soft
            plan_gen_k_chunks.append(torch.cat((causal_k[plan_start:causal_end], full_k[full_start:full_end]), dim=0))
            plan_gen_v_chunks.append(torch.cat((causal_v[plan_start:causal_end], full_v[full_start:full_end]), dim=0))

    plan_gen_lengths = [int(chunk.shape[0]) for chunk in plan_gen_k_chunks]
    plan_gen_offsets = _offsets_from_lengths(plan_gen_lengths, causal_q.device)

    causal_res = (
        attention(
            causal_q.unsqueeze(0),
            causal_k.unsqueeze(0),
            causal_v.unsqueeze(0),
            cumulative_seqlen_Q=causal_offsets,
            cumulative_seqlen_KV=causal_k_offsets,
            max_seqlen_Q=max(causal_lengths),
            max_seqlen_KV=max(causal_lengths),
            is_causal=True,
            causal_type=CausalType.DontCare,
        )
        .squeeze(0)
        .flatten(-2, -1)
    )

    full_actual_len = full_starts[-1]
    plan_gen_k = torch.cat(plan_gen_k_chunks, dim=0)
    plan_gen_v = torch.cat(plan_gen_v_chunks, dim=0)
    gen_res = (
        attention(
            full_q[:full_actual_len].unsqueeze(0),
            plan_gen_k.unsqueeze(0),
            plan_gen_v.unsqueeze(0),
            cumulative_seqlen_Q=full_offsets,
            cumulative_seqlen_KV=plan_gen_offsets,
            max_seqlen_Q=max(gen_lengths),
            max_seqlen_KV=max(plan_gen_lengths),
        )
        .squeeze(0)
        .flatten(-2, -1)
    )

    full_out = full_q.new_zeros((full_q.shape[0], gen_res.shape[-1]))
    full_out[:full_actual_len] = gen_res
    return from_mode_splits(causal_res, full_out, packed_query_states)


def _repeat_kv_for_gqa(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[1]
    if num_kv_heads == num_query_heads:
        return x
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads}).")
    return x.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


def _dense_masked_attention_per_sample(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed_mask: torch.Tensor,
) -> torch.Tensor:
    q = q.transpose(0, 1).unsqueeze(0)  # [1,Hq,Lq,D]
    num_query_heads = q.shape[1]
    k = _repeat_kv_for_gqa(k, num_query_heads).transpose(0, 1).unsqueeze(0)  # [1,Hq,Lk,D]
    v = _repeat_kv_for_gqa(v, num_query_heads).transpose(0, 1).unsqueeze(0)  # [1,Hq,Lk,D]
    attn_mask = allowed_mask.unsqueeze(0).unsqueeze(0)  # [1,1,Lq,Lk]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.squeeze(0).transpose(0, 1)  # [Lq,Hq,D]


def _normalize_fastwam_attention_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend == "sdpa":
        return "sdpa"
    if backend == "flex":
        return "flex"
    raise ValueError(f"FastWAM attention backend must be 'sdpa' or 'flex', got {value!r}.")


def _fastwam_attention_backend() -> str:
    backend = os.environ.get("FASTWAM_ATTENTION_BACKEND", "flex")
    return _normalize_fastwam_attention_backend(backend)


def _fastwam_spans_cache_key(sample_spans: SampleModalitySpans) -> tuple:
    def _span_key(spans):
        return tuple((span.start, span.end, span.token_shape) for span in spans)

    return (
        _span_key(sample_spans.text),
        _span_key(sample_spans.vision),
        _span_key(sample_spans.action),
        _span_key(sample_spans.sound),
    )


def _fastwam_flex_cache_key(
    *,
    full_q: torch.Tensor,
    all_k: torch.Tensor,
    attention_meta: SplitInfo,
) -> tuple:
    full_lens = _fastwam_full_lengths_from_meta(attention_meta)
    padded_q_len = _fastwam_flex_padded_len(max(full_lens, default=0))
    padded_kv_len = _fastwam_flex_padded_len(max(attention_meta.sample_lens, default=0))
    return (
        attention_meta.action_video_attention_mode,
        full_q.device.type,
        full_q.device.index,
        padded_q_len,
        padded_kv_len,
        _FASTWAM_FLEX_BLOCK_SIZE,
        _FASTWAM_FLEX_LENGTH_ALIGNMENT,
        tuple(full_lens),
        tuple(int(x) for x in attention_meta.sample_lens),
        tuple(_fastwam_spans_cache_key(spans) for spans in (attention_meta.sample_modality_spans or [])),
    )


def _fastwam_full_lengths_from_meta(attention_meta: SplitInfo) -> list[int]:
    return [
        int(split_len)
        for split_len, attn_mode in zip(attention_meta.split_lens, attention_meta.attn_modes, strict=True)
        if attn_mode == "full"
    ]


def _fastwam_flex_padded_len(length: int) -> int:
    if length <= 0:
        return 0
    return ((length + _FASTWAM_FLEX_LENGTH_ALIGNMENT - 1) // _FASTWAM_FLEX_LENGTH_ALIGNMENT) * (
        _FASTWAM_FLEX_LENGTH_ALIGNMENT
    )


def _fastwam_frame_token_len(span) -> tuple[int, int, int, int]:
    if span.token_shape is None or len(span.token_shape) != 3:
        raise ValueError(f"Vision span requires token_shape=(T,H,W), got {span.token_shape}.")
    latent_t, patch_h, patch_w = span.token_shape
    frame_len = patch_h * patch_w
    return latent_t, patch_h, patch_w, frame_len


def _fastwam_visible_vision_ranges(span, action_video_attention_mode: str) -> list[tuple[int, int]]:
    latent_t, _, _, frame_len = _fastwam_frame_token_len(span)
    first = (span.start, span.start + frame_len)
    if action_video_attention_mode == "fastwam_first_frame" or latent_t <= 1:
        return [first]
    if action_video_attention_mode == "fastwam_first_last_frame":
        last_start = span.start + (latent_t - 1) * frame_len
        return [first, (last_start, last_start + frame_len)]
    raise ValueError(f"Unsupported FastWAM action/video attention mode: {action_video_attention_mode!r}.")


def _fastwam_full_q_interval(
    *,
    span_start: int,
    span_end: int,
    query_abs_start: int,
    query_abs_end: int,
) -> tuple[int, int]:
    start = max(span_start, query_abs_start)
    end = min(span_end, query_abs_end)
    if start >= end:
        return 0, 0
    return start - query_abs_start, end - query_abs_start


def _fastwam_flex_metadata_tensors(
    *,
    full_lens: list[int],
    sample_lens: list[int],
    sample_spans: list[SampleModalitySpans],
    action_video_attention_mode: str,
    device: torch.device,
    q_len: int | None = None,
    kv_len: int | None = None,
) -> dict[str, torch.Tensor]:
    batch_size = len(sample_lens)
    max_full_len = max(full_lens, default=0) if q_len is None else q_len
    max_sample_len = max(sample_lens, default=0) if kv_len is None else kv_len
    if max(full_lens, default=0) > max_full_len:
        raise ValueError(f"q_len={max_full_len} is smaller than max full length {max(full_lens)}.")
    if max(sample_lens, default=0) > max_sample_len:
        raise ValueError(f"kv_len={max_sample_len} is smaller than max sample length {max(sample_lens)}.")
    q_valid = torch.zeros((batch_size, max_full_len), dtype=torch.bool, device=device)
    q_visible_vision = torch.zeros_like(q_valid)
    q_vision = torch.zeros_like(q_valid)
    q_action = torch.zeros_like(q_valid)

    kv_valid = torch.zeros((batch_size, max_sample_len), dtype=torch.bool, device=device)
    kv_text = torch.zeros_like(kv_valid)
    kv_visible_vision = torch.zeros_like(kv_valid)
    kv_vision = torch.zeros_like(kv_valid)
    kv_action = torch.zeros_like(kv_valid)

    sample_abs_start = 0
    for sample_idx, (full_len, sample_len, spans) in enumerate(zip(full_lens, sample_lens, sample_spans, strict=True)):
        query_abs_start = sample_abs_start + sample_len - full_len
        query_abs_end = query_abs_start + full_len
        sample_abs_end = sample_abs_start + sample_len

        q_valid[sample_idx, :full_len] = True
        kv_valid[sample_idx, :sample_len] = True

        for span in spans.text:
            kv_text[sample_idx, span.start - sample_abs_start : span.end - sample_abs_start] = True

        for span in spans.vision:
            for visible_start, visible_end in _fastwam_visible_vision_ranges(span, action_video_attention_mode):
                kv_visible_vision[
                    sample_idx,
                    visible_start - sample_abs_start : visible_end - sample_abs_start,
                ] = True
                q_start, q_end = _fastwam_full_q_interval(
                    span_start=visible_start,
                    span_end=visible_end,
                    query_abs_start=query_abs_start,
                    query_abs_end=query_abs_end,
                )
                q_visible_vision[sample_idx, q_start:q_end] = True
            kv_vision[sample_idx, span.start - sample_abs_start : span.end - sample_abs_start] = True

            q_start, q_end = _fastwam_full_q_interval(
                span_start=span.start,
                span_end=span.end,
                query_abs_start=query_abs_start,
                query_abs_end=query_abs_end,
            )
            q_vision[sample_idx, q_start:q_end] = True

        for span in spans.action:
            kv_action[sample_idx, span.start - sample_abs_start : span.end - sample_abs_start] = True
            q_start, q_end = _fastwam_full_q_interval(
                span_start=span.start,
                span_end=span.end,
                query_abs_start=query_abs_start,
                query_abs_end=query_abs_end,
            )
            q_action[sample_idx, q_start:q_end] = True

        sample_abs_start = sample_abs_end

    return {
        "q_valid": q_valid,
        "q_visible_vision": q_visible_vision,
        "q_vision": q_vision,
        "q_action": q_action,
        "kv_valid": kv_valid,
        "kv_text": kv_text,
        "kv_visible_vision": kv_visible_vision,
        "kv_vision": kv_vision,
        "kv_action": kv_action,
    }


def _fastwam_flex_mask_mod(metadata: dict[str, torch.Tensor]):
    def mask_mod(b, h, q_idx, kv_idx):
        del h
        q_valid = metadata["q_valid"][b, q_idx]
        kv_valid = metadata["kv_valid"][b, kv_idx]
        action_allowed = (
            metadata["kv_text"][b, kv_idx] | metadata["kv_visible_vision"][b, kv_idx] | metadata["kv_action"][b, kv_idx]
        )
        visible_vision_allowed = metadata["kv_text"][b, kv_idx] | metadata["kv_visible_vision"][b, kv_idx]
        future_vision_allowed = metadata["kv_text"][b, kv_idx] | metadata["kv_vision"][b, kv_idx]
        legacy_allowed = torch.ones_like(action_allowed, dtype=torch.bool)

        allowed = torch.where(
            metadata["q_action"][b, q_idx],
            action_allowed,
            torch.where(
                metadata["q_visible_vision"][b, q_idx],
                visible_vision_allowed,
                torch.where(metadata["q_vision"][b, q_idx], future_vision_allowed, legacy_allowed),
            ),
        )
        padded_query_allowed = (~q_valid) & (kv_idx == 0)
        return (q_valid & kv_valid & allowed) | padded_query_allowed

    return mask_mod


def _fastwam_pad_flex_inputs(
    *,
    full_q: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    full_lens: list[int],
    sample_lens: list[int],
    q_len: int | None = None,
    kv_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(sample_lens)
    max_full_len = max(full_lens, default=0) if q_len is None else q_len
    max_sample_len = max(sample_lens, default=0) if kv_len is None else kv_len
    q = full_q.new_zeros((batch_size, full_q.shape[1], max_full_len, full_q.shape[2]))
    k = all_k.new_zeros((batch_size, all_k.shape[1], max_sample_len, all_k.shape[2]))
    v = all_v.new_zeros((batch_size, all_v.shape[1], max_sample_len, all_v.shape[2]))

    full_offset = 0
    sample_offset = 0
    for sample_idx, (full_len, sample_len) in enumerate(zip(full_lens, sample_lens, strict=True)):
        q[sample_idx, :, :full_len, :] = full_q[full_offset : full_offset + full_len].transpose(0, 1)
        k[sample_idx, :, :sample_len, :] = all_k[sample_offset : sample_offset + sample_len].transpose(0, 1)
        v[sample_idx, :, :sample_len, :] = all_v[sample_offset : sample_offset + sample_len].transpose(0, 1)
        full_offset += full_len
        sample_offset += sample_len

    return q.contiguous(), k.contiguous(), v.contiguous()


def _fastwam_unpad_flex_output(
    out: torch.Tensor,
    full_q: torch.Tensor,
    full_lens: list[int],
) -> torch.Tensor:
    pieces = [
        out[sample_idx, :, :full_len, :].transpose(0, 1)
        for sample_idx, full_len in enumerate(full_lens)
        if full_len > 0
    ]
    if pieces:
        out_packed = torch.cat(pieces, dim=0)
    else:
        out_packed = full_q.new_zeros((0, full_q.shape[1], full_q.shape[2]))
    if out_packed.shape[0] < full_q.shape[0]:
        pad = full_q.new_zeros(full_q.shape[0] - out_packed.shape[0], full_q.shape[1], full_q.shape[2])
        out_packed = torch.cat([out_packed, pad], dim=0)
    return out_packed.unsqueeze(0)


def _get_fastwam_flex_block_mask(
    *,
    full_q: torch.Tensor,
    all_k: torch.Tensor,
    attention_meta: SplitInfo,
):
    if flex_attention is None or create_block_mask is None:
        raise RuntimeError(
            "FASTWAM_ATTENTION_BACKEND=flex requires torch.nn.attention.flex_attention, "
            "but it is not available in this PyTorch build."
        )
    if attention_meta.sample_modality_spans is None:
        raise ValueError("fastwam_first_frame attention requires sample_modality_spans.")

    key = _fastwam_flex_cache_key(full_q=full_q, all_k=all_k, attention_meta=attention_meta)
    cached = attention_meta._fastwam_flex_block_mask_cache.get(key)
    if cached is not None:
        return cached

    full_lens = _fastwam_full_lengths_from_meta(attention_meta)
    q_len = _fastwam_flex_padded_len(max(full_lens, default=0))
    kv_len = _fastwam_flex_padded_len(max(attention_meta.sample_lens, default=0))
    metadata = _fastwam_flex_metadata_tensors(
        full_lens=full_lens,
        sample_lens=attention_meta.sample_lens,
        sample_spans=attention_meta.sample_modality_spans,
        action_video_attention_mode=attention_meta.action_video_attention_mode,
        device=full_q.device,
        q_len=q_len,
        kv_len=kv_len,
    )
    mask_mod = _fastwam_flex_mask_mod(metadata)
    block_mask = create_block_mask(
        mask_mod,
        B=len(attention_meta.sample_lens),
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=full_q.device,
        BLOCK_SIZE=_FASTWAM_FLEX_BLOCK_SIZE,
        _compile=False,
    )
    # Keep metadata tensors alive because mask_mod closes over them and Flex uses
    # mask_mod for partial blocks inside the compiled kernel.
    attention_meta._fastwam_flex_block_mask_cache[key] = (block_mask, metadata)
    return block_mask, metadata


def _fastwam_first_frame_full_attention_flex(
    *,
    full_q: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    attention_meta: SplitInfo,
) -> torch.Tensor:
    full_lens = _fastwam_full_lengths_from_meta(attention_meta)
    actual_full_len = sum(full_lens)
    full_q_for_attention = full_q[:actual_full_len]
    if actual_full_len == 0:
        return full_q.new_zeros((1, full_q.shape[0], full_q.shape[1], full_q.shape[2]))

    block_mask, metadata = _get_fastwam_flex_block_mask(
        full_q=full_q_for_attention,
        all_k=all_k,
        attention_meta=attention_meta,
    )
    q_len = int(metadata["q_valid"].shape[1])
    kv_len = int(metadata["kv_valid"].shape[1])
    q, k, v = _fastwam_pad_flex_inputs(
        full_q=full_q_for_attention,
        all_k=all_k,
        all_v=all_v,
        full_lens=full_lens,
        sample_lens=attention_meta.sample_lens,
        q_len=q_len,
        kv_len=kv_len,
    )
    out = flex_attention(
        query=q,
        key=k,
        value=v,
        block_mask=block_mask,
        enable_gqa=(full_q.shape[1] != all_k.shape[1]),
    )
    return _fastwam_unpad_flex_output(out, full_q, full_lens)


def _fastwam_allowed_mask_for_sample(
    *,
    query_abs_start: int,
    full_len: int,
    sample_abs_start: int,
    sample_len: int,
    sample_spans: SampleModalitySpans,
    action_video_attention_mode: str = "fastwam_first_frame",
    device: torch.device,
) -> torch.Tensor:
    query_abs = torch.arange(query_abs_start, query_abs_start + full_len, device=device)
    key_abs = torch.arange(sample_abs_start, sample_abs_start + sample_len, device=device)
    allowed = torch.zeros((full_len, sample_len), dtype=torch.bool, device=device)

    text_key = torch.zeros(sample_len, dtype=torch.bool, device=device)
    for span in sample_spans.text:
        text_key |= (key_abs >= span.start) & (key_abs < span.end)

    action_key = torch.zeros(sample_len, dtype=torch.bool, device=device)
    action_query = torch.zeros(full_len, dtype=torch.bool, device=device)
    for span in sample_spans.action:
        action_key |= (key_abs >= span.start) & (key_abs < span.end)
        action_query |= (query_abs >= span.start) & (query_abs < span.end)

    visible_vision_key = torch.zeros(sample_len, dtype=torch.bool, device=device)
    visible_vision_query = torch.zeros(full_len, dtype=torch.bool, device=device)
    vision_key = torch.zeros(sample_len, dtype=torch.bool, device=device)
    vision_query = torch.zeros(full_len, dtype=torch.bool, device=device)
    for span in sample_spans.vision:
        for visible_start, visible_end in _fastwam_visible_vision_ranges(span, action_video_attention_mode):
            visible_vision_key |= (key_abs >= visible_start) & (key_abs < visible_end)
            visible_vision_query |= (query_abs >= visible_start) & (query_abs < visible_end)
        vision_key |= (key_abs >= span.start) & (key_abs < span.end)
        vision_query |= (query_abs >= span.start) & (query_abs < span.end)

    action_allowed = text_key | visible_vision_key | action_key
    visible_vision_allowed = text_key | visible_vision_key
    future_vision_allowed = text_key | vision_key
    legacy_allowed = torch.ones(sample_len, dtype=torch.bool, device=device)

    allowed[action_query] = action_allowed
    allowed[visible_vision_query] = visible_vision_allowed
    allowed[vision_query & ~visible_vision_query] = future_vision_allowed
    allowed[~(action_query | vision_query)] = legacy_allowed
    return allowed


def _fastwam_first_frame_full_attention(
    *,
    full_q: torch.Tensor,
    full_q_offsets: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    sample_offsets: torch.Tensor,
    attention_meta: SplitInfo,
) -> torch.Tensor:
    del sample_offsets
    if attention_meta.sample_modality_spans is None:
        raise ValueError("fastwam_first_frame attention requires sample_modality_spans.")
    if full_q_offsets.shape[0] != len(attention_meta.sample_modality_spans) + 1:
        raise ValueError(
            "fastwam_first_frame attention expects one full split per sample; "
            f"got {full_q_offsets.shape[0] - 1} full splits for "
            f"{len(attention_meta.sample_modality_spans)} samples."
        )

    if _fastwam_attention_backend() == "flex":
        return _fastwam_first_frame_full_attention_flex(
            full_q=full_q,
            all_k=all_k,
            all_v=all_v,
            attention_meta=attention_meta,
        )

    outputs: list[torch.Tensor] = []
    full_offset = 0
    sample_abs_start = 0
    full_lens = _fastwam_full_lengths_from_meta(attention_meta)
    for full_len, sample_len, sample_spans in zip(
        full_lens, attention_meta.sample_lens, attention_meta.sample_modality_spans, strict=True
    ):
        # sample_spans maps absolute packed-token ranges to text / vision /
        # action. The mask below allows action queries to see text, first-frame
        # vision tokens, and action tokens, but not future video tokens.
        q_i = full_q[full_offset : full_offset + full_len]
        k_i = all_k[sample_abs_start : sample_abs_start + sample_len]
        v_i = all_v[sample_abs_start : sample_abs_start + sample_len]
        query_abs_start = sample_abs_start + sample_len - full_len
        allowed_mask = _fastwam_allowed_mask_for_sample(
            query_abs_start=query_abs_start,
            full_len=full_len,
            sample_abs_start=sample_abs_start,
            sample_len=sample_len,
            sample_spans=sample_spans,
            action_video_attention_mode=attention_meta.action_video_attention_mode,
            device=full_q.device,
        )
        outputs.append(_dense_masked_attention_per_sample(q_i, k_i, v_i, allowed_mask))
        full_offset += full_len
        sample_abs_start += sample_len

    full_out = torch.cat(outputs, dim=0)
    if full_offset < full_q.shape[0]:
        full_out = torch.cat([full_out, full_q.new_zeros(full_q.shape[0] - full_offset, *full_q.shape[1:])], dim=0)
    return full_out.unsqueeze(0)


def three_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
) -> SequencePack:
    """
    Performs three-way attention, with understanding and generations attentions fully decomposed,
    and allows sparsity / multi-dimensional masking in the generation tower.

    When attention_meta is provided with null_action_supertokens=True, zeros V for the first
    num_action_tokens_per_supertoken tokens of each sample's GEN sequence (null action
    supertokens for temporal causal training). The metadata encodes is_causal=(True, False):
    causal across T supertokens, full within each supertoken S.

    NOTE: the three-way decomposition is only done so we can handle sparsity in the gen tower,
    but a KEY assumption is that the "full" tokens all correspond to the same modality!
    We should be careful when extending this to beyond t2i and t2v.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    sample_offsets = packed_query_states["sample_offsets"]

    if attention_meta is not None and attention_meta.null_action_supertokens:
        # Zero V for the first num_action_tokens_per_supertoken tokens of each
        # sample's GEN sequence (null action supertokens at t=0).
        # out_i = Σ_j softmax(QKᵀ/√d)_j · V_j — terms with V_j=0 contribute exactly 0 to the output,
        # regardless of attention weights. Softmax mass is still allocated to these positions (not
        # redistributed), so this differs from hard key masking, but the output contribution is 0.
        full_v = full_v.clone()
        starts = full_q_offsets[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # NOTE: cosmos_framework attention is BSHD in, BSHD out
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]
    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # If there's no metadata, it's a dense layer
    if natten_metadata is None:
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_k_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_full_len"],
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]
    else:
        assert natten_metadata is not None
        full_sa, full_sa_lse = multi_dimensional_attention_varlen(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            metadata=natten_metadata,
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]

    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        return_lse=True,
    )  # full_ca: [1,N_full,heads,head_dim], full_ca_lse: [1,N_full,heads]

    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions(
        outputs=[full_sa, full_ca], lse_tensors=[full_sa_lse, full_ca_lse], torch_compile=False
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def multi_control_two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    split_info: SplitInfo,
) -> SequencePack:
    """Two-way attention for multi-control transfer inference.

    N independent single-control attention passes; noisy output = weighted sum.

    Layout of the "full/gen" segment (mirrors the packed batch built by ``build_transfer_batch``):

        full = [ctrl_1 | ctrl_2 | ... | ctrl_N | noisy]

    For each control i, one independent maskless SDPA is computed:

        ctrl_i and noisy both attend to KV = [text | ctrl_i | noisy]

    The final outputs are:
      - ctrl_i output: from pass i only
      - noisy output:  w_1 * noisy_out_1 + ... + w_N * noisy_out_N  (weighted sum)

    All SDPA calls are maskless → Flash Attention is always active.
    N=1, w=1.0 → identical to ``two_way_attention``.

    Padding safety:
      Both ``get_causal_seq`` and ``get_full_only_seq`` can return padded rows.
      We unpad to valid token counts before each SDPA so that padded rows
      never enter the softmax denominator.

    Args:
        packed_query/key/value_states: SequencePack for a single sample.
        split_info: SplitInfo carrying ``control_stream_token_ranges``,
            ``noisy_token_range``, and ``control_weights`` (all must be non-None).
    """
    assert split_info.control_stream_token_ranges is not None
    assert split_info.noisy_token_range is not None
    assert split_info.control_weights is not None

    ctrl_ranges = split_info.control_stream_token_ranges
    noisy_s, noisy_e = split_info.noisy_token_range
    weights = split_info.control_weights

    # ── 1. Text self-attention (causal, unchanged) ───────────────────────────
    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)

    use_dont_care_mask = causal_q_offsets is causal_k_offsets
    causal_res = attention(
        causal_q.unsqueeze(0),
        causal_k.unsqueeze(0),
        causal_v.unsqueeze(0),
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # [N_text, Hq*D]

    # ── 2. Extract unpadded full/gen tokens ──────────────────────────────────
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, _ = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    n_text = int(causal_k_offsets[-1])
    n_full = int(full_q_offsets[-1])

    # Unpad to avoid padded rows entering the softmax denominator.
    causal_k_v = causal_k[:n_text]  # [N_text, Hkv, D]
    causal_v_v = causal_v[:n_text]  # [N_text, Hkv, D]
    full_q_v = full_q[:n_full]  # [N_full, Hq,  D]
    full_k_v = full_k[:n_full]  # [N_full, Hkv, D]
    full_v_v = full_v[:n_full]  # [N_full, Hkv, D]

    noisy_q = full_q_v[noisy_s:noisy_e]  # [N_noisy, Hq,  D]
    noisy_k = full_k_v[noisy_s:noisy_e]  # [N_noisy, Hkv, D]
    noisy_v = full_v_v[noisy_s:noisy_e]  # [N_noisy, Hkv, D]

    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Maskless attention using cosmos_framework attention() → [N_q, Hq*D]."""
        n_q, n_kv = q.shape[0], k.shape[0]
        seqlens_q = torch.tensor([n_q], dtype=torch.int32, device=q.device)
        seqlens_kv = torch.tensor([n_kv], dtype=torch.int32, device=k.device)
        res = attention(
            q.unsqueeze(0),  # [1, N_q,  Hq,  D]
            k.unsqueeze(0),  # [1, N_kv, Hkv, D]
            v.unsqueeze(0),  # [1, N_kv, Hkv, D]
            seqlens_Q=seqlens_q,
            seqlens_KV=seqlens_kv,
            max_seqlen_Q=n_q,
            max_seqlen_KV=n_kv,
        )  # [1, N_q, Hq, D]
        return res.squeeze(0).flatten(-2, -1)  # [N_q, Hq*D]

    # ── 3. N independent single-control passes ────────────────────────────────
    # For each control i: KV = [text | ctrl_i | noisy] — maskless SDPA.
    # ctrl_i attends to [text, ctrl_i, noisy] → stored directly in full_out.
    # noisy  attends to [text, ctrl_i, noisy] → accumulated as weighted sum.
    full_out_v = full_q_v.new_zeros(n_full, causal_out.shape[-1])
    noisy_out_acc: torch.Tensor | None = None

    for i, (cs, ce) in enumerate(ctrl_ranges):
        ctrl_k_i = full_k_v[cs:ce]
        ctrl_v_i = full_v_v[cs:ce]
        ctrl_q_i = full_q_v[cs:ce]

        # KV context for this pass: [text | ctrl_i | noisy]
        kv_k_i = torch.cat([causal_k_v, ctrl_k_i, noisy_k], dim=0)
        kv_v_i = torch.cat([causal_v_v, ctrl_v_i, noisy_v], dim=0)

        # ctrl_i output — stored directly
        full_out_v[cs:ce] = _sdpa(ctrl_q_i, kv_k_i, kv_v_i)

        # noisy output for pass i — accumulate weighted sum
        noisy_out_i = _sdpa(noisy_q, kv_k_i, kv_v_i)
        if noisy_out_acc is None:
            noisy_out_acc = weights[i] * noisy_out_i
        else:
            noisy_out_acc = noisy_out_acc + weights[i] * noisy_out_i

    assert noisy_out_acc is not None
    full_out_v[noisy_s:noisy_e] = noisy_out_acc

    # Re-pad to original shape so downstream layers see consistent tensor sizes.
    full_out = full_q.new_zeros(full_q.shape[0], full_out_v.shape[-1])
    full_out[:n_full] = full_out_v

    return from_mode_splits(causal_out, full_out, packed_query_states)


def dispatch_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    assert memory_value is None, "Base dispatch_attention does not handle MemoryValue"
    if isinstance(attention_mask, SplitInfo) and attention_mask.soft_plan_num_tokens > 0:
        if attention_mask.control_stream_token_ranges is not None:
            raise ValueError("Soft-plan bottleneck attention is not defined for multi-control transfer batches.")
        output = soft_plan_bottleneck_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
        )
    elif isinstance(attention_mask, SplitInfo) and attention_mask.control_stream_token_ranges is not None:
        output = multi_control_two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
        )
    elif isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        output = three_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
        )
    elif isinstance(attention_mask, SplitInfo):
        output = two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_meta=attention_mask,
        )
    else:
        raise TypeError(f"Unsupported attention metadata: {type(attention_mask)}")
    return output, None


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: list[tuple[int, int, int]] | None = None,
    natten_parameter_list: list | None = None,
    block_size: int = 128,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    skip_natten_metadata: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    sample_modality_spans: list[SampleModalitySpans] | None = None,
    action_video_attention_mode: str = "legacy_full",
    soft_plan_num_tokens: int = 0,
    mid_soft: bool = False,
    soft_plan_ranges: list[tuple[int, int]] | None = None,
    pad_for_cuda_graphs: bool = False,
) -> tuple[SequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta).
    """
    device = packed_sequence.device
    natten_metadata_list = None
    if action_video_attention_mode in _FASTWAM_ACTION_VIDEO_ATTENTION_MODES:
        if joint_attn_implementation != "two_way":
            raise ValueError(
                f"action_video_attention_mode={action_video_attention_mode!r} requires "
                f"joint_attn_implementation='two_way', got {joint_attn_implementation!r}."
            )
        if sample_modality_spans is None:
            raise ValueError(
                f"action_video_attention_mode={action_video_attention_mode!r} requires sample_modality_spans."
            )

    if joint_attn_implementation == "two_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            sample_modality_spans=sample_modality_spans,
            action_video_attention_mode=action_video_attention_mode,
            soft_plan_num_tokens=soft_plan_num_tokens,
            mid_soft=mid_soft,
            soft_plan_ranges=soft_plan_ranges,
        )
        make_pack = sequence_pack_from_packed_sequence
    elif joint_attn_implementation == "three_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            is_three_way=True,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=action_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
            sample_modality_spans=sample_modality_spans,
            action_video_attention_mode=action_video_attention_mode,
        )
        make_pack = sequence_pack_from_packed_sequence
        # Some memory-driven attention paths implement temporal visibility in
        # their own attention kernels; skip NATTEN metadata for those paths.
        if not skip_natten_metadata:
            # Temporal causal: encode (T, S) supertoken layout; spatial NATTEN: encode (H, W) layout.
            if video_temporal_causal:
                natten_metadata_list = generate_temporal_causal_natten_metadata(
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
                    num_layers=num_layers,
                    head_dim=head_dim,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                )
            else:
                natten_metadata_list = generate_natten_metadata(
                    token_shapes=token_shapes,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                    natten_parameter_list=natten_parameter_list,
                )
    else:
        raise ValueError(
            f"Invalid joint_attn_implementation: {joint_attn_implementation}. Must be 'two_way' or 'three_way'."
        )

    input_pack = make_pack(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    # Not needed anymore, can cause recompilations.
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return input_pack, attention_meta, natten_metadata_list
