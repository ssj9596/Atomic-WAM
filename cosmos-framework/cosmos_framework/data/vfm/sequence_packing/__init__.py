# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""High-level entry points for VFM sequence packing."""

from cosmos_framework.data.vfm.sequence_packing.packers import (
    build_fastwam_action_only_packed_sequence,
    pack_input_sequence,
)
from cosmos_framework.data.vfm.sequence_packing.types import (
    ModalityData,
    ModalityTokenSpan,
    PackedSequence,
    SampleModalitySpans,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)

__all__ = [
    "ModalityData",
    "ModalityTokenSpan",
    "PackedSequence",
    "SampleModalitySpans",
    "SequencePlan",
    "build_fastwam_action_only_packed_sequence",
    "build_sequence_plans_from_data_batch",
    "pack_input_sequence",
]
