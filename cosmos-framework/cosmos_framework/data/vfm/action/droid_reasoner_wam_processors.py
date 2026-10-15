# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small processors used only by the DROID Reasoner/WAM joint experiment."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.vfm.dataflow.base import RawItemProcessor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX


class DROIDReasonerSupervisionProcessor(RawItemProcessor):
    """Wrap a VLM processor and retain only valid assistant supervision.

    The DROID subgoal dataset deliberately keeps windows whose annotation is
    missing or has a coverage gap so its length remains identical to the base
    DROID dataset.  Such records still need to pass through the loader, but
    must not create an EOS-only synthetic target.  This wrapper therefore
    masks their complete label row with ``ignore_index``.
    """

    def __init__(
        self,
        processor: RawItemProcessor,
        availability_key: str = "reasoner_ce_available",
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self._processor = processor
        self._availability_key = availability_key
        self._ignore_index = int(ignore_index)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Reasoner availability must be scalar, got shape {tuple(value.shape)}")
            return bool(value.item())
        return bool(value)

    @staticmethod
    def _task_text(conversations: Any) -> str:
        """Extract the human task without retaining the full conversation."""

        if not isinstance(conversations, (list, tuple)):
            return ""
        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("from") != "human":
                continue
            value = turn.get("value", "")
            if not isinstance(value, str):
                continue
            text = value.replace("<image>", "").strip()
            # The DROID formatter places the actual instruction between these
            # markers.  Keep only that small field for optional W&B logging.
            marker = "Task:"
            end_marker = "\nNext 2-second executable action plan:"
            if marker in text:
                text = text.split(marker, 1)[1]
                if end_marker in text:
                    text = text.split(end_marker, 1)[0]
            return " ".join(text.split()).strip()
        return ""

    @staticmethod
    def _user_prompt(conversations: Any) -> str:
        """Return the exact human turn, including all image placeholders."""

        if not isinstance(conversations, (list, tuple)):
            return ""
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human":
                value = turn.get("value", "")
                return value if isinstance(value, str) else ""
        return ""

    @staticmethod
    def _input_images(item: dict[str, Any]) -> list[Any]:
        image = item.get("image")
        if image is not None:
            return [image]
        images = item.get("images")
        if isinstance(images, (list, tuple)):
            return list(images)
        return [] if images is None else [images]

    @classmethod
    def _preview_image(cls, item: dict[str, Any], per_image_max_size: int = 256) -> Any:
        """Return a bounded 1/3-image contact sheet without retaining model tensors."""

        images = cls._input_images(item)
        if not images:
            return None

        try:
            from PIL import Image

            previews = []
            for image in images:
                if isinstance(image, Image.Image):
                    preview = image.convert("RGB").copy()
                elif isinstance(image, torch.Tensor):
                    tensor = image.detach().cpu()
                    if tensor.ndim != 3:
                        continue
                    if tensor.shape[0] in (1, 3, 4):
                        tensor = tensor[:3].permute(1, 2, 0)
                    if torch.is_floating_point(tensor):
                        min_value = float(tensor.min().item())
                        max_value = float(tensor.max().item())
                        if min_value >= -1.01 and max_value <= 1.01:
                            tensor = (tensor + 1.0) * 127.5
                        tensor = tensor.clamp(0.0, 255.0)
                    preview = Image.fromarray(tensor.to(torch.uint8).numpy()).convert("RGB")
                else:
                    continue
                preview.thumbnail(
                    (int(per_image_max_size), int(per_image_max_size)),
                    Image.Resampling.BILINEAR,
                )
                previews.append(preview)
            if not previews:
                return None
            if len(previews) == 1:
                return previews[0]
            contact_sheet = Image.new(
                "RGB",
                (sum(preview.width for preview in previews), max(preview.height for preview in previews)),
            )
            x = 0
            for preview in previews:
                contact_sheet.paste(preview, (x, 0))
                x += preview.width
            return contact_sheet
        except Exception:
            # Optional logging metadata must never make a training sample fail.
            return None

    def process(self, item: dict) -> dict:
        result = self._processor.process(item)
        available = self._as_bool(item.get(self._availability_key, False))
        if not available:
            result["labels"] = torch.full_like(result["labels"], self._ignore_index)
            result["token_mask"] = torch.zeros_like(result["token_mask"], dtype=torch.bool)
        # Preserve only bounded text/preview metadata for optional
        # phase-specific W&B logging.  Derive it from the existing conversation
        # instead of bloating the raw dataset schema with duplicate training
        # fields or retaining model-side pixel tensors.
        conversations = item.get("conversations", [])
        assistant_targets = [
            turn.get("value", "") for turn in conversations if isinstance(turn, dict) and turn.get("from") == "gpt"
        ]
        if assistant_targets:
            result["reasoner_gold_target_text"] = str(assistant_targets[-1])
        result["reasoner_task_text"] = self._task_text(conversations)
        result["reasoner_user_prompt"] = self._user_prompt(conversations)
        result["reasoner_preview_image"] = self._preview_image(item)
        result["reasoner_image_count"] = len(self._input_images(item))
        result["task_type"] = "reasoner"
        return result


class DROIDWAMStreamProcessor(RawItemProcessor):
    """Stamp the WAM task type on an already-minimal WAM record."""

    def process(self, item: dict) -> dict:
        result = dict(item)
        result["task_type"] = "wam"
        return result


__all__ = ["DROIDReasonerSupervisionProcessor", "DROIDWAMStreamProcessor"]
