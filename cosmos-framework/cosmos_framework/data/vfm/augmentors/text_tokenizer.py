# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Augmentor for tokenizing input text

import json
import random
from typing import Optional

import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_MAX_NUM_TOKENS = 4096


def _find_subsequence(tokens: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(tokens):
        return None
    for start in range(len(tokens) - len(needle) + 1):
        if tokens[start : start + len(needle)] == needle:
            return start
    return None


class TextTokenizerTransform(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args["cfg_dropout_rate"]
        self.use_system_prompt = self.args.get("use_system_prompt", False)

        self._processor = lazy_instantiate(tokenizer_config)

    def __call__(self, data_dict: dict) -> dict:
        input_caption = data_dict[self.input_keys[0]]

        if isinstance(input_caption, dict):
            # Encode dict into a json string. This json string is then passed to the transformer tokenizer.
            input_caption = json.dumps(input_caption)
            data_dict[self.input_keys[0]] = input_caption

        if self.cfg_dropout_rate > 0:
            # If CFG is used, randomly dropout the input caption
            # We dropout the input caption by replacing it with an empty string
            if random.random() < self.cfg_dropout_rate:
                input_caption = ""
                data_dict[self.input_keys[0]] = input_caption

        marker = "\n<|mid_soft_plan_insert|>\n"
        instruction = data_dict.get("_mid_soft_instruction")
        insert_marker = isinstance(instruction, str) and bool(instruction) and input_caption.startswith(instruction)
        if insert_marker:
            input_caption = instruction + marker + input_caption[len(instruction) :]
        text_ids = self._processor.tokenize_text(input_caption, is_video=False, use_system_prompt=self.use_system_prompt)
        if insert_marker:
            tokenizer = getattr(self._processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("mid_soft text tokenization requires a processor exposing its underlying tokenizer.")
            marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
            marker_start = _find_subsequence(text_ids, marker_ids)
            if marker_start is None:
                raise ValueError("mid_soft marker was not found in text-only prompt tokenization.")
            text_ids = text_ids[:marker_start] + text_ids[marker_start + len(marker_ids) :]
            data_dict["soft_plan_insert_index"] = int(marker_start)
        text_ids = text_ids[:_MAX_NUM_TOKENS]  # truncate the text ids to the maximum number of tokens
        # This will take care of wierd edge cases where we generate extremely long captions
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."

_SYSTEM_PROMPT_TRANSFER = "You are a helpful assistant that generates images or videos following the user's instructions and control signals (edge maps, blur, depth, or segmentation)."

_SYSTEM_PROMPTS = {
    "editing": _SYSTEM_PROMPT_IMAGE_EDITING,
    "transfer": _SYSTEM_PROMPT_TRANSFER,
}


class TextTokenizerTransformForEditing(Augmentor):
    """Tokenizer augmentor for interleaved tasks: image editing or transfer (control-conditioned generation).

    Uses a task-specific system prompt. Pass args["task"] = "editing" (default) or "transfer".
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args.get("cfg_dropout_rate", 0.0)
        task = self.args.get("task", "editing")
        self._system_prompt = _SYSTEM_PROMPTS.get(task, _SYSTEM_PROMPTS["editing"])

        self._processor = lazy_instantiate(tokenizer_config)

    def __call__(self, data_dict: dict) -> dict | None:
        input_caption = data_dict.get(self.input_keys[0], "")
        if self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            input_caption = ""
            data_dict[self.input_keys[0]] = input_caption
        text_ids = self._processor.tokenize_text(input_caption, system_prompt=self._system_prompt)
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


class TextTokenizerTransformForTransfer(TextTokenizerTransformForEditing):
    """Tokenizer augmentor for transfer (control-conditioned) generation. Uses transfer system prompt."""

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        args = dict(args) if args else {}
        args["task"] = "transfer"
        super().__init__(input_keys, output_keys, args)
