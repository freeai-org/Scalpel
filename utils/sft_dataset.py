"""Dependency-light weighted multimodal dataset for final-logit KD recovery."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from sft_scripts.utils.sft_io import get_conversation_text, resolve_image_path

from .field_weights import token_weights_from_offsets


class WeightedSFTDataset(Dataset):
    """Prepare one image/text SFT sample and its assistant-token weights."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        processor: Any,
        max_length: int,
        loss_mode: str = "weighted",
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.loss_mode = loss_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        user_text, assistant_text = get_conversation_text(sample)
        image_path = resolve_image_path(str(sample.get("image", "")))
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        ]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        mm_token_type_ids = full_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[0]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)

        if self.loss_mode == "weighted":
            assistant_encoding = self.processor.tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            assistant_token_weights = token_weights_from_offsets(
                assistant_text,
                assistant_encoding["offset_mapping"],
            )
            weight_len = min(
                len(assistant_token_weights),
                max(0, len(input_ids) - prompt_len),
            )
            if weight_len > 0:
                loss_weights[prompt_len : prompt_len + weight_len] = (
                    assistant_token_weights[:weight_len]
                )

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
            loss_weights = loss_weights[: self.max_length]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[: self.max_length]

        if (labels != -100).sum().item() == 0:
            labels[-1] = input_ids[-1]

        item: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
        }
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            item["image_grid_thw"] = full_inputs["image_grid_thw"]
        if mm_token_type_ids is not None:
            item["mm_token_type_ids"] = mm_token_type_ids
        return item
