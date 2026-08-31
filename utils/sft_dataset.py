"""Dependency-light weighted multimodal dataset for final-logit KD recovery."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from sft_scripts.utils.sft_io import get_conversation_text, resolve_image_path

from .field_weights import token_weights_from_offsets
from .mixture_data import GROUP_TO_ID


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
        user_content: list[dict[str, str]] = []
        image_value = str(sample.get("image") or "").strip()
        if image_value:
            image_path = resolve_image_path(image_value)
            if not image_path.is_file():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            user_content.append({"type": "image", "image": str(image_path)})
        user_content.append({"type": "text", "text": user_text})

        prompt_messages = [
            {
                "role": "user",
                "content": user_content,
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
            sample_id = sample.get("id", index)
            source = sample.get("source", "unknown")
            raise ValueError(
                "Complete SFT sample exceeds max_length; refusing to truncate "
                f"assistant content: id={sample_id!r}, source={source!r}, "
                f"tokens={len(input_ids)}, max_length={self.max_length}. "
                "Rebuild the QA mixture with the same tokenizer and "
                "max_sequence_tokens."
            )

        if (labels != -100).sum().item() == 0:
            labels[-1] = input_ids[-1]

        item: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
            "group_id": torch.tensor(
                GROUP_TO_ID.get(str(sample.get("group") or ""), -1),
                dtype=torch.long,
            ),
        }
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            item["image_grid_thw"] = full_inputs["image_grid_thw"]
        if mm_token_type_ids is not None:
            item["mm_token_type_ids"] = mm_token_type_ids
        return item
