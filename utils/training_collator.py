"""Dependency-light dynamic batching for multimodal weighted KD."""

from __future__ import annotations

import torch


def pad_1d_tensor(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: int | float,
) -> torch.Tensor:
    """Right-pad one token-aligned tensor to a shared batch length."""

    if tensor.numel() >= target_length:
        return tensor[:target_length]
    padding = torch.full(
        (target_length - tensor.numel(),),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=0)


class DynamicKDCollator:
    """Pad token tensors and concatenate Qwen3-VL image tensors."""

    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(
        self,
        features: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        target_length = min(
            max(int(feature["input_ids"].numel()) for feature in features),
            self.max_length,
        )
        batch = {
            "input_ids": torch.stack(
                [
                    pad_1d_tensor(
                        feature["input_ids"],
                        target_length,
                        self.pad_token_id,
                    )
                    for feature in features
                ]
            ),
            "attention_mask": torch.stack(
                [
                    pad_1d_tensor(
                        feature["attention_mask"],
                        target_length,
                        0,
                    )
                    for feature in features
                ]
            ),
            "labels": torch.stack(
                [
                    pad_1d_tensor(feature["labels"], target_length, -100)
                    for feature in features
                ]
            ),
            "loss_weights": torch.stack(
                [
                    pad_1d_tensor(
                        feature["loss_weights"],
                        target_length,
                        1.0,
                    )
                    for feature in features
                ]
            ),
        }
        if "mm_token_type_ids" in features[0]:
            batch["mm_token_type_ids"] = torch.stack(
                [
                    pad_1d_tensor(
                        feature["mm_token_type_ids"],
                        target_length,
                        0,
                    )
                    for feature in features
                ]
            )
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [feature["pixel_values"] for feature in features],
                dim=0,
            )
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat(
                [feature["image_grid_thw"] for feature in features],
                dim=0,
            )
        return batch
