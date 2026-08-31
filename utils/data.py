"""SFT 数据读取与 Qwen3-VL 多模态输入构造。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .field_weights import token_weights_from_offsets
from .mixture_data import load_training_samples
from sft_scripts.utils.sft_io import (
    get_conversation_text,
    resolve_image_path,
)


@dataclass(slots=True)
class PreparedSample:
    """单条带 batch 维度的模型输入。"""

    source_index: int
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    loss_weights: torch.Tensor


class HighwayDataset:
    """惰性处理 26K 数据，避免启动时一次性解码所有拼接图。"""

    def __init__(self, path: Path, processor: Any, max_length: int = 2048) -> None:
        self.path = path
        self.samples, _ = load_training_samples(path)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedSample:
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
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        sequence_length = min(int(full["input_ids"].shape[-1]), self.max_length)
        labels = full["input_ids"][:, :sequence_length].clone()
        prompt_length = min(int(prompt["input_ids"].shape[-1]), sequence_length)
        labels[:, :prompt_length] = -100
        if not bool((labels != -100).any()):
            labels[:, -1] = full["input_ids"][:, sequence_length - 1]

        assistant_encoding = self.processor.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        assistant_weights = token_weights_from_offsets(
            assistant_text,
            assistant_encoding["offset_mapping"],
        )
        loss_weights = torch.ones_like(labels, dtype=torch.float32)
        weight_count = min(
            int(assistant_weights.numel()),
            max(0, sequence_length - prompt_length),
        )
        if weight_count:
            loss_weights[
                :,
                prompt_length : prompt_length + weight_count,
            ] = assistant_weights[:weight_count]

        text_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in full.items():
            if not isinstance(value, torch.Tensor):
                continue
            model_inputs[key] = value[:, :sequence_length] if key in text_keys else value
        return PreparedSample(index, model_inputs, labels, loss_weights)

    def valid_indices(self) -> list[int]:
        """过滤路径损坏样本；真正的图像解码仍保持惰性。"""

        return [
            index
            for index, sample in enumerate(self.samples)
            if not str(sample.get("image") or "").strip()
            or resolve_image_path(str(sample["image"])).is_file()
        ]


def repeated_random_subsets(
    population: list[int],
    repeats: int,
    sample_size: int,
    seed: int,
) -> list[list[int]]:
    """每次独立随机抽样；固定 seed 后实验表可以精确复现。"""

    if sample_size > len(population):
        raise ValueError(f"验证集只有 {len(population)} 条有效样本，不能抽取 {sample_size} 条")
    generator = random.Random(seed)
    return [generator.sample(population, sample_size) for _ in range(repeats)]


def shuffled_indices(
    length: int,
    epochs: int,
    seed: int,
    limit: int | None = None,
) -> Iterator[tuple[int, int]]:
    """产生 ``(epoch, sample_index)``，每个 epoch 都使用不同但可复现的顺序。"""

    selected_length = min(length, limit) if limit is not None else length
    base = list(range(length))
    for epoch in range(epochs):
        generator = random.Random(seed + epoch)
        generator.shuffle(base)
        for index in base[:selected_length]:
            yield epoch, index


def move_sample(sample: PreparedSample, device: torch.device) -> PreparedSample:
    return PreparedSample(
        source_index=sample.source_index,
        model_inputs={key: value.to(device) for key, value in sample.model_inputs.items()},
        labels=sample.labels.to(device),
        loss_weights=sample.loss_weights.to(device),
    )


def supervised_positions(labels: torch.Tensor, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回预测 assistant token 的隐藏态位置及对应目标 token。

    位置 ``t`` 的 logits 预测标签 ``t+1``，因此这里显式完成 causal shift。
    长回答只均匀抽取最多 ``max_tokens`` 个位置，控制 15 万词表投影的显存。
    """

    shifted = labels[:, 1:]
    positions = torch.nonzero(shifted[0] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("样本截断后没有可监督的 assistant token")
    if positions.numel() > max_tokens:
        offsets = torch.linspace(
            0,
            positions.numel() - 1,
            steps=max_tokens,
            device=positions.device,
        ).round().long()
        positions = positions[offsets]
    targets = shifted[:, positions]
    return positions, targets
