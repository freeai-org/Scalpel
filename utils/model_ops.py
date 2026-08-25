"""Qwen3-VL 模型结构适配。

这里集中处理模型内部路径、视觉特征缓存和物理删层，算法模块无需知道
``model.model.language_model.layers`` 这类版本相关细节。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass(slots=True)
class LanguageInputs:
    """视觉塔只计算一次后，可重复送入语言模型的输入。"""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: list[torch.Tensor] | None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "visual_pos_masks": self.visual_pos_masks,
            "deepstack_visual_embeds": self.deepstack_visual_embeds,
            "use_cache": False,
        }


class BypassDecoderLayer(nn.Module):
    """保持 decoder 调用协议不变，但把该层变成恒等映射。"""

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return hidden_states


def get_layers(model: Qwen3VLForConditionalGeneration) -> nn.ModuleList:
    return model.model.language_model.layers


def load_model(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attention: str | None = None,
) -> Qwen3VLForConditionalGeneration:
    """从本地 HuggingFace 目录加载模型，不访问网络。"""

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if attention:
        kwargs["attn_implementation"] = attention
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, **kwargs)
    model.to(device)
    model.config.use_cache = False
    return model


def load_processor(path: Path) -> Any:
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    # Decoder-only batched generation must continue from the rightmost real
    # token. Right padding would make it continue from a pad position instead.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return processor


def enable_generation_cache(model: Any) -> None:
    """Persist the cache settings expected by autoregressive deployment."""

    model.config.use_cache = True
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = True


@torch.no_grad()
def prepare_language_inputs(
    model: Qwen3VLForConditionalGeneration,
    model_inputs: dict[str, torch.Tensor],
) -> LanguageInputs:
    """运行一次 embedding 与视觉塔，缓存语言层所需张量。

    一条验证样本需要比较几十个候选层。缓存后，每个候选只重跑语言 decoder，
    不会把相同拼接图反复送入视觉编码器。
    """

    core = model.model
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    mm_token_type_ids = model_inputs.get("mm_token_type_ids")
    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    if model_inputs.get("pixel_values_videos") is not None:
        raise NotImplementedError("Highway 当前数据流只支持图片，多视频输入尚未接入")

    inputs_embeds = core.get_input_embeddings()(input_ids)
    image_mask = None
    deepstack_visual_embeds = None
    pixel_values = model_inputs.get("pixel_values")
    if pixel_values is not None:
        image_outputs = core.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        image_mask, _ = core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask = image_mask[..., 0]
        deepstack_visual_embeds = list(image_outputs.deepstack_features)

    position_ids = core.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    return LanguageInputs(
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )


def final_logits(
    model: Qwen3VLForConditionalGeneration,
    inputs: LanguageInputs,
    positions: torch.Tensor,
) -> torch.Tensor:
    """完整语言 forward，只投影需要评估的位置。"""

    output = model.model.language_model(**inputs.as_kwargs())
    selected_hidden = output.last_hidden_state[:, positions, :]
    return model.lm_head(selected_hidden)


@contextmanager
def bypass_layer(
    model: Qwen3VLForConditionalGeneration,
    layer_index: int,
) -> Iterator[None]:
    """临时把一层替换成恒等映射，退出上下文后原样恢复。"""

    layers = get_layers(model)
    original = layers[layer_index]
    layers[layer_index] = BypassDecoderLayer()
    try:
        yield
    finally:
        layers[layer_index] = original


@contextmanager
def temporary_delete_layer(
    model: Qwen3VLForConditionalGeneration,
    layer_index: int,
) -> Iterator[None]:
    """Temporarily remove a block with contiguous cache layer indices.

    Generation uses KV cache, so a plain identity block would leave a missing
    cache entry at the bypassed layer.  Temporary physical deletion keeps the
    cache indices contiguous and restores the exact original modules/config on
    exit.
    """

    layers = get_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"layer_index={layer_index} out of range")
    original_layers = layers
    original_count = len(layers)
    original_attention_indices = [
        getattr(getattr(layer, "self_attn", None), "layer_idx", None)
        for layer in layers
    ]
    model.model.language_model.layers = nn.ModuleList(
        [layer for index, layer in enumerate(layers) if index != layer_index]
    )
    for current_index, layer in enumerate(get_layers(model)):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = current_index
    new_count = original_count - 1
    model.config.text_config.num_hidden_layers = new_count
    model.model.language_model.config.num_hidden_layers = new_count
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = new_count
    try:
        yield
    finally:
        model.model.language_model.layers = original_layers
        for layer, original_index in zip(original_layers, original_attention_indices):
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and hasattr(attention, "layer_idx")
                and original_index is not None
            ):
                attention.layer_idx = original_index
        model.config.text_config.num_hidden_layers = original_count
        model.model.language_model.config.num_hidden_layers = original_count
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = original_count


def physical_delete_layer(
    model: Qwen3VLForConditionalGeneration,
    layer_index: int,
    original_layer_ids: list[int],
) -> tuple[int, list[int]]:
    """物理删除 decoder block，并同步配置与 attention 的当前层号。"""

    layers = get_layers(model)
    if len(original_layer_ids) != len(layers):
        raise ValueError("original_layer_ids 与当前模型层数不一致")
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"不能删除当前第 {layer_index} 层")

    deleted_original_id = original_layer_ids[layer_index]
    model.model.language_model.layers = nn.ModuleList(
        [layer for index, layer in enumerate(layers) if index != layer_index]
    )
    new_layers = get_layers(model)
    for current_index, layer in enumerate(new_layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = current_index

    new_count = len(new_layers)
    model.config.text_config.num_hidden_layers = new_count
    model.model.language_model.config.num_hidden_layers = new_count
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = new_count
    new_ids = original_layer_ids[:layer_index] + original_layer_ids[layer_index + 1 :]
    return deleted_original_id, new_ids


def save_pruned_model(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    output_dir: Path,
    state: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enable_generation_cache(model)
    # 机器只有约 16GB 主存时，单个 4GB safetensors 分片的序列化临时缓冲
    # 可能申请失败。1GB 分片既能直接被 HF 重载，也显著降低保存峰值。
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="1GB",
    )
    processor.save_pretrained(output_dir)
    from .io_utils import atomic_model_state

    atomic_model_state(output_dir / "highway_state.json", state)
