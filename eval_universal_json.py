#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL / MiniCPM-V / InternVL / LLaVA-NeXT / Meow-Omni 单图 JSON 评测脚本。

远程 A100 使用示例：

python /home/yuhang/soulgard-vl/sft_scripts/eval_universal_json.py \
  --model-path /home/yuhang/models/minicpm-v-4_5 \
  --backend auto \
  --batch-size 2 \
  --attn-implementation sdpa

如果不传 --output-dir，结果默认写入：
  <PROJECT_ROOT>/results_a100/result_<model_dir_name>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_scripts.utils.sft_json import extract_json_from_text
from sft_scripts.utils.sft_metrics import (
    BEHAVIOR_FIELD_NAMES,
    behavior_match,
    cat_count_match,
    cat_presence_match,
    cats_visible,
    compare_behavior_fields,
    safe_div,
)
from sft_scripts.utils.sft_prompts import USER_PROMPT


DEFAULT_VAL_JSON = PROJECT_ROOT / "sft_data/sft_cat_val_doubao_lite.json"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_NEW_TOKENS = 2048

REQUIRED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "cats_visible",
    "lighting",
    "other_beings",
    "cats",
    "interactions",
    "environment_anomalies",
    "summary",
]
REQUIRED_CAT_FIELDS = [
    "cat_id",
    "location_on",
    "vertical_position",
    "nearby_anchors",
    "nearby_beings",
    "action",
    "attention_to",
    "posture",
    "abnormalities",
    "is_partially_occluded",
]
REQUIRED_ATTENTION_FIELDS = ["target_type", "target_id"]
REQUIRED_POSTURE_FIELDS = ["overall_body", "ears", "tail", "face", "fur_state"]
REQUIRED_EARS_TAIL_FIELDS = ["visible", "position"]
REQUIRED_FACE_FIELDS = ["visible", "eyelid", "mouth"]


class VisionRunner(Protocol):
    backend_name: str
    resolved_model_path: str
    device: str

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        ...


def windows_path_to_wsl(path: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def is_probably_hf_repo(path: str) -> bool:
    return bool(re.match(r"^[\w.-]+/[\w.\-]+$", path))


def resolve_model_path(model_path: str) -> str:
    resolved = windows_path_to_wsl(model_path)
    if is_probably_hf_repo(resolved):
        return resolved
    return str(Path(resolved).expanduser())


def output_dir_for_model(model_path: str) -> Path:
    resolved = resolve_model_path(model_path).rstrip("/")
    name = resolved.split("/")[-1] if "/" in resolved else resolved
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"
    return PROJECT_ROOT / "results_a100" / f"result_{safe_name}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def load_sft_samples(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = load_jsonl(path)
        source_cache: dict[Path, list[dict[str, Any]]] = {}
        samples = []
        for row in records:
            if "conversations" in row and "image" in row:
                samples.append(row)
                continue
            source_json = row.get("source_json")
            index = row.get("index_in_source")
            if source_json is None or index is None:
                raise ValueError(f"{path} JSONL rows must contain full samples or source_json/index_in_source")
            source_path = Path(str(source_json))
            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path
            if source_path not in source_cache:
                source_cache[source_path] = load_json(source_path)
            samples.append(source_cache[source_path][int(index)])
        return samples

    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"SFT dataset must be a list: {path}")
    return data


def get_assistant_json(sample: dict[str, Any]) -> dict[str, Any] | None:
    for message in sample.get("conversations", []):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, dict):
                return content
            return extract_json_from_text(str(content))
    return None


def get_image_path(sample: dict[str, Any]) -> Path:
    image_value = sample["image"]
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    if str(image_value).startswith(("datasets/", "sft_data/")):
        return PROJECT_ROOT / image_path
    return PROJECT_ROOT / "datasets" / image_path


def infer_backend(model_path: str, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    lower = model_path.lower()
    if "minicpm-v" in lower or "minicpm-o" in lower:
        return "minicpm-v"
    if "meow-omni" in lower or "meow_omni" in lower:
        return "meow-omni"
    if "llava" in lower:
        return "llava-next"
    if "internvl" in lower:
        return "internvl"
    if "minicpm4.1" in lower or "minicpm4-1" in lower or "minicpm4_1" in lower:
        raise ValueError("MiniCPM4.1-8B 是文本模型，不能做图片评测；请使用 MiniCPM-V，例如 minicpm-v-4_5。")
    return "qwen3vl"


def summarize_device_map(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        print(f"模型主设备: {getattr(model, 'device', 'unknown')}")
        return

    placements: dict[str, int] = {}
    for device in device_map.values():
        placements[str(device)] = placements.get(str(device), 0) + 1
    print(f"模型 device_map 汇总: {placements}")
    slow_devices = [device for device in placements if device == "cpu" or device.startswith("disk")]
    if slow_devices:
        print(
            "警告: 模型部分权重被放到 CPU/disk，推理会非常慢；"
            "请换更大显存、降低模型规模，或使用多卡/量化加载。"
            )


def local_path_exists(path: str) -> bool:
    return Path(path).expanduser().exists()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def patch_transformers_tied_weight_keys() -> None:
    """兼容部分 MiniCPM-V remote code 与新版 transformers 的 tied weight 字段差异。"""
    from transformers.modeling_utils import PreTrainedModel

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def normalize_tied_keys(keys: Any) -> dict[str, None]:
        if isinstance(keys, dict):
            return keys
        if keys is None:
            return {}
        return {str(key): None for key in keys}

    def get_all_tied_weights_keys(self) -> dict[str, None]:
        stored = getattr(self, "_soulgard_all_tied_weights_keys", None)
        if stored is not None:
            return normalize_tied_keys(stored)
        return normalize_tied_keys(getattr(self, "_tied_weights_keys", None))

    def set_all_tied_weights_keys(self, value: Any) -> None:
        self._soulgard_all_tied_weights_keys = normalize_tied_keys(value)

    PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)


def ensure_minicpm_tokenizer_attrs(tokenizer: Any) -> None:
    """确保 MiniCPM-V processor 需要的图像特殊 token 属性存在。"""
    token_names = {
        "im_start": "<image>",
        "im_end": "</image>",
        "slice_start": "<slice>",
        "slice_end": "</slice>",
        "im_id_start": "<image_id>",
        "im_id_end": "</image_id>",
    }
    for name, token in token_names.items():
        if not hasattr(tokenizer, name):
            setattr(tokenizer, name, token)

    token_id_attrs = {
        "im_start_id": "im_start",
        "im_end_id": "im_end",
        "slice_start_id": "slice_start",
        "slice_end_id": "slice_end",
        "im_id_start_id": "im_id_start",
        "im_id_end_id": "im_id_end",
    }
    for id_attr, token_attr in token_id_attrs.items():
        if not hasattr(tokenizer, id_attr):
            setattr(tokenizer, id_attr, tokenizer.convert_tokens_to_ids(getattr(tokenizer, token_attr)))

    if not hasattr(tokenizer, "bos_id"):
        setattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
    if not hasattr(tokenizer, "eos_id"):
        setattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None))
    if not hasattr(tokenizer, "unk_id"):
        setattr(tokenizer, "unk_id", getattr(tokenizer, "unk_token_id", None))
    if not hasattr(tokenizer, "newline_id"):
        setattr(tokenizer, "newline_id", tokenizer.convert_tokens_to_ids("\n"))


def missing_fields(data: dict[str, Any], required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if field not in data]


def validate_prediction_fields(pred_json: dict[str, Any] | None) -> list[str]:
    """Return schema-level field errors without changing the prediction."""
    if not isinstance(pred_json, dict):
        return ["prediction_not_dict"]

    errors = []
    top_missing = missing_fields(pred_json, REQUIRED_TOP_LEVEL_FIELDS)
    if top_missing:
        errors.append("missing_top_level:" + ",".join(top_missing))

    try:
        visible_count = int(pred_json.get("cats_visible", 0))
    except (TypeError, ValueError):
        visible_count = 0
        errors.append("cats_visible_not_int")

    for list_field in ("other_beings", "cats", "interactions", "environment_anomalies"):
        if list_field in pred_json and not isinstance(pred_json.get(list_field), list):
            errors.append(f"{list_field}_not_list")

    cats = pred_json.get("cats", [])
    if not isinstance(cats, list):
        cats = []
    if visible_count > 0 and not cats:
        errors.append("cats_empty_when_cats_visible_positive")

    for idx, cat in enumerate(cats):
        prefix = f"cats[{idx}]"
        if not isinstance(cat, dict):
            errors.append(f"{prefix}_not_dict")
            continue

        cat_missing = missing_fields(cat, REQUIRED_CAT_FIELDS)
        if cat_missing:
            errors.append(f"{prefix}.missing:" + ",".join(cat_missing))

        for list_field in ("nearby_anchors", "nearby_beings", "abnormalities"):
            if list_field in cat and not isinstance(cat.get(list_field), list):
                errors.append(f"{prefix}.{list_field}_not_list")

        attention = cat.get("attention_to")
        if "attention_to" in cat:
            if not isinstance(attention, dict):
                errors.append(f"{prefix}.attention_to_not_dict")
            else:
                attention_missing = missing_fields(attention, REQUIRED_ATTENTION_FIELDS)
                if attention_missing:
                    errors.append(f"{prefix}.attention_to.missing:" + ",".join(attention_missing))

        posture = cat.get("posture")
        if "posture" in cat:
            if not isinstance(posture, dict):
                errors.append(f"{prefix}.posture_not_dict")
            else:
                posture_missing = missing_fields(posture, REQUIRED_POSTURE_FIELDS)
                if posture_missing:
                    errors.append(f"{prefix}.posture.missing:" + ",".join(posture_missing))

                for part in ("ears", "tail"):
                    value = posture.get(part)
                    if part in posture:
                        if not isinstance(value, dict):
                            errors.append(f"{prefix}.posture.{part}_not_dict")
                        else:
                            part_missing = missing_fields(value, REQUIRED_EARS_TAIL_FIELDS)
                            if part_missing:
                                errors.append(f"{prefix}.posture.{part}.missing:" + ",".join(part_missing))

                face = posture.get("face")
                if "face" in posture:
                    if not isinstance(face, dict):
                        errors.append(f"{prefix}.posture.face_not_dict")
                    else:
                        face_missing = missing_fields(face, REQUIRED_FACE_FIELDS)
                        if face_missing:
                            errors.append(f"{prefix}.posture.face.missing:" + ",".join(face_missing))

    return errors


class Qwen3VLRunner:
    backend_name = "qwen3vl"

    def __init__(
        self,
        model_path: str,
        prompt: str,
        attn_implementation: str | None,
        local_files_only: bool,
        enable_thinking: bool,
        device_map: str,
        image_max_pixels: int | None,
    ) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        self.resolved_model_path = resolve_model_path(model_path)
        self.prompt = prompt.strip()
        self.enable_thinking = enable_thinking
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载 Qwen3-VL 模型: {self.resolved_model_path}")
        print(f"推理设备: {self.device}")
        print(f"Qwen thinking 模式: {'开启' if self.enable_thinking else '关闭'}")

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if image_max_pixels is not None:
            processor_kwargs["max_pixels"] = image_max_pixels
            print(f"图像最大像素: {image_max_pixels}")
        self.processor = AutoProcessor.from_pretrained(self.resolved_model_path, **processor_kwargs)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.device == "cuda":
            model_kwargs["device_map"] = device_map

        try:
            if AutoModelForImageTextToText is None:
                raise ImportError("AutoModelForImageTextToText is unavailable")
            self.model = AutoModelForImageTextToText.from_pretrained(self.resolved_model_path, **model_kwargs)
        except Exception:
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.resolved_model_path, **model_kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.report_device_map()

    def report_device_map(self) -> None:
        summarize_device_map(self.model)

    def build_inputs(self, image_paths: list[Path]):
        message_batch = []
        for image_path in image_paths:
            message_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
            )
        messages = message_batch[0] if len(message_batch) == 1 else message_batch
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            # ``True`` is not normalized correctly by the Transformers 5.8
            # multimodal chat-template path for variable image-token lengths.
            "processor_kwargs": {"padding": "longest"},
        }
        chat_template = getattr(self.processor, "chat_template", "")
        if isinstance(chat_template, str) and "enable_thinking" in chat_template:
            template_kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            template_kwargs.pop("enable_thinking", None)
            print("当前 processor 不支持 enable_thinking 参数，已按普通 chat template 构造输入。")
            return self.processor.apply_chat_template(messages, **template_kwargs)

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        inputs = self.build_inputs(image_paths).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [(raw_text, extract_json_from_text(raw_text)) for raw_text in raw_texts]


class LlavaNextRunner:
    backend_name = "llava-next"

    def __init__(
        self,
        model_path: str,
        prompt: str,
        attn_implementation: str | None,
        local_files_only: bool,
        device_map: str,
    ) -> None:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        self.resolved_model_path = resolve_model_path(model_path)
        self.prompt = prompt.strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"加载 LLaVA-NeXT 模型: {self.resolved_model_path}")
        print(f"推理设备: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.resolved_model_path,
            local_files_only=local_files_only,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.device == "cuda":
            model_kwargs["device_map"] = device_map

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.resolved_model_path,
            **model_kwargs,
        ).eval()
        if self.device == "cpu":
            self.model.to("cpu")
        summarize_device_map(self.model)

    def build_prompt(self) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        prompts = [self.build_prompt() for _ in images]
        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        raw_texts = self.processor.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [(raw_text, extract_json_from_text(raw_text)) for raw_text in raw_texts]


class MiniCPMVRunner:
    backend_name = "minicpm-v"

    def __init__(
        self,
        model_path: str,
        prompt: str,
        attn_implementation: str | None,
        local_files_only: bool,
        enable_thinking: bool,
        temperature: float | None,
        top_p: float | None,
        stream: bool,
    ) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        patch_transformers_tied_weight_keys()

        self.resolved_model_path = resolve_model_path(model_path)
        self.prompt = prompt.strip()
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"加载 MiniCPM-V 模型: {self.resolved_model_path}")
        print(f"推理设备: {self.device}")
        print("注意: MiniCPM-V 官方 chat demo 是单样本接口，本脚本会逐张图片推理。")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.resolved_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        ensure_minicpm_tokenizer_attrs(self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(
            self.resolved_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.processor.tokenizer = self.tokenizer

        self.model = AutoModel.from_pretrained(self.resolved_model_path, **model_kwargs).eval()
        if not callable(getattr(self.model, "chat", None)):
            raise ValueError(
                "当前模型没有 MiniCPM-V 的 model.chat 接口，不能使用 --backend minicpm-v；"
                "如果这是 Qwen-VL/图文模型请改用 --backend qwen3vl，如果是纯文本 Qwen 模型则不能做图片评测。"
            )
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.processor = self.processor

    def chat_once(self, image_path: Path, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, self.prompt]}]

        kwargs: dict[str, Any] = {
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "processor": self.processor,
            "enable_thinking": self.enable_thinking,
            "max_new_tokens": max_new_tokens,
            "stream": self.stream,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        with torch.no_grad():
            try:
                output = self.model.chat(**kwargs)
            except TypeError:
                # 部分 MiniCPM-V 版本只接受官方 demo 的最小参数集合。
                output = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    enable_thinking=self.enable_thinking,
                    stream=self.stream,
                )

        if self.stream:
            generated_text = ""
            for new_text in output:
                generated_text += new_text
            output = generated_text
        if isinstance(output, tuple) and output:
            output = output[0]
        return output if isinstance(output, str) else str(output)

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        outputs = []
        for image_path in image_paths:
            raw_text = self.chat_once(image_path, max_new_tokens)
            outputs.append((raw_text, extract_json_from_text(raw_text)))
        return outputs


class MeowOmniRunner:
    backend_name = "meow-omni"

    def __init__(
        self,
        model_path: str,
        prompt: str,
        local_files_only: bool,
    ) -> None:
        self.resolved_model_path = resolve_model_path(model_path)
        self.prompt = prompt.strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"加载 Meow-Omni 模型: {self.resolved_model_path}")
        print(f"推理设备: {self.device}")
        print("注意: Meow-Omni 官方 demo 使用 generate 接口，本脚本会逐张图片推理。")
        self.patch_transformers_output_recorder()
        patch_transformers_tied_weight_keys()

        model_dir = Path(self.resolved_model_path)
        model_config = None
        if model_dir.exists():
            import hashlib
            import importlib
            import types
            from transformers import AutoProcessor

            package_name = "_soulgard_meow_omni_" + hashlib.sha1(str(model_dir).encode()).hexdigest()[:12]
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__file__ = str(model_dir / "__init__.py")
                package.__path__ = [str(model_dir)]
                sys.modules[package_name] = package

            model_module = importlib.import_module(f"{package_name}.modeling_meow_omni_1")
            config_module = importlib.import_module(f"{package_name}.configuration_meow_omni_1")
            config_dict = read_json_if_exists(model_dir / "config.json")
            ts_config = config_dict.get("ts_config")
            if isinstance(ts_config, dict) and ts_config.get("decoder_start_token_id") is None:
                ts_config["decoder_start_token_id"] = 0
            model_config = config_module.MeowOmni1Config(**config_dict)

            processor_cls = AutoProcessor
            model_cls = model_module.MeowOmni1ForCausalLM
            self.patch_generation_mixin(model_cls)
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor

            processor_cls = AutoProcessor
            model_cls = AutoModelForCausalLM

        self.processor = processor_cls.from_pretrained(
            self.resolved_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if model_config is not None:
            model_kwargs["config"] = model_config

        self.model = model_cls.from_pretrained(self.resolved_model_path, **model_kwargs).eval()
        if not hasattr(self.model, "generation_config"):
            from transformers import GenerationConfig

            self.model.generation_config = GenerationConfig.from_model_config(self.model.config)
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")
        summarize_device_map(self.model)

    def patch_transformers_output_recorder(self) -> None:
        """兼容 Meow-Omni remote code 与新版 transformers 的工具类差异。"""
        from transformers.utils import generic

        if hasattr(generic, "OutputRecorder"):
            return

        class OutputRecorder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.output = None

            def __enter__(self) -> "OutputRecorder":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        generic.OutputRecorder = OutputRecorder

    def patch_generation_mixin(self, model_cls: Any) -> None:
        """兼容旧 remote code 依赖 PreTrainedModel 自带 generate 的写法。"""
        if hasattr(model_cls, "generate"):
            return

        from transformers.generation import GenerationMixin

        for name, attr in GenerationMixin.__dict__.items():
            if name.startswith("__") or hasattr(model_cls, name):
                continue
            setattr(model_cls, name, attr)
        model_cls._validate_model_kwargs = lambda self, model_kwargs: None

        original_expand = GenerationMixin.__dict__["_expand_inputs_for_generation"]

        def expand_inputs_for_generation(
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: torch.LongTensor | None = None,
            **model_kwargs: Any,
        ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
            if expand_size == 1:
                return input_ids, model_kwargs
            return original_expand.__func__(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs,
            )

        model_cls._expand_inputs_for_generation = staticmethod(expand_inputs_for_generation)

    def build_prompt(self) -> str:
        return (
            "User: <image>./</image>\n"
            "当前输入是一张真实图片，没有音频和时序数据；必须先观察图片内容再填写 JSON。"
            "不要为了让 JSON 简短或安全而默认输出无猫。"
            "只有当画面完全没有猫的身体、头部、耳朵、尾巴、四肢、毛发或局部轮廓时，cats_visible 才能为 0。"
            "只要看到任何猫或疑似猫的局部，cats_visible 必须大于 0，cats 数组必须包含对应猫对象并补齐所有字段。"
            "你必须只输出一个完整、可被 json.loads 解析的 JSON 对象。"
            "禁止输出解释、Markdown、代码块、帧数说明或 JSON 外的任何文字。"
            "即使画面无猫，也必须补齐 schema 要求的所有字段并正确闭合所有括号。\n"
            f"{self.prompt}\n"
            "Assistant:"
        )

    def normalize_inputs(self, inputs: Any) -> Any:
        pixel_values = inputs.get("pixel_values")
        if isinstance(pixel_values, list) and pixel_values and isinstance(pixel_values[0], list):
            tiles = pixel_values[0]
            if tiles:
                max_width = max(tile.shape[-1] for tile in tiles)
                padded_tiles = [
                    F.pad(tile, (0, max_width - tile.shape[-1])) if tile.shape[-1] != max_width else tile
                    for tile in tiles
                ]
                inputs["pixel_values"] = torch.stack(padded_tiles).to(self.device, dtype=self.dtype)

        tgt_sizes = inputs.get("tgt_sizes")
        if isinstance(tgt_sizes, list) and len(tgt_sizes) == 1 and isinstance(tgt_sizes[0], torch.Tensor):
            inputs["tgt_sizes"] = tgt_sizes[0].to(self.device)

        for key in ("image_bound", "audio_bounds", "spk_bounds"):
            value = inputs.get(key)
            if isinstance(value, list):
                inputs[key] = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in value]

        for key in ("audio_features", "audio_feature_lens"):
            value = inputs.get(key)
            if isinstance(value, list) and not value:
                inputs.pop(key, None)

        return inputs

    def infer_one(self, image_path: Path, max_new_tokens: int) -> tuple[str, dict[str, Any] | None]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[self.build_prompt()],
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        inputs = self.normalize_inputs(inputs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, input_length:] if input_length else output_ids
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            raw_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not raw_text.strip():
                raw_text = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        else:
            raw_text = str(output_ids[0].tolist())
        return raw_text, extract_json_from_text(raw_text)

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        return [self.infer_one(image_path, max_new_tokens) for image_path in image_paths]


def internvl_image_to_tensor(image: Image.Image, input_size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.convert("RGB").resize((input_size, input_size), resampling)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def internvl_dynamic_preprocess(
    image: Image.Image,
    input_size: int,
    max_tiles: int,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        },
        key=lambda item: item[0] * item[1],
    )

    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * input_size * input_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = input_size * best_ratio[0]
    target_height = input_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_width, target_height))
    processed_images = []
    for block_idx in range(blocks):
        box = (
            (block_idx % (target_width // input_size)) * input_size,
            (block_idx // (target_width // input_size)) * input_size,
            ((block_idx % (target_width // input_size)) + 1) * input_size,
            ((block_idx // (target_width // input_size)) + 1) * input_size,
        )
        processed_images.append(resized.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((input_size, input_size)))
    return processed_images


class InternVLRunner:
    backend_name = "internvl"

    def __init__(
        self,
        model_path: str,
        prompt: str,
        attn_implementation: str | None,
        local_files_only: bool,
        input_size: int,
        max_tiles: int,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.resolved_model_path = resolve_model_path(model_path)
        self.prompt = prompt.strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.max_tiles = max_tiles
        self.local_files_only = local_files_only or local_path_exists(self.resolved_model_path)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"加载 InternVL 模型: {self.resolved_model_path}")
        print(f"推理设备: {self.device}")
        print(f"InternVL local_files_only: {self.local_files_only}")
        print(f"InternVL 图像切块: input_size={input_size}, max_tiles={max_tiles}")
        print("注意: InternVL 官方接口是 model.chat，本脚本会逐张图片推理。")
        print(
            "InternVL attention: "
            f"use_flash_attn={attn_implementation == 'flash_attention_2'} "
            f"(传入 attn_implementation={attn_implementation or 'None'})"
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_flash_attn": attn_implementation == "flash_attention_2",
        }
        try:
            tokenizer_start = time.perf_counter()
            print("开始加载 InternVL tokenizer...")
            self.preflight_tokenizer_files()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.resolved_model_path,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                use_fast=False,
            )
            print(f"InternVL tokenizer 加载完成: {time.perf_counter() - tokenizer_start:.2f}s")
        except Exception as exc:
            error_text = str(exc)
            if "protobuf" in error_text.lower():
                raise RuntimeError(
                    "InternVL tokenizer 需要 protobuf；请先安装: pip install protobuf"
                ) from exc
            if any(
                pattern in error_text.lower()
                for pattern in ("sentencepiece", "tokenizer.model", "tiktoken", "error parsing line")
            ):
                raise RuntimeError(
                    "InternVL tokenizer 需要 sentencepiece/protobuf；请先安装: "
                    "pip install sentencepiece protobuf"
                ) from exc
            raise

        model_start = time.perf_counter()
        print("开始加载 InternVL 权重...")
        self.model = AutoModel.from_pretrained(self.resolved_model_path, **model_kwargs).eval()
        print(f"InternVL 权重加载完成: {time.perf_counter() - model_start:.2f}s")
        if not callable(getattr(self.model, "chat", None)):
            raise ValueError(
                "当前模型没有 InternVL 的 model.chat 接口。"
                "如果这是 OpenGVLab/InternVL3-*-hf 变体，请改用 --backend qwen3vl；"
                "如果是普通 InternVL3-*，请使用 --backend internvl。"
            )
        if self.device == "cuda":
            cuda_start = time.perf_counter()
            print("开始移动 InternVL 到 CUDA...")
            self.model = self.model.cuda()
            print(f"InternVL CUDA 就绪: {time.perf_counter() - cuda_start:.2f}s")
        summarize_device_map(self.model)

    def preflight_tokenizer_files(self) -> None:
        model_dir = Path(self.resolved_model_path)
        if not model_dir.exists():
            print("InternVL tokenizer 预检: 模型路径不是本地目录，跳过本地文件检查。")
            return

        interesting_files = sorted(
            path.name
            for pattern in ("tokenizer*", "tokenization*", "configuration*", "config.json")
            for path in model_dir.glob(pattern)
            if path.is_file()
        )
        print(f"InternVL tokenizer 文件: {interesting_files}")

        tokenizer_config = read_json_if_exists(model_dir / "tokenizer_config.json")
        if tokenizer_config:
            print(
                "InternVL tokenizer_config: "
                f"tokenizer_class={tokenizer_config.get('tokenizer_class')}, "
                f"auto_map={tokenizer_config.get('auto_map')}"
            )

        tokenizer_model = model_dir / "tokenizer.model"
        if not tokenizer_model.exists():
            print("警告: 没找到 tokenizer.model；如果本地目录不完整，AutoTokenizer 可能失败。")
            return

        try:
            import sentencepiece as spm
            import google.protobuf  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "InternVL tokenizer 需要 sentencepiece/protobuf；请安装: "
                "pip install sentencepiece protobuf"
            ) from exc

        sp_start = time.perf_counter()
        processor = spm.SentencePieceProcessor()
        loaded = processor.Load(str(tokenizer_model))
        if not loaded:
            raise RuntimeError(f"SentencePiece 无法加载 tokenizer.model: {tokenizer_model}")
        print(
            "InternVL tokenizer.model 预检完成: "
            f"vocab={processor.GetPieceSize()}, "
            f"time={time.perf_counter() - sp_start:.2f}s"
        )

    def load_pixel_values(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tiles = internvl_dynamic_preprocess(image, self.input_size, self.max_tiles)
        pixel_values = torch.stack([internvl_image_to_tensor(tile, self.input_size) for tile in tiles])
        if self.device == "cuda":
            return pixel_values.to(torch.bfloat16).cuda()
        return pixel_values

    def chat_once(self, image_path: Path, max_new_tokens: int) -> str:
        pixel_values = self.load_pixel_values(image_path)
        question = "<image>\n" + self.prompt
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        with torch.no_grad():
            output = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return output if isinstance(output, str) else str(output)

    def infer_batch(self, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, dict[str, Any] | None]]:
        outputs = []
        for image_path in image_paths:
            raw_text = self.chat_once(image_path, max_new_tokens)
            outputs.append((raw_text, extract_json_from_text(raw_text)))
        return outputs


def build_runner(args: argparse.Namespace) -> VisionRunner:
    backend = infer_backend(args.model_path, args.backend)
    if backend == "qwen3vl":
        return Qwen3VLRunner(
            args.model_path,
            args.prompt,
            args.attn_implementation,
            args.local_files_only,
            args.enable_thinking,
            args.device_map,
            args.image_max_pixels,
        )
    if backend == "llava-next":
        return LlavaNextRunner(
            args.model_path,
            args.prompt,
            args.attn_implementation,
            args.local_files_only,
            args.device_map,
        )
    if backend == "meow-omni":
        return MeowOmniRunner(
            args.model_path,
            args.prompt,
            args.local_files_only,
        )
    if backend == "minicpm-v":
        return MiniCPMVRunner(
            args.model_path,
            args.prompt,
            args.attn_implementation,
            args.local_files_only,
            args.enable_thinking,
            args.temperature,
            args.top_p,
            args.minicpm_stream,
        )
    if backend == "internvl":
        return InternVLRunner(
            args.model_path,
            args.prompt,
            args.attn_implementation,
            args.local_files_only,
            args.internvl_input_size,
            args.internvl_max_tiles,
        )
    raise ValueError(f"未知推理后端: {backend}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    wall_start = time.perf_counter()
    samples = load_sft_samples(Path(args.val_json))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    output_dir = Path(args.output_dir) if args.output_dir else output_dir_for_model(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    if predictions_path.exists():
        predictions_path.unlink()

    load_start = time.perf_counter()
    runner = build_runner(args)
    load_seconds = time.perf_counter() - load_start

    total = 0
    parse_failed = 0
    image_missing = 0
    cat_presence_correct = 0
    cat_count_correct = 0
    behavior_total_on_detected_cat = 0
    behavior_correct_on_detected_cat = 0
    behavior_total_on_gt_cat = 0
    behavior_correct_on_gt_cat = 0
    behavior_field_correct = {field: 0 for field in BEHAVIOR_FIELD_NAMES}
    behavior_field_total = {field: 0 for field in BEHAVIOR_FIELD_NAMES}
    inference_seconds = 0.0
    generated_items = 0
    inference_failed = 0
    first_inference_error: str | None = None
    batch_fallbacks = 0
    batch_fallback_recovered_items = 0
    first_batch_fallback_error: str | None = None
    schema_field_failed = 0
    first_schema_field_error: str | None = None
    gt_cat_samples = 0
    gt_cat_detected_by_pred = 0

    batch_size = max(1, args.batch_size)
    if runner.backend_name == "minicpm-v" and batch_size != 1:
        print(f"MiniCPM-V 后端会逐张推理；保留 batch-size={batch_size} 只是为了复用评测循环。")
    print(f"评测后端: {runner.backend_name}")
    print(f"评测 batch size: {batch_size}")
    print(f"输出目录: {output_dir}")

    debug_printed = 0

    def maybe_print_debug_record(record: dict[str, Any]) -> None:
        nonlocal debug_printed
        if args.debug_print_samples <= 0 or debug_printed >= args.debug_print_samples:
            return
        raw_prediction = str(record.get("raw_prediction") or "")
        raw_limit = max(0, args.debug_raw_chars)
        raw_preview = raw_prediction[:raw_limit]
        if raw_limit and len(raw_prediction) > raw_limit:
            raw_preview += "\n...<已截断>"

        print(f"\n===== 调试样本 {record.get('index')} =====")
        print(f"图片: {record.get('image_path')}")
        print(f"错误: {record.get('errors')}")
        print("原始输出:")
        print(raw_preview)
        print("解析后 JSON:")
        print(json.dumps(record.get("prediction"), ensure_ascii=False, indent=2))
        debug_printed += 1

    for batch_start in tqdm(range(0, len(samples), batch_size), desc=f"eval_{runner.backend_name}"):
        batch_samples = samples[batch_start : batch_start + batch_size]
        valid_items = []

        for offset, sample in enumerate(batch_samples):
            idx = batch_start + offset
            image_path = get_image_path(sample)
            gt_json = get_assistant_json(sample)
            total += 1

            record: dict[str, Any] = {
                "index": idx,
                "image": sample.get("image"),
                "image_path": str(image_path),
                "ground_truth": gt_json,
                "prediction": None,
                "raw_prediction": "",
                "errors": [],
            }

            if gt_json is None:
                record["errors"].append("ground_truth_json_parse_failed")
                parse_failed += 1
                maybe_print_debug_record(record)
                append_jsonl(predictions_path, record)
                continue

            if not image_path.exists():
                record["errors"].append("image_missing")
                image_missing += 1
                maybe_print_debug_record(record)
                append_jsonl(predictions_path, record)
                continue

            valid_items.append((record, gt_json, image_path))

        if not valid_items:
            continue

        try:
            infer_start = time.perf_counter()
            batch_outputs = runner.infer_batch([image_path for _, _, image_path in valid_items], args.max_new_tokens)
            inference_seconds += time.perf_counter() - infer_start
            generated_items += len(batch_outputs)
        except Exception as exc:
            inference_seconds += time.perf_counter() - infer_start
            if len(valid_items) == 1:
                error_text = f"inference_failed:{exc}"
                inference_failed += 1
                if first_inference_error is None:
                    first_inference_error = error_text
                record = valid_items[0][0]
                record["errors"].append(error_text)
                maybe_print_debug_record(record)
                append_jsonl(predictions_path, record)
                continue

            batch_fallbacks += 1
            batch_error = f"batch_inference_failed:{exc}"
            if first_batch_fallback_error is None:
                first_batch_fallback_error = batch_error
            print(
                f"Batch inference failed for {len(valid_items)} items; "
                "retrying one item at a time."
            )
            recovered_items = []
            recovered_outputs = []
            for item in valid_items:
                record, _, image_path = item
                try:
                    single_start = time.perf_counter()
                    single_output = runner.infer_batch(
                        [image_path],
                        args.max_new_tokens,
                    )
                    inference_seconds += time.perf_counter() - single_start
                    recovered_items.append(item)
                    recovered_outputs.append(single_output[0])
                    generated_items += 1
                    batch_fallback_recovered_items += 1
                    record["warnings"] = [batch_error]
                except Exception as single_exc:
                    inference_seconds += time.perf_counter() - single_start
                    error_text = f"inference_failed:{single_exc}"
                    inference_failed += 1
                    if first_inference_error is None:
                        first_inference_error = error_text
                    record["errors"].append(error_text)
                    record["warnings"] = [batch_error]
                    maybe_print_debug_record(record)
                    append_jsonl(predictions_path, record)
            valid_items = recovered_items
            batch_outputs = recovered_outputs
            if not valid_items:
                continue

        for (record, gt_json, _), (raw_text, pred_json) in zip(valid_items, batch_outputs):
            record["raw_prediction"] = raw_text
            record["prediction"] = pred_json
            if pred_json is None:
                record["errors"].append("prediction_json_parse_failed")
                parse_failed += 1

            schema_field_errors = validate_prediction_fields(pred_json)
            record["schema_field_errors"] = schema_field_errors
            if schema_field_errors:
                schema_field_failed += 1
                if first_schema_field_error is None:
                    first_schema_field_error = f"index={record['index']}:{';'.join(schema_field_errors)}"

            is_cat_presence_correct = cat_presence_match(gt_json, pred_json)
            is_cat_count_correct = cat_count_match(gt_json, pred_json)
            is_behavior_correct = behavior_match(gt_json, pred_json)

            cat_presence_correct += int(is_cat_presence_correct)
            cat_count_correct += int(is_cat_count_correct)

            if cats_visible(gt_json) > 0:
                gt_cat_samples += 1
                gt_cat_detected_by_pred += int(cats_visible(pred_json) > 0)

            if cats_visible(gt_json) > 0:
                behavior_total_on_gt_cat += 1
                behavior_correct_on_gt_cat += int(is_behavior_correct)

            if cats_visible(gt_json) > 0 and cats_visible(pred_json) > 0:
                behavior_total_on_detected_cat += 1
                behavior_correct_on_detected_cat += int(is_behavior_correct)
                gt_cats = gt_json.get("cats", [])
                pred_cats = pred_json.get("cats", []) if pred_json else []
                field_details = []
                for gt_cat, pred_cat in zip(gt_cats, pred_cats):
                    cat_field_result = compare_behavior_fields(gt_cat, pred_cat)
                    field_details.append(cat_field_result)
                    for field, is_correct in cat_field_result.items():
                        behavior_field_total[field] += 1
                        behavior_field_correct[field] += int(is_correct)
                record["behavior_field_correct"] = field_details

            record["cat_presence_correct"] = is_cat_presence_correct
            record["cat_count_correct"] = is_cat_count_correct
            record["behavior_correct"] = is_behavior_correct
            maybe_print_debug_record(record)
            append_jsonl(predictions_path, record)

    behavior_field_metrics = {
        field: {
            "accuracy": safe_div(behavior_field_correct[field], behavior_field_total[field]),
            "correct": behavior_field_correct[field],
            "total": behavior_field_total[field],
        }
        for field in BEHAVIOR_FIELD_NAMES
    }

    wall_seconds = time.perf_counter() - wall_start
    metrics = {
        "total_samples": total,
        "parse_failed": parse_failed,
        "image_missing": image_missing,
        "inference_failed": inference_failed,
        "first_inference_error": first_inference_error,
        "batch_fallbacks": batch_fallbacks,
        "batch_fallback_recovered_items": batch_fallback_recovered_items,
        "first_batch_fallback_error": first_batch_fallback_error,
        "cat_recognition_accuracy": safe_div(cat_presence_correct, total),
        "cat_count_accuracy": safe_div(cat_count_correct, total),
        "behavior_analysis_accuracy": safe_div(behavior_correct_on_detected_cat, behavior_total_on_detected_cat),
        "behavior_accuracy_on_gt_cat_images": safe_div(behavior_correct_on_gt_cat, behavior_total_on_gt_cat),
        "counts": {
            "cat_presence_correct": cat_presence_correct,
            "cat_count_correct": cat_count_correct,
            "behavior_correct_on_detected_cat": behavior_correct_on_detected_cat,
            "behavior_total_on_detected_cat": behavior_total_on_detected_cat,
            "behavior_correct_on_gt_cat": behavior_correct_on_gt_cat,
            "behavior_total_on_gt_cat": behavior_total_on_gt_cat,
            "inference_failed": inference_failed,
        },
        "schema_validation": {
            "failed": schema_field_failed,
            "first_error": first_schema_field_error,
        },
        "smoke_checks": {
            "gt_cat_samples": gt_cat_samples,
            "gt_cat_detected_by_pred": gt_cat_detected_by_pred,
            "strict_smoke_checks": args.strict_smoke_checks,
        },
        "behavior_field_metrics": behavior_field_metrics,
        "speed": {
            "wall_seconds": round(wall_seconds, 4),
            "model_load_seconds": round(load_seconds, 4),
            "inference_seconds": round(inference_seconds, 4),
            "generated_items": generated_items,
            "samples_per_second_including_load": safe_div(total, wall_seconds),
            "inference_items_per_second": safe_div(generated_items, inference_seconds),
            "avg_inference_seconds_per_item": safe_div(inference_seconds, generated_items),
        },
        "paths": {
            "model_path": runner.resolved_model_path,
            "val_json": str(Path(args.val_json)),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "batch_size": batch_size,
            "max_new_tokens": args.max_new_tokens,
            "attn_implementation": args.attn_implementation,
        },
        "backend": {
            "name": runner.backend_name,
            "requested": args.backend,
            "device": runner.device,
            "enable_thinking": args.enable_thinking,
            "local_files_only": args.local_files_only,
        },
    }
    dump_json(metrics_path, metrics)
    return metrics


def enforce_strict_smoke_checks(metrics: dict[str, Any]) -> None:
    schema_validation = metrics.get("schema_validation", {})
    smoke_checks = metrics.get("smoke_checks", {})
    failures = []

    if int(schema_validation.get("failed") or 0) > 0:
        failures.append(f"字段校验失败样本数: {schema_validation.get('failed')}")
        if schema_validation.get("first_error"):
            failures.append(f"首个字段错误: {schema_validation['first_error']}")

    gt_cat_samples = int(smoke_checks.get("gt_cat_samples") or 0)
    gt_cat_detected_by_pred = int(smoke_checks.get("gt_cat_detected_by_pred") or 0)
    if gt_cat_samples > 0 and gt_cat_detected_by_pred == 0:
        failures.append(
            "smoke 数据包含 GT 有猫样本，但模型没有检出任何一只猫；"
            "这通常表示图像输入/模板或 prompt 仍有问题，不能公平进入全量评测。"
        )

    if not failures:
        print("严格 smoke 检查通过: 字段完整，且有猫样本至少检出一只猫。")
        return

    print("\n===== 严格 smoke 检查失败 =====", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用单图 JSON 评测：支持 Qwen3-VL、MiniCPM-V、InternVL、LLaVA-NeXT 和 Meow-Omni。"
    )
    parser.add_argument("--model-path", required=True, help="模型路径或 HF repo id")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "qwen3vl", "minicpm-v", "internvl", "llava-next", "meow-omni"],
        help="推理后端",
    )
    parser.add_argument("--val-json", default=str(DEFAULT_VAL_JSON), help="评估数据路径")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认 results_a100/result_<model_name>")
    parser.add_argument("--max-samples", type=int, default=None, help="仅评估前 N 条")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Qwen3-VL/LLaVA-NeXT 可批量；MiniCPM-V/InternVL/Meow-Omni 实际逐张推理",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=USER_PROMPT.strip(), help="评测 prompt")
    parser.add_argument("--enable-thinking", action="store_true", help="启用模型思考模式；JSON 评测默认关闭")
    parser.add_argument("--local-files-only", action="store_true", help="只从本地模型目录或本地 HF cache 加载")
    parser.add_argument("--temperature", type=float, default=None, help="传给 MiniCPM-V chat；默认不传")
    parser.add_argument("--top-p", type=float, default=None, help="传给 MiniCPM-V chat；默认不传")
    parser.add_argument("--minicpm-stream", action="store_true", help="MiniCPM-V 使用官方 stream=True 路径并收集输出")
    parser.add_argument("--debug-print-samples", type=int, default=0, help="打印前 N 条样本的错误、原始输出和解析 JSON")
    parser.add_argument("--debug-raw-chars", type=int, default=2000, help="调试打印时原始输出最多显示多少字符")
    parser.add_argument(
        "--strict-smoke-checks",
        action="store_true",
        help="烟测用: 要求输出字段完整；若样本含 GT 有猫图，还要求至少检出一只猫，否则返回非 0",
    )
    parser.add_argument("--device-map", default="auto", help="Qwen3-VL/LLaVA-NeXT CUDA 加载 device_map；默认 auto")
    parser.add_argument("--image-max-pixels", type=int, default=None, help="Qwen3-VL processor 最大图像像素；30B 烟测可先设 262144")
    parser.add_argument("--internvl-input-size", type=int, default=448, help="InternVL 图像 tile 尺寸")
    parser.add_argument("--internvl-max-tiles", type=int, default=12, help="InternVL 单图最大 tile 数；烟测可设 4")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Qwen3-VL/LLaVA-NeXT 可用 eager/sdpa/flash_attention_2；MiniCPM-V 建议 sdpa 或 flash_attention_2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    eval_metrics = evaluate(parsed_args)
    print("\n===== 通用 JSON 评测完成 =====")
    print(f"评测后端: {eval_metrics['backend']['name']}")
    print(f"猫识别准确率: {eval_metrics['cat_recognition_accuracy']:.4f}")
    print(f"猫数量准确率: {eval_metrics['cat_count_accuracy']:.4f}")
    print(f"行为分析准确率: {eval_metrics['behavior_analysis_accuracy']:.4f}")
    if eval_metrics["speed"]["generated_items"] == 0 and eval_metrics.get("inference_failed", 0) > 0:
        print("警告: 所有有效样本推理都失败了，本次准确率全 0 不是有效评测结果。")
        print(f"首个推理错误: {eval_metrics.get('first_inference_error')}")
    speed = eval_metrics["speed"]
    print(
        "速度: "
        f"wall={speed['wall_seconds']:.2f}s, "
        f"load={speed['model_load_seconds']:.2f}s, "
        f"infer={speed['inference_seconds']:.2f}s, "
        f"infer_items/s={speed['inference_items_per_second']:.4f}"
    )
    print("行为字段准确率:")
    for field, item in eval_metrics["behavior_field_metrics"].items():
        print(f"  {field}: {item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    print(f"指标文件: {eval_metrics['paths']['metrics']}")
    print(f"逐样本预测: {eval_metrics['paths']['predictions']}")
    if parsed_args.strict_smoke_checks:
        enforce_strict_smoke_checks(eval_metrics)
