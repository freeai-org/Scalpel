"""Model-loading helpers for final-logit weighted KD recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText


@dataclass(slots=True)
class TrainableStats:
    trainable_parameters: int
    total_parameters: int

    @property
    def ratio(self) -> float:
        return self.trainable_parameters / max(self.total_parameters, 1)


def model_kwargs(attention: str) -> dict[str, Any]:
    return {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
    }


def freeze_teacher(path: Path, attention: str, device: torch.device) -> Any:
    """加载固定 reference teacher，并确保它不参与梯度更新。"""

    teacher = AutoModelForImageTextToText.from_pretrained(
        path,
        **model_kwargs(attention),
    )
    teacher.to(device)
    teacher.eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def attach_lora_to_student(
    path: Path,
    attention: str,
    device: torch.device,
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[Any, TrainableStats]:
    """加载删层后的 student，并在所有 linear 模块上挂 LoRA。"""

    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LoRA recovery requires peft. Install it in the training "
            "environment before running train_weighted_kd.py."
        ) from exc

    student = AutoModelForImageTextToText.from_pretrained(
        path,
        **model_kwargs(attention),
    )
    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    student = get_peft_model(
        student,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()
    student.to(device)
    stats = TrainableStats(
        trainable_parameters=sum(
            parameter.numel()
            for parameter in student.parameters()
            if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for parameter in student.parameters()),
    )
    student.print_trainable_parameters()
    return student, stats
