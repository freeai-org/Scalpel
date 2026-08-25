"""Field-weighted response distillation for a physically pruned Qwen3-VL."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from sft_scripts.stage1_stage2_train_weighted_json_sft import WeightedSFTDataset
except ModuleNotFoundError:
    from sft_scripts.stage2_train_weighted_json_sft import WeightedSFTDataset
from sft_scripts.utils.qwen3vl_train_utils import LossJsonlCallback
from sft_scripts.utils.sft_io import load_training_samples

from .utils.field_weights import weight_config
from .utils.io_utils import write_json
from .utils.metrics import field_weighted_kd_loss
from .utils.model_ops import enable_generation_cache
from .train_summarize import read_loss_rows, summarize_rows
from .utils.training_collator import DynamicKDCollator


class WeightedKDTrainer(Trainer):
    """Trainer with field-weighted hard CE plus teacher KL at the same tokens."""

    def __init__(self, *args: Any, teacher_model: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self._component_sums = {"hard_weighted_ce": 0.0, "soft_weighted_kl": 0.0}
        self._component_count = 0

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        loss_weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        valid_mask = shift_labels != -100
        if not bool(valid_mask.any()):
            raise ValueError("Training batch contains no supervised assistant tokens")

        first_valid = int(
            torch.nonzero(valid_mask, as_tuple=False)[:, 1].min().item()
        )
        keep_positions = torch.arange(
            first_valid,
            shift_labels.shape[-1],
            device=shift_labels.device,
            dtype=torch.long,
        )
        # ``no_grad`` tensors may safely participate as constants in the student
        # backward graph.  ``inference_mode`` tensors cannot be saved by autograd.
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs,
                logits_to_keep=keep_positions,
            )
        student_outputs = model(
            **inputs,
            logits_to_keep=keep_positions,
        )
        teacher_logits = teacher_outputs.logits.float()
        student_logits = student_outputs.logits.float()
        target_labels = shift_labels[:, keep_positions].to(student_logits.device)
        target_weights = shift_weights[:, keep_positions].to(
            student_logits.device,
            torch.float32,
        )
        losses = field_weighted_kd_loss(
            student_logits,
            teacher_logits,
            target_labels,
            target_weights,
        )
        self._component_sums["hard_weighted_ce"] += float(
            losses.hard_ce.detach().item()
        )
        self._component_sums["soft_weighted_kl"] += float(
            losses.soft_kl.detach().item()
        )
        self._component_count += 1
        return (
            (losses.total, student_outputs)
            if return_outputs
            else losses.total
        )

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._component_count:
            logs = {
                **logs,
                **{
                    name: value / self._component_count
                    for name, value in self._component_sums.items()
                },
            }
            self._component_sums = {
                "hard_weighted_ce": 0.0,
                "soft_weighted_kl": 0.0,
            }
            self._component_count = 0
        super().log(logs, *args, **kwargs)


def model_kwargs(attention: str) -> dict[str, Any]:
    return {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
    }


def load_teacher(path: Path, attention: str, device: torch.device) -> Any:
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


def load_student_with_lora(
    path: Path,
    attention: str,
    device: torch.device,
    rank: int,
    alpha: int,
    dropout: float,
) -> Any:
    student = AutoModelForImageTextToText.from_pretrained(
        path,
        **model_kwargs(attention),
    )
    student.config.use_cache = False
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
    student.enable_input_require_grads()
    student.to(device)
    student.print_trainable_parameters()
    return student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("--batch-size and --grad-accum must be positive")
    samples = load_training_samples(
        args.train_data,
        args.max_samples or None,
    )
    processor = AutoProcessor.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"
    processor.image_processor.max_pixels = args.image_max_pixels
    processor.image_processor.min_pixels = 3136
    dataset = WeightedSFTDataset(
        samples,
        processor,
        args.max_length,
        "weighted",
    )
    probe = dataset[0]
    if int((probe["labels"] != -100).sum().item()) < 1:
        raise ValueError("First training sample has no supervised tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = args.output_dir / "loss.jsonl"
    resolved = {
        "teacher_model": str(args.teacher_model.resolve()),
        "student_model": str(args.student_model.resolve()),
        "train_data": str(args.train_data.resolve()),
        "train_samples": len(samples),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "image_max_pixels": args.image_max_pixels,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "loss": "field_weighted_ce + field_weighted_KL(teacher||student)",
        "temperature": 1.0,
        "field_weights": weight_config(),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
    }
    write_json(args.output_dir / "resolved_config.json", resolved)

    teacher = load_teacher(args.teacher_model, args.attention, device)
    student = load_student_with_lora(
        args.student_model,
        args.attention,
        device,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0)
    collator = DynamicKDCollator(
        pad_token_id=int(pad_token_id),
        max_length=args.max_length,
    )
    steps_per_epoch = math.ceil(
        len(samples) / (args.batch_size * args.grad_accum)
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = WeightedKDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[
            LossJsonlCallback(
                loss_log_path,
                reset=args.resume_from_checkpoint is None,
            )
        ],
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
    )
    write_json(
        args.output_dir / "loss_summary.json",
        summarize_rows(read_loss_rows(loss_log_path)),
    )
    trainer.model.save_pretrained(args.adapter_dir)
    processor.save_pretrained(args.adapter_dir)
    trainer.save_state()
    write_json(
        args.output_dir / "summary.json",
        {
            **resolved,
            "steps_per_epoch": steps_per_epoch,
            "train_metrics": train_result.metrics,
            "loss_log": str(loss_log_path),
            "adapter_dir": str(args.adapter_dir),
            "export_dir": None if args.skip_export else str(args.export_dir),
        },
    )
    if args.skip_export:
        print("Training smoke test complete; merged export skipped.")
        return 0

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    merged_model = trainer.model.merge_and_unload()
    enable_generation_cache(merged_model)
    args.export_dir.mkdir(parents=True, exist_ok=False)
    merged_model.save_pretrained(
        args.export_dir,
        safe_serialization=True,
        max_shard_size="1GB",
    )
    processor.save_pretrained(args.export_dir)
    state_path = args.student_model / "highway_state.json"
    if state_path.exists():
        shutil.copy2(state_path, args.export_dir / "highway_state.json")
    print(f"Recovery model saved: {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
