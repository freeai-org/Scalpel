"""Final-logit weighted KD recovery for a physically pruned Qwen3-VL.

Scalpel 每轮先物理删除一层，再用固定 reference teacher 的最终输出分布
和 ground-truth assistant tokens 恢复 student。训练只更新 LoRA adapter；
导出时会把 LoRA merge 回 student，生成下一轮继续剪枝的完整模型。
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from sft_scripts.utils.sft_io import load_training_samples

from .train_summarize import read_loss_rows, summarize_rows
from .utils.field_weights import weight_config
from .utils.io_utils import load_layer_state, write_json
from .utils.kd_runtime import attach_lora_to_student, freeze_teacher
from .utils.metrics import field_weighted_kd_loss
from .utils.model_ops import get_layers, save_pruned_model
from .utils.recovery_config import RECOVERY_METHOD
from .utils.sft_dataset import WeightedSFTDataset
from .utils.training_collator import DynamicKDCollator
from .utils.training_log import LossJsonlCallback


class WeightedKDTrainer(Trainer):
    """Teacher-forcing 下对齐最终 logits，并记录 CE/KL 两个分量。"""

    def __init__(
        self,
        *args: Any,
        teacher_model: Any,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self._component_sums = {
            "hard_weighted_ce": 0.0,
            "soft_weighted_kl": 0.0,
        }
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

        # 只保留第一个 assistant token 之后的位置，少算 prompt 的无效 logits。
        first_valid = int(torch.nonzero(valid_mask, as_tuple=False)[:, 1].min().item())
        keep_positions = torch.arange(
            first_valid,
            shift_labels.shape[-1],
            device=shift_labels.device,
            dtype=torch.long,
        )
        model_inputs = {**inputs, "use_cache": False}

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**model_inputs)
        student_outputs = model(**model_inputs)

        teacher_logits = teacher_outputs.logits[:, :-1, :][:, keep_positions].float()
        student_logits = student_outputs.logits[:, :-1, :][:, keep_positions].float()
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
            temperature=self.temperature,
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
                    key: value / self._component_count
                    for key, value in self._component_sums.items()
                },
            }
            for key in self._component_sums:
                self._component_sums[key] = 0.0
            self._component_count = 0
        super().log(logs, *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--deleted-layer", type=int, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--temperature", type=float, default=1.0)
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


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("--batch-size and --grad-accum must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("--lora-rank and --lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.deleted_layer < 0:
        raise ValueError("--deleted-layer must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)

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

    teacher = freeze_teacher(args.teacher_model, args.attention, device)
    student, trainable_stats = attach_lora_to_student(
        args.student_model,
        args.attention,
        device,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = args.output_dir / "loss.jsonl"
    adapter_dir = args.output_dir / "adapter"
    resolved = {
        "recovery_method": RECOVERY_METHOD,
        "teacher_model": str(args.teacher_model.resolve()),
        "student_model": str(args.student_model.resolve()),
        "deleted_current_layer": args.deleted_layer,
        "teacher_target": "final_logits",
        "student_target": "final_logits",
        "train_scope": "student_lora_all_linear",
        "trainable_parameters": trainable_stats.trainable_parameters,
        "total_parameters": trainable_stats.total_parameters,
        "trainable_ratio": trainable_stats.ratio,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": "all-linear",
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
        "loss": "field_weighted_ce + field_weighted_KL(teacher_final||student_final)",
        "temperature": args.temperature,
        "field_weights": weight_config(),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
    }
    write_json(args.output_dir / "resolved_config.json", resolved)

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
        temperature=args.temperature,
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
    trainer.save_state()
    student.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    write_json(
        args.output_dir / "summary.json",
        {
            **resolved,
            "steps_per_epoch": steps_per_epoch,
            "train_metrics": train_result.metrics,
            "loss_log": str(loss_log_path),
            "adapter_dir": str(adapter_dir),
            "export_dir": None if args.skip_export else str(args.export_dir),
        },
    )
    if args.skip_export:
        print("Weighted KD smoke test complete; model export skipped.")
        return 0

    if args.export_dir.exists():
        raise FileExistsError(f"Output model already exists: {args.export_dir}")
    merged_student = student.merge_and_unload()
    state = load_layer_state(args.student_model, len(get_layers(merged_student)))
    recovery_record = {
        "method": RECOVERY_METHOD,
        "teacher_model": str(args.teacher_model.resolve()),
        "deleted_current_layer": args.deleted_layer,
        "train_scope": resolved["train_scope"],
        "loss": resolved["loss"],
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": "all-linear",
    }
    state = {
        **state,
        "recoveries": [*state.get("recoveries", []), recovery_record],
        "last_recovery": recovery_record,
    }
    del teacher, student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_pruned_model(merged_student, processor, args.export_dir, state)
    print(f"Weighted-KD recovered model saved: {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
