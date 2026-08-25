"""Local boundary recovery for a physically pruned Qwen3-VL.

After deleting current layer ``i``, only student layer ``i-1`` is trainable.
The frozen model before deletion is the teacher. Recovery projects teacher
boundary ``h_i`` and student boundary ``h_{i-1}`` through the same frozen final
norm and LM head, then minimizes field-weighted ``KL(q_i || q_{i-1})``.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from sft_scripts.utils.sft_io import load_training_samples

from .train_summarize import read_loss_rows, summarize_rows
from .utils.field_weights import weight_config
from .utils.io_utils import load_layer_state, write_json
from .utils.metrics import field_weighted_boundary_kl
from .utils.model_ops import (
    boundary_hidden,
    boundary_logits,
    configure_previous_layer_trainable,
    get_layers,
    prepare_language_inputs,
    save_pruned_model,
)
from .utils.recovery_config import RECOVERY_METHOD
from .utils.sft_dataset import WeightedSFTDataset
from .utils.training_log import LossJsonlCallback
from .utils.training_collator import DynamicKDCollator


class BoundaryKDTrainer(Trainer):
    """Train only student ``i-1`` to reproduce teacher boundary ``i``."""

    def __init__(
        self,
        *args: Any,
        teacher_model: Any,
        deleted_current_layer: int,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.deleted_current_layer = deleted_current_layer
        self.temperature = temperature
        self._component_sum = 0.0
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

        # Embedding and vision modules are frozen and identical on both sides.
        # Cache them once, then run only to the two aligned boundary positions.
        language_inputs = prepare_language_inputs(model, inputs)
        with torch.no_grad():
            teacher_hidden = boundary_hidden(
                self.teacher_model,
                language_inputs,
                self.deleted_current_layer,
            )
            teacher_logits = boundary_logits(
                self.teacher_model,
                teacher_hidden,
                keep_positions,
            )
        student_hidden = boundary_hidden(
            model,
            language_inputs,
            self.deleted_current_layer - 1,
        )
        # Use the teacher's frozen norm and LM head as one shared logit lens.
        student_logits = boundary_logits(
            self.teacher_model,
            student_hidden,
            keep_positions,
        )
        target_labels = shift_labels[:, keep_positions].to(student_logits.device)
        target_weights = shift_weights[:, keep_positions].to(
            student_logits.device,
            torch.float32,
        )
        losses = field_weighted_boundary_kl(
            student_logits,
            teacher_logits,
            target_labels,
            target_weights,
            temperature=self.temperature,
        )
        self._component_sum += float(losses.boundary_kl.detach().item())
        self._component_count += 1
        return (
            (losses.total, {"boundary_hidden": student_hidden})
            if return_outputs
            else losses.total
        )

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._component_count:
            logs = {
                **logs,
                "boundary_weighted_kl": (
                    self._component_sum / self._component_count
                ),
            }
            self._component_sum = 0.0
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


def load_student_for_boundary_recovery(
    path: Path,
    attention: str,
    device: torch.device,
    deleted_current_layer: int,
) -> tuple[Any, list[int], int]:
    student = AutoModelForImageTextToText.from_pretrained(
        path,
        **model_kwargs(attention),
    )
    student.config.use_cache = False
    trainable_layers, trainable_parameters = configure_previous_layer_trainable(
        student,
        deleted_current_layer,
    )
    student.to(device)
    print(
        "trainable layers: "
        f"{trainable_layers} || trainable params: {trainable_parameters:,}"
    )
    return student, trainable_layers, trainable_parameters


def validate_boundary_alignment(
    teacher: Any,
    student: Any,
    deleted_current_layer: int,
) -> None:
    teacher_layers = len(get_layers(teacher))
    student_layers = len(get_layers(student))
    if teacher_layers != student_layers + 1:
        raise ValueError(
            "Boundary recovery requires teacher and student to differ by exactly "
            f"one layer; got teacher={teacher_layers}, student={student_layers}"
        )
    if not 1 <= deleted_current_layer < teacher_layers:
        raise ValueError(
            "--deleted-layer must be in [1, teacher_layers-1]; "
            f"got {deleted_current_layer} for {teacher_layers} layers"
        )


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


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("--batch-size and --grad-accum must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.deleted_layer < 1:
        raise ValueError("--deleted-layer must be at least 1")

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

    teacher = load_teacher(args.teacher_model, args.attention, device)
    student, trainable_layers, trainable_parameters = (
        load_student_for_boundary_recovery(
            args.student_model,
            args.attention,
            device,
            args.deleted_layer,
        )
    )
    validate_boundary_alignment(teacher, student, args.deleted_layer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = args.output_dir / "loss.jsonl"
    resolved = {
        "recovery_method": RECOVERY_METHOD,
        "teacher_model": str(args.teacher_model.resolve()),
        "student_model": str(args.student_model.resolve()),
        "deleted_current_layer": args.deleted_layer,
        "teacher_boundary_layer": args.deleted_layer,
        "student_boundary_layer": args.deleted_layer - 1,
        "trainable_layers": trainable_layers,
        "trainable_parameters": trainable_parameters,
        "train_scope": "student_language_layer_i_minus_1_only",
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
        "loss": "field_weighted_KL(q_teacher_i||q_student_i_minus_1)",
        "temperature": args.temperature,
        "logit_lens": "shared_frozen_teacher_final_norm_and_lm_head",
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
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = BoundaryKDTrainer(
        model=student,
        teacher_model=teacher,
        deleted_current_layer=args.deleted_layer,
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
    write_json(
        args.output_dir / "summary.json",
        {
            **resolved,
            "steps_per_epoch": steps_per_epoch,
            "train_metrics": train_result.metrics,
            "loss_log": str(loss_log_path),
            "export_dir": None if args.skip_export else str(args.export_dir),
        },
    )
    if args.skip_export:
        print("Boundary recovery smoke test complete; model export skipped.")
        return 0

    if args.export_dir.exists():
        raise FileExistsError(f"Output model already exists: {args.export_dir}")
    state = load_layer_state(args.student_model, len(get_layers(student)))
    recovery_record = {
        "method": RECOVERY_METHOD,
        "teacher_model": str(args.teacher_model.resolve()),
        "deleted_current_layer": args.deleted_layer,
        "teacher_boundary_layer": args.deleted_layer,
        "student_boundary_layer": args.deleted_layer - 1,
        "trainable_layers": trainable_layers,
        "loss": resolved["loss"],
    }
    state = {
        **state,
        "recoveries": [*state.get("recoveries", []), recovery_record],
        "last_recovery": recovery_record,
    }
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_pruned_model(student, processor, args.export_dir, state)
    print(f"Boundary-recovered model saved: {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
