"""Evaluate real CE/KL/top-1 drops for saved Highway checkpoints.

This is different from probe:

- probe temporarily bypasses one layer inside one model to choose a candidate;
- this script compares saved checkpoints after distillation.

It reports both full-validation drops and the exact fixed-probe-sample drops.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .utils.data import HighwayDataset, move_sample, supervised_positions
from .eval_generation import (
    ModelTarget,
    checkpoint_for_round,
    read_json,
    read_jsonl,
    write_json,
)
from .utils.io_utils import append_jsonl, write_csv
from .utils.metrics import compare_logits
from .utils.model_ops import final_logits, load_model, load_processor, prepare_language_inputs


def release_models() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class Comparison:
    name: str
    split: str
    reference: ModelTarget
    candidate: ModelTarget
    sample_rows: list[dict[str, int]]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def model_targets(args: argparse.Namespace) -> list[ModelTarget]:
    targets = [ModelTarget("base_28_layers", args.base_model, 0)]
    for round_number in range(1, args.accepted_rounds + 1):
        checkpoint = checkpoint_for_round(
            args.source_run_dir,
            args.checkpoint_root,
            round_number,
        )
        if checkpoint is None:
            print(
                f"[eval-logits] skip round {round_number}: checkpoint not found",
                file=sys.stderr,
            )
            continue
        targets.append(
            ModelTarget(
                f"round_{round_number:02d}_post_recovery",
                checkpoint,
                round_number,
            )
        )
    return targets


def full_sample_rows(dataset: HighwayDataset, max_samples: int) -> list[dict[str, int]]:
    indices = dataset.valid_indices()
    if max_samples > 0:
        indices = indices[:max_samples]
    return [
        {
            "sample_index": int(index),
            "occurrence": occurrence,
            "probe_repeat": 0,
            "repeat_offset": occurrence,
        }
        for occurrence, index in enumerate(indices)
    ]


def probe_sample_rows(
    source_run_dir: Path,
    round_number: int,
    max_samples: int,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    fixed_manifest = source_run_dir / "probe" / "probe_10x10.jsonl"
    if fixed_manifest.exists():
        for occurrence, source_row in enumerate(read_jsonl(fixed_manifest)):
            rows.append(
                {
                    "sample_index": int(source_row["index_in_source"]),
                    "occurrence": occurrence,
                    "probe_repeat": int(source_row["repeat_id"]),
                    "repeat_offset": int(
                        source_row.get("position_in_repeat", occurrence + 1)
                    )
                    - 1,
                }
            )
    else:
        # Compatibility with the older per-round sampling record.
        selected = read_json(
            source_run_dir
            / f"round_{round_number:02d}"
            / "probe_selected.json"
        )
        for repeat_index, sample_indices in enumerate(
            selected["sample_subsets"], start=1
        ):
            for offset, sample_index in enumerate(sample_indices):
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "occurrence": len(rows),
                        "probe_repeat": repeat_index,
                        "repeat_offset": offset,
                    }
                )
    return rows[:max_samples] if max_samples > 0 else rows


def build_comparisons(
    args: argparse.Namespace,
    dataset: HighwayDataset,
    targets: list[ModelTarget],
) -> list[Comparison]:
    by_deleted = {target.deleted_layers: target for target in targets}
    base = by_deleted[0]
    full_rows = full_sample_rows(dataset, args.max_samples)
    comparisons: list[Comparison] = []

    if not args.skip_full:
        for round_number in range(1, args.accepted_rounds + 1):
            candidate = by_deleted.get(round_number)
            if candidate is None:
                continue
            comparisons.append(
                Comparison(
                    name=f"full_cumulative_round_{round_number:02d}",
                    split="full_val",
                    reference=base,
                    candidate=candidate,
                    sample_rows=full_rows,
                )
            )

        if args.include_full_stage:
            for round_number in range(1, args.accepted_rounds + 1):
                reference = by_deleted.get(round_number - 1)
                candidate = by_deleted.get(round_number)
                if reference is None or candidate is None:
                    continue
                comparisons.append(
                    Comparison(
                        name=f"full_stage_round_{round_number:02d}",
                        split="full_val",
                        reference=reference,
                        candidate=candidate,
                        sample_rows=full_rows,
                    )
                )

    if not args.skip_probe:
        for round_number in range(1, args.accepted_rounds + 1):
            reference = by_deleted.get(round_number - 1)
            candidate = by_deleted.get(round_number)
            if reference is None or candidate is None:
                continue
            comparisons.append(
                Comparison(
                    name=f"probe_round_{round_number:02d}_fixed",
                    split="probe_fixed",
                    reference=reference,
                    candidate=candidate,
                    sample_rows=probe_sample_rows(
                        args.source_run_dir,
                        round_number,
                        args.max_samples,
                    ),
                )
            )
    return comparisons


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "reference_ce",
        "candidate_ce",
        "ce_delta",
        "kl",
        "top1_agreement",
    ]
    summary: dict[str, Any] = {"samples": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = mean(values)
        summary[f"std_{key}"] = std(values)
    return summary


@torch.inference_mode()
def evaluate_comparison(
    args: argparse.Namespace,
    dataset: HighwayDataset,
    comparison: Comparison,
) -> dict[str, Any]:
    out_dir = args.output_dir / comparison.name
    detail_path = out_dir / "detail.jsonl"
    summary_path = out_dir / "summary.json"
    command_path = out_dir / "config.json"
    if summary_path.exists() and not args.force:
        print(f"[eval-logits] reuse {summary_path}")
        return read_json(summary_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        detail_path.unlink()
    write_json(
        command_path,
        {
            "comparison": {
                "name": comparison.name,
                "split": comparison.split,
                "reference": {
                    "label": comparison.reference.label,
                    "model_path": str(comparison.reference.model_path),
                    "deleted_layers": comparison.reference.deleted_layers,
                },
                "candidate": {
                    "label": comparison.candidate.label,
                    "model_path": str(comparison.candidate.model_path),
                    "deleted_layers": comparison.candidate.deleted_layers,
                },
                "sample_rows": comparison.sample_rows,
            },
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_length": args.max_length,
        },
    )

    print(
        f"[eval-logits] {comparison.name}: "
        f"{comparison.reference.label} -> {comparison.candidate.label}, "
        f"samples={len(comparison.sample_rows)}"
    )
    reference_model = load_model(
        comparison.reference.model_path,
        args.device,
        args.dtype,
        args.attention,
    )
    candidate_model = load_model(
        comparison.candidate.model_path,
        args.device,
        args.dtype,
        args.attention,
    )
    reference_model.eval()
    candidate_model.eval()

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for position, sample_row in enumerate(comparison.sample_rows, start=1):
        sample = move_sample(dataset[sample_row["sample_index"]], args.device)
        token_positions, targets = supervised_positions(sample.labels, args.max_tokens)

        reference_inputs = prepare_language_inputs(reference_model, sample.model_inputs)
        candidate_inputs = prepare_language_inputs(candidate_model, sample.model_inputs)
        reference_logits = final_logits(reference_model, reference_inputs, token_positions)
        candidate_logits = final_logits(candidate_model, candidate_inputs, token_positions)
        metrics = compare_logits(
            reference_logits,
            candidate_logits,
            targets,
            args.temperature,
        )
        row = {
            **sample_row,
            **asdict(metrics),
        }
        rows.append(row)
        append_jsonl(detail_path, row)
        if position == 1 or position % args.log_every == 0 or position == len(comparison.sample_rows):
            elapsed = time.monotonic() - started
            samples_per_second = position / max(elapsed, 1e-9)
            remaining = len(comparison.sample_rows) - position
            eta_seconds = remaining / max(samples_per_second, 1e-9)
            print(
                f"[eval-logits] {comparison.name} {position}/{len(comparison.sample_rows)} "
                f"mean ΔCE={mean([float(item['ce_delta']) for item in rows]):.6f} "
                f"mean KL={mean([float(item['kl']) for item in rows]):.6f} "
                f"eta={eta_seconds / 60:.1f}min"
            )
        del sample, reference_inputs, candidate_inputs, reference_logits, candidate_logits

    summary = {
        "comparison": {
            "name": comparison.name,
            "split": comparison.split,
            "reference_label": comparison.reference.label,
            "candidate_label": comparison.candidate.label,
            "reference_model": str(comparison.reference.model_path),
            "candidate_model": str(comparison.candidate.model_path),
            "reference_deleted_layers": comparison.reference.deleted_layers,
            "candidate_deleted_layers": comparison.candidate.deleted_layers,
        },
        **summarize_rows(rows),
        "elapsed_seconds": time.monotonic() - started,
        "detail_path": str(detail_path),
    }
    write_json(summary_path, summary)
    del reference_model, candidate_model
    release_models()
    return summary


def write_overall_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        comparison = summary["comparison"]
        rows.append(
            {
                **comparison,
                "samples": summary["samples"],
                "mean_reference_ce": summary["mean_reference_ce"],
                "mean_candidate_ce": summary["mean_candidate_ce"],
                "mean_ce_delta": summary["mean_ce_delta"],
                "std_ce_delta": summary["std_ce_delta"],
                "mean_kl": summary["mean_kl"],
                "std_kl": summary["std_kl"],
                "mean_top1_agreement": summary["mean_top1_agreement"],
                "std_top1_agreement": summary["std_top1_agreement"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "detail_path": summary["detail_path"],
            }
        )
    summary_csv = output_dir / "logit_eval_summary.csv"
    write_csv(summary_csv, rows)
    return summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CE/KL/top-1 drops for saved Highway checkpoints."
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attention",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument(
        "--include-full-stage",
        action="store_true",
        help="Also run full validation for each stage reference -> current pair.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> int:
    args = parse_args()
    args.device = torch.device(args.device)
    args.dtype = dtype_from_name(args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(args.base_model)
    dataset = HighwayDataset(args.val_json, processor, args.max_length)
    targets = model_targets(args)
    comparisons = build_comparisons(args, dataset, targets)
    write_json(
        args.output_dir / "logit_eval_plan.json",
        {
            "targets": [
                {
                    "label": target.label,
                    "model_path": str(target.model_path),
                    "deleted_layers": target.deleted_layers,
                }
                for target in targets
            ],
            "comparisons": [
                {
                    "name": comparison.name,
                    "split": comparison.split,
                    "reference": comparison.reference.label,
                    "candidate": comparison.candidate.label,
                    "samples": len(comparison.sample_rows),
                }
                for comparison in comparisons
            ],
        },
    )
    if args.dry_run:
        for comparison in comparisons:
            print(
                f"[eval-logits] dry-run {comparison.name}: "
                f"{comparison.reference.label} -> {comparison.candidate.label}, "
                f"samples={len(comparison.sample_rows)}"
            )
        return 0

    summaries = [
        evaluate_comparison(args, dataset, comparison)
        for comparison in comparisons
    ]
    summary_csv = write_overall_summary(args.output_dir, summaries)
    print(f"[eval-logits] summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
