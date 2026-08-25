"""Run real JSON-generation evals for Highway pruning checkpoints.

The probe/recovery metrics are teacher-forcing diagnostics.  This script closes
the loop by calling the project's generation evaluator on:

- the full validation set;
- the exact fixed probe samples used when each layer was selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORE_METRICS = [
    "cat_recognition_accuracy",
    "cat_count_accuracy",
    "behavior_analysis_accuracy",
    "behavior_accuracy_on_gt_cat_images",
]


@dataclass(frozen=True)
class ModelTarget:
    label: str
    model_path: Path
    deleted_layers: int


@dataclass(frozen=True)
class EvalTask:
    dataset: str
    val_json: Path
    model: ModelTarget
    reference_label: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def round_dir(run_dir: Path, round_number: int) -> Path:
    return run_dir / f"round_{round_number:02d}"


def checkpoint_for_round(
    source_run_dir: Path,
    checkpoint_root: Path,
    round_number: int,
) -> Path | None:
    """Map a source run's checkpoint basename onto the checkpoint root to eval."""

    summary_path = round_dir(source_run_dir, round_number) / "round_summary.json"
    if not summary_path.exists():
        return None
    current_candidate = (
        checkpoint_root
        / f"round_{round_number:02d}"
        / "post_recovery_model"
    )
    if current_candidate.exists():
        return current_candidate

    # Compatibility with the older flat checkpoint layout.
    summary = read_json(summary_path)
    if not summary.get("accepted"):
        return None
    source_checkpoint = summary.get("checkpoint")
    if not source_checkpoint:
        return None
    candidate = checkpoint_root / Path(str(source_checkpoint)).name
    return candidate if candidate.exists() else None


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
                f"[eval-generation] skip round {round_number}: checkpoint not found",
                file=sys.stderr,
            )
            continue
        targets.append(
            ModelTarget(
                label=f"round_{round_number:02d}_post_recovery",
                model_path=checkpoint,
                deleted_layers=round_number,
            )
        )
    return targets


def create_probe_manifest(
    source_run_dir: Path,
    round_number: int,
    val_json: Path,
    manifest_dir: Path,
) -> Path:
    """Write the fixed probe rows as a tiny JSONL SFT manifest."""

    rows: list[dict[str, Any]] = []
    fixed_manifest = source_run_dir / "probe" / "probe_10x10.jsonl"
    if fixed_manifest.exists():
        for occurrence, source_row in enumerate(read_jsonl(fixed_manifest)):
            rows.append(
                {
                    "source_json": str(val_json),
                    "index_in_source": int(source_row["index_in_source"]),
                    "probe_round": round_number,
                    "probe_repeat": int(source_row["repeat_id"]),
                    "repeat_offset": int(
                        source_row.get("position_in_repeat", occurrence + 1)
                    )
                    - 1,
                }
            )
    else:
        # Compatibility with the older per-round sampling record.
        selected_path = round_dir(source_run_dir, round_number) / "probe_selected.json"
        selected = read_json(selected_path)
        for repeat_index, sample_indices in enumerate(
            selected["sample_subsets"], start=1
        ):
            for offset, sample_index in enumerate(sample_indices):
                rows.append(
                    {
                        "source_json": str(val_json),
                        "index_in_source": int(sample_index),
                        "probe_round": round_number,
                        "probe_repeat": repeat_index,
                        "repeat_offset": offset,
                    }
                )
    manifest = manifest_dir / f"probe_round_{round_number:02d}_fixed.jsonl"
    write_jsonl(manifest, rows)
    return manifest


def output_dir(root: Path, task: EvalTask) -> Path:
    return root / task.dataset / task.model.label


def run_task(args: argparse.Namespace, task: EvalTask) -> Path:
    out_dir = output_dir(args.output_dir, task)
    metrics_path = out_dir / "metrics.json"
    command_path = out_dir / "command.json"
    log_path = out_dir / "eval.log"
    command = [
        str(args.python),
        str(args.eval_script),
        "--model-path",
        str(task.model.model_path),
        "--backend",
        args.backend,
        "--val-json",
        str(task.val_json),
        "--output-dir",
        str(out_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.max_samples:
        command.extend(["--max-samples", str(args.max_samples)])

    write_json(
        command_path,
        {
            "dataset": task.dataset,
            "model_label": task.model.label,
            "reference_label": task.reference_label,
            "command": command,
        },
    )
    if metrics_path.exists() and not args.force:
        print(f"[eval-generation] reuse {metrics_path}")
        return metrics_path
    if args.dry_run:
        print("[eval-generation] dry-run:", " ".join(command))
        return metrics_path

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval-generation] run {task.dataset} / {task.model.label}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=args.project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Eval failed for {task.dataset}/{task.model.label}; see {log_path}"
        )
    print(f"[eval-generation] done in {elapsed / 60:.1f} min: {metrics_path}")
    return metrics_path


def plan_tasks(args: argparse.Namespace, targets: list[ModelTarget]) -> list[EvalTask]:
    by_deleted_layers = {target.deleted_layers: target for target in targets}
    tasks: list[EvalTask] = []

    if not args.skip_full:
        for target in targets:
            tasks.append(
                EvalTask(
                    dataset="full_val",
                    val_json=args.val_json,
                    model=target,
                    reference_label="base_28_layers",
                )
            )

    if not args.skip_probe:
        manifest_dir = args.output_dir / "probe_manifests"
        for round_number in range(1, args.accepted_rounds + 1):
            current = by_deleted_layers.get(round_number)
            previous = by_deleted_layers.get(round_number - 1)
            if current is None or previous is None:
                print(
                    f"[eval-generation] skip probe round {round_number}: "
                    "missing previous/current checkpoint",
                    file=sys.stderr,
                )
                continue
            manifest = create_probe_manifest(
                args.source_run_dir,
                round_number,
                args.val_json,
                manifest_dir,
            )
            dataset = f"probe_round_{round_number:02d}_fixed"
            tasks.append(
                EvalTask(
                    dataset=dataset,
                    val_json=manifest,
                    model=previous,
                    reference_label=previous.label,
                )
            )
            tasks.append(
                EvalTask(
                    dataset=dataset,
                    val_json=manifest,
                    model=current,
                    reference_label=previous.label,
                )
            )
    return tasks


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def summarize(args: argparse.Namespace, tasks: list[EvalTask]) -> Path:
    metrics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        metrics_path = output_dir(args.output_dir, task) / "metrics.json"
        if metrics_path.exists():
            metrics_by_key[(task.dataset, task.model.label)] = read_json(metrics_path)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        metrics = metrics_by_key.get((task.dataset, task.model.label))
        if metrics is None:
            continue
        reference_metrics = metrics_by_key.get((task.dataset, task.reference_label), metrics)
        speed = metrics.get("speed", {}) if isinstance(metrics.get("speed"), dict) else {}
        paths = metrics.get("paths", {}) if isinstance(metrics.get("paths"), dict) else {}
        row: dict[str, Any] = {
            "dataset": task.dataset,
            "model_label": task.model.label,
            "reference_label": task.reference_label,
            "deleted_layers": task.model.deleted_layers,
            "model_path": str(task.model.model_path),
            "metrics_path": str(output_dir(args.output_dir, task) / "metrics.json"),
            "predictions_path": str(paths.get("predictions", "")),
            "total_samples": metrics.get("total_samples", 0),
            "parse_failed": metrics.get("parse_failed", 0),
            "image_missing": metrics.get("image_missing", 0),
            "inference_failed": metrics.get("inference_failed", 0),
            "schema_failed": metrics.get("schema_validation", {}).get("failed", 0),
            "generated_items": speed.get("generated_items", 0),
            "wall_seconds": speed.get("wall_seconds", 0),
            "inference_items_per_second": speed.get("inference_items_per_second", 0),
        }
        for metric_name in CORE_METRICS:
            current_value = metric_value(metrics, metric_name)
            reference_value = metric_value(reference_metrics, metric_name)
            row[metric_name] = current_value
            row[f"delta_{metric_name}"] = current_value - reference_value
        rows.append(row)

    summary_path = args.output_dir / "generation_eval_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_path.write_text("", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full/probe generation eval for Highway checkpoints."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-rounds", type=int, default=3)
    parser.add_argument("--backend", default="qwen3vl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = model_targets(args)
    if len(targets) <= 1:
        print("[eval-generation] no pruned checkpoints found", file=sys.stderr)
    tasks = plan_tasks(args, targets)
    write_json(
        args.output_dir / "generation_eval_plan.json",
        {
            "targets": [target.__dict__ | {"model_path": str(target.model_path)} for target in targets],
            "tasks": [
                {
                    "dataset": task.dataset,
                    "val_json": str(task.val_json),
                    "model_label": task.model.label,
                    "reference_label": task.reference_label,
                }
                for task in tasks
            ],
        },
    )
    for task in tasks:
        run_task(args, task)
    summary_path = summarize(args, tasks)
    print(f"[eval-generation] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
