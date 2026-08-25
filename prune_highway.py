"""Restartable multi-round pruning with local boundary recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .utils.io_utils import write_json
from .utils.recovery_config import RECOVERY_METHOD


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def run_logged(
    command: list[str],
    log_path: Path,
    project_root: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture child output")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Command failed with exit code {return_code}; see {log_path}"
        )


def ensure_atomic_metrics(
    python: Path,
    project_root: Path,
    eval_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    atomic_path = eval_dir / "atomic_metrics.json"
    if not atomic_path.exists():
        run_logged(
            [
                str(python),
                "-m",
                "highway.eval_summarize",
                "--eval-dir",
                str(eval_dir),
            ],
            log_path,
            project_root,
        )
    return read_json(atomic_path)


def ensure_full_eval(
    *,
    python: Path,
    project_root: Path,
    eval_script: Path,
    model_path: Path,
    val_data: Path,
    eval_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    attention: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = eval_dir / "metrics.json"
    log_path = eval_dir / "console.log"
    if not metrics_path.exists():
        run_logged(
            [
                str(python),
                str(eval_script),
                "--model-path",
                str(model_path),
                "--backend",
                "qwen3vl",
                "--val-json",
                str(val_data),
                "--output-dir",
                str(eval_dir),
                "--batch-size",
                str(batch_size),
                "--max-new-tokens",
                str(max_new_tokens),
                "--local-files-only",
                "--image-max-pixels",
                "262144",
                "--attn-implementation",
                attention,
            ],
            log_path,
            project_root,
        )
    atomic = ensure_atomic_metrics(
        python,
        project_root,
        eval_dir,
        log_path,
    )
    return read_json(metrics_path), atomic


def latest_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = []
    for path in checkpoint_root.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def generated_model_ready(path: Path) -> bool:
    """Only accept a generated checkpoint after its final state marker exists."""

    if not path.is_dir():
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return (
        has_weights
        and (path / "config.json").is_file()
        and (path / "highway_state.json").is_file()
    )


def quarantine_incomplete_model(path: Path) -> Path | None:
    """Preserve an interrupted export without letting it masquerade as complete."""

    if not path.exists() or generated_model_ready(path):
        return None
    suffix = 1
    while True:
        destination = path.with_name(f"{path.name}.incomplete_{suffix:02d}")
        if not destination.exists():
            path.rename(destination)
            print(f"[orchestrator] quarantined incomplete model: {destination}")
            return destination
        suffix += 1


def require_current_recovery_method(
    summary: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    method = summary.get("recovery_method")
    if method != RECOVERY_METHOD:
        raise RuntimeError(
            f"{source} uses recovery_method={method!r}, but this code requires "
            f"{RECOVERY_METHOD!r}. Use a new run-dir/model-root instead of "
            "mixing global all-linear KD results with local boundary recovery."
        )
    return summary


def ensure_recovery_training(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    teacher_model: Path,
    pre_model: Path,
    post_model: Path,
    deleted_current_layer: int,
) -> dict[str, Any]:
    train_dir = round_dir / "train"
    summary_path = train_dir / "summary.json"
    if summary_path.exists():
        require_current_recovery_method(read_json(summary_path), summary_path)
    quarantine_incomplete_model(post_model)
    if not generated_model_ready(post_model):
        command = [
            str(args.python),
            "-m",
            "highway.train_weighted_kd",
            "--teacher-model",
            str(teacher_model),
            "--student-model",
            str(pre_model),
            "--deleted-layer",
            str(deleted_current_layer),
            "--train-data",
            str(args.train_data),
            "--output-dir",
            str(train_dir),
            "--export-dir",
            str(post_model),
            "--epochs",
            str(args.recovery_epochs),
            "--max-length",
            "2560",
            "--image-max-pixels",
            "262144",
            "--batch-size",
            str(args.recovery_batch_size),
            "--grad-accum",
            str(args.recovery_effective_batch_size // args.recovery_batch_size),
            "--learning-rate",
            "1e-4",
            "--temperature",
            "1.0",
            "--logging-steps",
            "5",
            "--save-steps",
            "100",
            "--attention",
            args.attention,
        ]
        checkpoint = latest_checkpoint(train_dir / "checkpoints")
        if checkpoint is not None:
            command.extend(["--resume-from-checkpoint", str(checkpoint)])
        run_logged(
            command,
            train_dir / "console.log",
            args.project_root,
        )
    if not generated_model_ready(post_model):
        raise RuntimeError(f"Recovery model export is incomplete: {post_model}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Recovery summary is missing: {summary_path}")
    return require_current_recovery_method(read_json(summary_path), summary_path)


def ensure_post_baseline_preflight(args: argparse.Namespace) -> None:
    marker = args.run_dir / "preflight" / "post_baseline_boundary_complete.json"
    if marker.exists():
        return
    preflight_dir = args.run_dir / "preflight"
    smoke_probe_dir = preflight_dir / "prune_probe_layers_3_27"
    smoke_selected = smoke_probe_dir / "selected_layer.json"
    if not smoke_selected.exists():
        run_logged(
            [
                str(args.python),
                "-m",
                "highway.prune_probe",
                "--reference-model",
                str(args.reference_model),
                "--candidate-model",
                str(args.reference_model),
                "--val-data",
                str(args.val_data),
                "--probe-manifest",
                str(args.run_dir / "probe" / "probe_10x10.jsonl"),
                "--baseline-predictions",
                str(args.run_dir / "baseline" / "predictions.jsonl"),
                "--output-dir",
                str(smoke_probe_dir),
                "--candidate-layers",
                "3,27",
                "--max-probe-rows",
                "16",
                "--generation-batch-size",
                str(args.generation_batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--attention",
                args.attention,
            ],
            smoke_probe_dir / "console.log",
            args.project_root,
        )

    smoke_model = args.model_root / "preflight" / "pruned_layer_12"
    smoke_deletion = preflight_dir / "pruned_layer_12.deletion.json"
    quarantine_incomplete_model(smoke_model)
    if not generated_model_ready(smoke_model):
        run_logged(
            [
                str(args.python),
                "-m",
                "highway.prune_layer",
                "--model",
                str(args.reference_model),
                "--layer",
                "12",
                "--output-model",
                str(smoke_model),
                "--record",
                str(smoke_deletion),
                "--attention",
                args.attention,
            ],
            preflight_dir / "prune_layer_12.console.log",
            args.project_root,
        )
    if not generated_model_ready(smoke_model):
        raise RuntimeError(f"Pruning smoke model is incomplete: {smoke_model}")

    smoke_eval_dir = preflight_dir / "eval_pruned_layer_12"
    if not (smoke_eval_dir / "metrics.json").exists():
        run_logged(
            [
                str(args.python),
                str(args.eval_script),
                "--model-path",
                str(smoke_model),
                "--backend",
                "qwen3vl",
                "--val-json",
                str(args.val_data),
                "--output-dir",
                str(smoke_eval_dir),
                "--max-samples",
                "1",
                "--batch-size",
                "1",
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--local-files-only",
                "--image-max-pixels",
                "262144",
                "--attn-implementation",
                args.attention,
            ],
            smoke_eval_dir / "console.log",
            args.project_root,
        )

    # Two optimizer steps are required: with a warmup scheduler, a one-step
    # smoke can execute its only optimizer step at learning rate zero.
    kd_dir = preflight_dir / "boundary_kd_two_step"
    if not (kd_dir / "summary.json").exists():
        run_logged(
            [
                str(args.python),
                "-m",
                "highway.train_weighted_kd",
                "--teacher-model",
                str(args.reference_model),
                "--student-model",
                str(smoke_model),
                "--deleted-layer",
                "12",
                "--train-data",
                str(args.train_data),
                "--output-dir",
                str(kd_dir),
                "--export-dir",
                str(args.model_root / "preflight" / "unused_boundary_smoke_export"),
                "--epochs",
                "1",
                "--max-samples",
                "2",
                "--max-steps",
                "2",
                "--batch-size",
                "1",
                "--grad-accum",
                "1",
                "--logging-steps",
                "1",
                "--save-steps",
                "100",
                "--attention",
                args.attention,
                "--skip-export",
            ],
            kd_dir / "console.log",
            args.project_root,
        )
    write_json(
        marker,
        {
            "status": "passed",
            "prune_probe": str(smoke_selected),
            "pruned_model": str(smoke_model),
            "pruned_eval": str(smoke_eval_dir / "metrics.json"),
            "kd_summary": str(kd_dir / "summary.json"),
            "recovery_method": RECOVERY_METHOD,
        },
    )


def round_summary(
    *,
    round_number: int,
    selected: dict[str, Any],
    deletion: dict[str, Any],
    pre_metrics: dict[str, Any],
    pre_atomic: dict[str, Any],
    train_summary: dict[str, Any],
    post_metrics: dict[str, Any],
    post_atomic: dict[str, Any],
    baseline_atomic: dict[str, Any],
) -> dict[str, Any]:
    baseline_accuracy = float(baseline_atomic["macro_field_accuracy"])
    pre_accuracy = float(pre_atomic["macro_field_accuracy"])
    post_accuracy = float(post_atomic["macro_field_accuracy"])
    return {
        "round": round_number,
        "selected_layer": selected,
        "deletion": deletion,
        "pre_recovery": {
            "macro_field_accuracy": pre_accuracy,
            "delta_vs_baseline": pre_accuracy - baseline_accuracy,
            "parse_failed": pre_metrics.get("parse_failed"),
            "speed": pre_metrics.get("speed"),
        },
        "training": train_summary,
        "post_recovery": {
            "macro_field_accuracy": post_accuracy,
            "delta_vs_baseline": post_accuracy - baseline_accuracy,
            "recovered_from_pre": post_accuracy - pre_accuracy,
            "parse_failed": post_metrics.get("parse_failed"),
            "speed": post_metrics.get("speed"),
        },
    }


def write_final_tables(
    run_dir: Path,
    baseline_metrics: dict[str, Any],
    baseline_atomic: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    baseline_accuracy = float(baseline_atomic["macro_field_accuracy"])
    baseline_speed = float(
        baseline_metrics.get("speed", {}).get("inference_items_per_second") or 0.0
    )
    rows = [
        {
            "stage": "baseline_28_layers",
            "round": 0,
            "layers": 28,
            "selected_current_layer": "",
            "selected_original_layer": "",
            "probe_reference_macro_accuracy": "",
            "probe_candidate_macro_accuracy": "",
            "probe_relative_hard_regret": "",
            "probe_normalized_js": "",
            "probe_pruning_risk": "",
            "macro_field_accuracy": baseline_accuracy,
            "delta_vs_baseline": 0.0,
            "inference_items_per_second": baseline_speed,
            "speedup_vs_baseline": 1.0,
        }
    ]
    for summary in summaries:
        round_number = int(summary["round"])
        layers = 28 - round_number
        selected = summary["selected_layer"]
        for phase in ("pre_recovery", "post_recovery"):
            stage = summary[phase]
            stage_speed = float(
                stage.get("speed", {}).get("inference_items_per_second") or 0.0
            )
            rows.append(
                {
                    "stage": f"round_{round_number:02d}_{phase}",
                    "round": round_number,
                    "layers": layers,
                    "selected_current_layer": selected["current_layer"],
                    "selected_original_layer": selected["original_layer"],
                    "probe_reference_macro_accuracy": selected[
                        "mean_reference_macro_accuracy"
                    ],
                    "probe_candidate_macro_accuracy": selected[
                        "mean_candidate_macro_accuracy"
                    ],
                    "probe_relative_hard_regret": selected[
                        "mean_relative_hard_regret"
                    ],
                    "probe_normalized_js": selected["mean_normalized_js"],
                    "probe_pruning_risk": selected["mean_pruning_risk"],
                    "macro_field_accuracy": stage["macro_field_accuracy"],
                    "delta_vs_baseline": stage["delta_vs_baseline"],
                    "inference_items_per_second": stage_speed,
                    "speedup_vs_baseline": (
                        stage_speed / baseline_speed
                        if baseline_speed > 0
                        else 0.0
                    ),
                }
            )
    table_path = final_dir / "all_rounds.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(final_dir / "all_rounds.json", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--recovery-batch-size", type=int, default=1)
    parser.add_argument("--recovery-effective-batch-size", type=int, default=16)
    parser.add_argument("--recovery-epochs", type=float, default=2.0)
    parser.add_argument(
        "--retain-pre-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument(
        "--post-baseline-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    if args.recovery_batch_size < 1:
        raise ValueError("--recovery-batch-size must be positive")
    if args.recovery_epochs <= 0:
        raise ValueError("--recovery-epochs must be positive")
    if (
        args.recovery_effective_batch_size < args.recovery_batch_size
        or args.recovery_effective_batch_size % args.recovery_batch_size
    ):
        raise ValueError(
            "--recovery-effective-batch-size must be a positive multiple of "
            "--recovery-batch-size"
        )
    write_json(
        args.run_dir / "orchestrator_config.json",
        {
            "project_root": str(args.project_root),
            "python": str(args.python),
            "eval_script": str(args.eval_script),
            "reference_model": str(args.reference_model),
            "train_data": str(args.train_data),
            "val_data": str(args.val_data),
            "run_dir": str(args.run_dir),
            "model_root": str(args.model_root),
            "rounds": args.rounds,
            "eval_batch_size": args.eval_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "recovery_batch_size": args.recovery_batch_size,
            "recovery_effective_batch_size": args.recovery_effective_batch_size,
            "recovery_epochs": args.recovery_epochs,
            "recovery_method": RECOVERY_METHOD,
            "recovery_train_scope": "student_language_layer_i_minus_1_only",
            "recovery_loss": "field_weighted_boundary_kl",
            "retain_pre_models": args.retain_pre_models,
            "attention": args.attention,
            "post_baseline_preflight": args.post_baseline_preflight,
        },
    )
    baseline_dir = args.run_dir / "baseline"
    baseline_metrics, baseline_atomic = ensure_full_eval(
        python=args.python,
        project_root=args.project_root,
        eval_script=args.eval_script,
        model_path=args.reference_model,
        val_data=args.val_data,
        eval_dir=baseline_dir,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        attention=args.attention,
    )
    if args.post_baseline_preflight:
        ensure_post_baseline_preflight(args)
    manifest = args.run_dir / "probe" / "probe_10x10.jsonl"
    baseline_predictions = baseline_dir / "predictions.jsonl"
    current_model = args.reference_model
    summaries = []

    for round_number in range(1, args.rounds + 1):
        round_dir = args.run_dir / f"round_{round_number:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        model_round_dir = args.model_root / f"round_{round_number:02d}"
        pre_model = model_round_dir / "pre_recovery_model"
        post_model = model_round_dir / "post_recovery_model"
        round_summary_path = round_dir / "round_summary.json"
        if round_summary_path.exists() and generated_model_ready(post_model):
            summary = read_json(round_summary_path)
            training = summary.get("training")
            if not isinstance(training, dict):
                raise RuntimeError(
                    f"{round_summary_path} has no auditable training summary"
                )
            require_current_recovery_method(training, round_summary_path)
            if not args.retain_pre_models and pre_model.exists():
                shutil.rmtree(pre_model)
            summaries.append(summary)
            current_model = post_model
            print(f"[orchestrator] round {round_number:02d} already complete")
            continue

        probe_dir = round_dir / "probe"
        selected_path = probe_dir / "selected_layer.json"
        if not selected_path.exists():
            run_logged(
                [
                    str(args.python),
                    "-m",
                    "highway.prune_probe",
                    "--reference-model",
                    str(args.reference_model),
                    "--candidate-model",
                    str(current_model),
                    "--val-data",
                    str(args.val_data),
                    "--probe-manifest",
                    str(manifest),
                    "--baseline-predictions",
                    str(baseline_predictions),
                    "--output-dir",
                    str(probe_dir),
                    "--generation-batch-size",
                    str(args.generation_batch_size),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--attention",
                    args.attention,
                    "--min-layer",
                    "3",
                ],
                probe_dir / "console.log",
                args.project_root,
            )
        selected_document = read_json(selected_path)
        selected = selected_document["selected"]
        selected_current_layer = int(selected["current_layer"])

        deletion_path = round_dir / "deletion.json"
        quarantine_incomplete_model(pre_model)
        if not generated_model_ready(pre_model):
            run_logged(
                [
                    str(args.python),
                    "-m",
                    "highway.prune_layer",
                    "--model",
                    str(current_model),
                    "--layer",
                    str(selected_current_layer),
                    "--output-model",
                    str(pre_model),
                    "--record",
                    str(deletion_path),
                    "--attention",
                    args.attention,
                ],
                round_dir / "prune.console.log",
                args.project_root,
            )
        if not generated_model_ready(pre_model):
            raise RuntimeError(f"Pruned model export is incomplete: {pre_model}")
        deletion = read_json(deletion_path)
        pre_metrics, pre_atomic = ensure_full_eval(
            python=args.python,
            project_root=args.project_root,
            eval_script=args.eval_script,
            model_path=pre_model,
            val_data=args.val_data,
            eval_dir=round_dir / "eval_pre_recovery",
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            attention=args.attention,
        )

        train_summary = ensure_recovery_training(
            args=args,
            round_dir=round_dir,
            teacher_model=current_model,
            pre_model=pre_model,
            post_model=post_model,
            deleted_current_layer=selected_current_layer,
        )
        post_metrics, post_atomic = ensure_full_eval(
            python=args.python,
            project_root=args.project_root,
            eval_script=args.eval_script,
            model_path=post_model,
            val_data=args.val_data,
            eval_dir=round_dir / "eval_post_recovery",
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            attention=args.attention,
        )
        summary = round_summary(
            round_number=round_number,
            selected=selected,
            deletion=deletion,
            pre_metrics=pre_metrics,
            pre_atomic=pre_atomic,
            train_summary=train_summary,
            post_metrics=post_metrics,
            post_atomic=post_atomic,
            baseline_atomic=baseline_atomic,
        )
        write_json(round_summary_path, summary)
        if not args.retain_pre_models and pre_model.exists():
            shutil.rmtree(pre_model)
            write_json(
                round_dir / "pre_model_removed.json",
                {
                    "status": "removed_after_post_recovery_eval",
                    "path": str(pre_model),
                    "replacement": str(post_model),
                    "reason": "bounded model-disk usage for nine pruning rounds",
                },
            )
        summaries.append(summary)
        current_model = post_model

    write_final_tables(
        args.run_dir,
        baseline_metrics,
        baseline_atomic,
        summaries,
    )
    write_json(
        args.run_dir / "final" / "experiment_complete.json",
        {
            "status": "complete",
            "rounds": args.rounds,
            "final_model": str(current_model),
        },
    )
    print(f"[orchestrator] complete: {current_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
