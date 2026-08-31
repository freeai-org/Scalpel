"""Restartable 10-round Qwen3-VL pruning experiment for the Scalpel mixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .utils.io_utils import write_json
from .utils.mixture_data import (
    GROUP_ORDER,
    MIXTURE_ORDER_GROUPED,
    MIXTURE_ORDER_INTERLEAVED,
    load_fixed_group_sample,
    parquet_files,
    partition_summaries,
    write_samples_jsonl,
)
from .utils.recovery_config import RECOVERY_METHOD


PACKAGE_NAME = __package__.split(".")[0]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def run_logged(
    command: list[str],
    log_path: Path,
    project_root: Path,
) -> None:
    """Run one restartable stage with both package roots on PYTHONPATH."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_parent = Path(__file__).resolve().parent.parent
    python_paths = [str(source_parent), str(project_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
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


def generated_model_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "highway_state.json").is_file()
        and (any(path.glob("*.safetensors")) or any(path.glob("*.bin")))
    )


def quarantine_incomplete_model(path: Path) -> Path | None:
    if not path.exists() or generated_model_ready(path):
        return None
    suffix = 1
    while True:
        destination = path.with_name(f"{path.name}.incomplete_{suffix:02d}")
        if not destination.exists():
            path.rename(destination)
            print(f"[mixture-experiment] quarantined {destination}", flush=True)
            return destination
        suffix += 1


def latest_checkpoint(root: Path) -> Path | None:
    candidates = []
    for path in root.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))[1]


def sha256_files(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_data(args: argparse.Namespace) -> Path:
    probe_dir = args.run_dir / "probe"
    probe_path = probe_dir / "fixed_4x20.jsonl"
    samples, probe_stats = load_fixed_group_sample(
        args.val_data,
        samples_per_group=args.test_samples_per_group,
        seed=args.data_seed,
    )
    expected_groups = [
        group
        for group in GROUP_ORDER
        for _ in range(args.test_samples_per_group)
    ]
    actual_groups = [str(sample.get("group")) for sample in samples]
    if actual_groups != expected_groups:
        raise RuntimeError("Fixed probe does not obey the requested group order")
    if probe_path.exists():
        existing_ids = [
            json.loads(line).get("id")
            for line in probe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_ids = [sample.get("id") for sample in samples]
        if existing_ids != expected_ids:
            raise RuntimeError(
                f"Existing fixed probe was created with another seed: {probe_path}"
            )
    else:
        write_samples_jsonl(probe_path, samples)

    partitions = partition_summaries(
        args.train_data,
        args.train_parts,
        args.data_seed,
    )
    write_json(args.run_dir / "train_partitions.json", {"parts": partitions})
    write_json(
        probe_dir / "manifest.json",
        {
            "seed": args.data_seed,
            "samples_per_group": args.test_samples_per_group,
            "group_order": list(GROUP_ORDER),
            "stats": probe_stats,
            "sample_ids": [sample.get("id") for sample in samples],
            "path": str(probe_path.resolve()),
        },
    )
    fingerprints_path = args.run_dir / "dataset_fingerprints.json"
    if not fingerprints_path.exists():
        write_json(
            fingerprints_path,
            {
                "train": {
                    "path": str(args.train_data.resolve()),
                    "sha256": sha256_files(parquet_files(args.train_data)),
                },
                "validation": {
                    "path": str(args.val_data.resolve()),
                    "sha256": sha256_files(parquet_files(args.val_data)),
                },
                "fixed_probe": {
                    "path": str(probe_path.resolve()),
                    "sha256": sha256_files([probe_path]),
                },
            },
        )
    return probe_path


def ensure_eval(
    args: argparse.Namespace,
    model: Path,
    data: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        run_logged(
            [
                str(args.python),
                "-m",
                f"{PACKAGE_NAME}.eval_text",
                "--model",
                str(model),
                "--data",
                str(data),
                "--output-dir",
                str(output_dir),
                "--max-length",
                str(args.max_length),
                "--logit-chunk-size",
                str(args.logit_chunk_size),
                "--attention",
                args.attention,
            ],
            output_dir / "console.log",
            args.project_root,
        )
    return read_json(metrics_path)


def ensure_probe(
    args: argparse.Namespace,
    model: Path,
    data: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selected_path = output_dir / "selected_layer.json"
    if not selected_path.exists():
        run_logged(
            [
                str(args.python),
                "-m",
                f"{PACKAGE_NAME}.prune_text",
                "--model",
                str(model),
                "--data",
                str(data),
                "--output-dir",
                str(output_dir),
                "--min-layer",
                str(args.min_layer),
                "--max-length",
                str(args.max_length),
                "--max-answer-tokens",
                str(args.probe_max_answer_tokens),
                "--logit-chunk-size",
                str(args.logit_chunk_size),
                "--attention",
                args.attention,
            ],
            output_dir / "console.log",
            args.project_root,
        )
    return read_json(selected_path)


def ensure_pruned_model(
    args: argparse.Namespace,
    current_model: Path,
    layer: int,
    output_model: Path,
    record: Path,
    log_path: Path,
) -> dict[str, Any]:
    quarantine_incomplete_model(output_model)
    if not generated_model_ready(output_model):
        run_logged(
            [
                str(args.python),
                "-m",
                f"{PACKAGE_NAME}.prune_layer",
                "--model",
                str(current_model),
                "--layer",
                str(layer),
                "--output-model",
                str(output_model),
                "--record",
                str(record),
                "--attention",
                args.attention,
            ],
            log_path,
            args.project_root,
        )
    if not generated_model_ready(output_model):
        raise RuntimeError(f"Pruned model export is incomplete: {output_model}")
    return read_json(record)


def ensure_training(
    args: argparse.Namespace,
    round_number: int,
    deleted_layer: int,
    pre_model: Path,
    post_model: Path,
    train_dir: Path,
) -> dict[str, Any]:
    summary_path = train_dir / "summary.json"
    quarantine_incomplete_model(post_model)
    if not generated_model_ready(post_model):
        command = [
            str(args.python),
            "-m",
            f"{PACKAGE_NAME}.train_weighted_kd",
            "--teacher-model",
            str(args.reference_model),
            "--student-model",
            str(pre_model),
            "--deleted-layer",
            str(deleted_layer),
            "--train-data",
            str(args.train_data),
            "--train-parts",
            str(args.train_parts),
            "--train-part-index",
            str(round_number),
            "--data-seed",
            str(args.data_seed),
            "--mixture-order",
            MIXTURE_ORDER_INTERLEAVED,
            "--preserve-data-order",
            "--output-dir",
            str(train_dir),
            "--export-dir",
            str(post_model),
            "--epochs",
            "1",
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.recovery_batch_size),
            "--grad-accum",
            str(args.recovery_effective_batch_size // args.recovery_batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--lora-rank",
            str(args.lora_rank),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--logging-steps",
            str(args.logging_steps),
            "--save-steps",
            str(args.save_steps),
            "--seed",
            str(args.data_seed + round_number),
            "--attention",
            args.attention,
        ]
        if args.require_math_code_per_effective_batch:
            command.extend(
                [
                    "--drop-incomplete-effective-batch",
                    "--require-effective-batch-groups",
                    "math",
                    "code",
                ]
            )
        checkpoint = latest_checkpoint(train_dir / "checkpoints")
        if checkpoint is not None:
            command.extend(["--resume-from-checkpoint", str(checkpoint)])
        run_logged(command, train_dir / "console.log", args.project_root)
    if not generated_model_ready(post_model) or not summary_path.exists():
        raise RuntimeError(f"Recovery stage is incomplete: {train_dir}")
    summary = read_json(summary_path)
    if summary.get("recovery_method") != RECOVERY_METHOD:
        raise RuntimeError(f"Unexpected recovery protocol in {summary_path}")
    if int(summary.get("train_part_index", -1)) != round_number:
        raise RuntimeError(f"Wrong training partition recorded in {summary_path}")
    if float(summary.get("epochs", 0)) != 1.0:
        raise RuntimeError(f"Recovery must train exactly one epoch: {summary_path}")
    return summary


def _final_loss(round_dir: Path) -> float | None:
    path = round_dir / "train" / "loss_summary.json"
    if not path.exists():
        return None
    return read_json(path).get("fields", {}).get("loss", {}).get("last")


def write_final_table(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    summaries: list[dict[str, Any]],
    base_layers: int,
) -> None:
    rows = [
        {
            "round": 0,
            "phase": "baseline",
            "layers": base_layers,
            "deleted_current_layer": "",
            "deleted_original_layer": "",
            "layer_pruning_risk": "",
            "train_part": "",
            "train_loss_last": "",
            "ppl": baseline["ppl"],
            "leaderboard_score": baseline["leaderboard_score"],
            "token_accuracy": baseline["token_accuracy"],
        }
    ]
    for summary in summaries:
        round_number = int(summary["round"])
        selected = summary["selected"]
        for phase in ("pre_recovery", "post_recovery"):
            metrics = summary[phase]
            rows.append(
                {
                    "round": round_number,
                    "phase": phase,
                    "layers": base_layers - round_number,
                    "deleted_current_layer": selected["current_layer"],
                    "deleted_original_layer": selected["original_layer"],
                    "layer_pruning_risk": selected["pruning_risk"],
                    "train_part": round_number if phase == "post_recovery" else "",
                    "train_loss_last": (
                        summary.get("train_loss_last", "")
                        if phase == "post_recovery"
                        else ""
                    ),
                    "ppl": metrics["ppl"],
                    "leaderboard_score": metrics["leaderboard_score"],
                    "token_accuracy": metrics["token_accuracy"],
                }
            )
    final_dir = args.run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "all_rounds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(final_dir / "all_rounds.json", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--train-parts", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=20260828)
    parser.add_argument("--test-samples-per-group", type=int, default=20)
    parser.add_argument("--min-layer", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--probe-max-answer-tokens", type=int, default=128)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--recovery-batch-size", type=int, default=1)
    parser.add_argument("--recovery-effective-batch-size", type=int, default=16)
    parser.add_argument(
        "--require-math-code-per-effective-batch",
        action="store_true",
        help="Drop the final partial window and require Math+Code in every effective batch.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--retain-pre-models", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    if not 1 <= args.rounds <= args.train_parts:
        raise ValueError("--rounds must be between 1 and --train-parts")
    if args.train_parts != 10:
        raise ValueError("This protocol requires exactly 10 training partitions")
    if args.test_samples_per_group != 20:
        raise ValueError("This protocol requires exactly 20 probe rows per group")
    if (
        args.recovery_batch_size < 1
        or args.recovery_effective_batch_size < args.recovery_batch_size
        or args.recovery_effective_batch_size % args.recovery_batch_size
    ):
        raise ValueError("Invalid recovery batch/effective-batch combination")
    config = read_json(args.reference_model / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("--reference-model is not a Qwen3-VL checkpoint")
    base_layers = int(config.get("text_config", {}).get("num_hidden_layers", 0))
    if base_layers < args.rounds + args.min_layer:
        raise ValueError("Not enough deletable language layers")
    return base_layers


def main() -> int:
    args = parse_args()
    base_layers = validate_args(args)
    probe_path = prepare_data(args)
    data_manifest_path = args.train_data.parent / "manifest.lock.json"
    data_manifest = (
        read_json(data_manifest_path)
        if data_manifest_path.exists()
        else {}
    )
    experiment_config_path = args.run_dir / "experiment_config.json"
    experiment_config = {
        "model": str(args.reference_model.resolve()),
        "model_type": "Qwen3-VL-2B-Instruct",
        "base_layers": base_layers,
        "rounds": args.rounds,
        "train_parts": args.train_parts,
        "epochs_per_round": 1,
        "data_seed": args.data_seed,
        "group_order": list(GROUP_ORDER),
        "training_sample_order": MIXTURE_ORDER_INTERLEAVED,
        "require_math_code_per_effective_batch": (
            args.require_math_code_per_effective_batch
        ),
        "loss_logging_rule": (
            "each loss row averages one proportionally interleaved "
            "logging window"
        ),
        "target_token_mix": {
            "english": 0.65,
            "chinese": 0.20,
            "math": 0.10,
            "code": 0.05,
        },
        "dataset_version": data_manifest.get("version"),
        "native_instruction_pairs_only": bool(
            data_manifest.get("require_native_instruction_pairs", False)
        ),
        "dataset_max_sequence_tokens": data_manifest.get(
            "max_sequence_tokens"
        ),
        "fixed_probe": str(probe_path.resolve()),
        "test_samples_per_group": args.test_samples_per_group,
        "leaderboard_definition": (
            "100 * macro_group_teacher_forced_token_accuracy"
        ),
        "recovery_method": RECOVERY_METHOD,
        "max_length": args.max_length,
        "probe_max_answer_tokens": args.probe_max_answer_tokens,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    }
    if experiment_config_path.exists():
        previous_config = read_json(experiment_config_path)
        previous_order = previous_config.get(
            "training_sample_order",
            MIXTURE_ORDER_GROUPED,
        )
        if previous_order != MIXTURE_ORDER_INTERLEAVED:
            raise ValueError(
                "Existing run uses a different training sample order: "
                f"existing={previous_order!r}, "
                f"requested={MIXTURE_ORDER_INTERLEAVED!r}. "
                "Use a fresh --run-dir and --model-root."
            )
    write_json(experiment_config_path, experiment_config)
    if args.prepare_only:
        print(f"[mixture-experiment] prepared: {args.run_dir}", flush=True)
        return 0

    baseline = ensure_eval(
        args,
        args.reference_model,
        probe_path,
        args.run_dir / "baseline",
    )
    current_model = args.reference_model
    summaries = []
    for round_number in range(1, args.rounds + 1):
        round_dir = args.run_dir / f"round_{round_number:02d}"
        model_dir = args.model_root / f"round_{round_number:02d}"
        pre_model = model_dir / "pre_recovery_model"
        post_model = model_dir / "post_recovery_model"
        summary_path = round_dir / "round_summary.json"
        if summary_path.exists() and generated_model_ready(post_model):
            summary = read_json(summary_path)
            summaries.append(summary)
            current_model = post_model
            if not args.retain_pre_models and pre_model.exists():
                shutil.rmtree(pre_model)
            print(f"[mixture-experiment] round {round_number:02d} complete", flush=True)
            continue

        selected_document = ensure_probe(
            args,
            current_model,
            probe_path,
            round_dir / "probe",
        )
        selected = selected_document["selected"]
        deleted_current_layer = int(selected["current_layer"])
        deletion = ensure_pruned_model(
            args,
            current_model,
            deleted_current_layer,
            pre_model,
            round_dir / "deletion.json",
            round_dir / "prune.console.log",
        )
        pre_metrics = ensure_eval(
            args,
            pre_model,
            probe_path,
            round_dir / "eval_pre_recovery",
        )
        training = ensure_training(
            args,
            round_number,
            deleted_current_layer,
            pre_model,
            post_model,
            round_dir / "train",
        )
        post_metrics = ensure_eval(
            args,
            post_model,
            probe_path,
            round_dir / "eval_post_recovery",
        )
        summary = {
            "round": round_number,
            "selected": selected,
            "deletion": deletion,
            "train_part": round_number,
            "training": training,
            "train_loss_last": _final_loss(round_dir),
            "pre_recovery": pre_metrics,
            "post_recovery": post_metrics,
        }
        write_json(summary_path, summary)
        if not args.retain_pre_models and pre_model.exists():
            shutil.rmtree(pre_model)
        summaries.append(summary)
        current_model = post_model
        write_final_table(args, baseline, summaries, base_layers)

    write_final_table(args, baseline, summaries, base_layers)
    write_json(
        args.run_dir / "final" / "experiment_complete.json",
        {
            "status": "complete",
            "rounds": args.rounds,
            "deleted_layers": args.rounds,
            "final_layers": base_layers - args.rounds,
            "final_model": str(current_model),
        },
    )
    print(f"[mixture-experiment] complete: {current_model}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
