"""Prepare immutable manifests and audit metadata for a Highway pruning run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from sft_scripts.utils.sft_io import get_assistant_json, load_sft_samples

from .utils.field_weights import weight_config
from .utils.io_utils import write_json
from .utils.recovery_config import RECOVERY_METHOD


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_subsets(
    population_size: int,
    repeats: int,
    samples_per_repeat: int,
    seed: int,
) -> list[list[int]]:
    if samples_per_repeat > population_size:
        raise ValueError("samples_per_repeat exceeds validation size")
    generator = random.Random(seed)
    population = list(range(population_size))
    return [
        generator.sample(population, samples_per_repeat)
        for _ in range(repeats)
    ]


def build_manifest_rows(
    val_path: Path,
    samples: list[dict[str, Any]],
    subsets: list[list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_id, indices in enumerate(subsets, start=1):
        for position, source_index in enumerate(indices, start=1):
            sample = samples[source_index]
            rows.append(
                {
                    "probe_row_id": len(rows),
                    "repeat_id": repeat_id,
                    "position_in_repeat": position,
                    "source_json": str(val_path),
                    "index_in_source": source_index,
                    "image": sample.get("image"),
                    "ground_truth": get_assistant_json(sample),
                }
            )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "probe_10x10.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "probe_10x10.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "probe_row_id",
            "repeat_id",
            "position_in_repeat",
            "index_in_source",
            "image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in fieldnames}
            for row in rows
        )


def environment_record() -> dict[str, Any]:
    import torch
    import transformers

    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_query.stdout.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples-per-repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--recovery-epochs", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    if args.recovery_epochs <= 0:
        raise ValueError("--recovery-epochs must be positive")
    train_samples = load_sft_samples(args.train_data)
    val_samples = load_sft_samples(args.val_data)
    subsets = sample_subsets(
        len(val_samples),
        args.repeats,
        args.samples_per_repeat,
        args.seed,
    )
    rows = build_manifest_rows(args.val_data.resolve(), val_samples, subsets)
    write_manifest(args.run_dir / "probe", rows)
    write_json(
        args.run_dir / "dataset_fingerprints.json",
        {
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "train": {
                "path": str(args.train_data.resolve()),
                "samples": len(train_samples),
                "sha256": sha256_file(args.train_data),
            },
            "validation": {
                "path": str(args.val_data.resolve()),
                "samples": len(val_samples),
                "sha256": sha256_file(args.val_data),
            },
            "probe": {
                "seed": args.seed,
                "repeats": args.repeats,
                "samples_per_repeat": args.samples_per_repeat,
                "rows": len(rows),
                "subsets": subsets,
            },
        },
    )
    write_json(
        args.run_dir / "resolved_config.json",
        {
            "reference_model": str(args.model.resolve()),
            "train_data": str(args.train_data.resolve()),
            "val_data": str(args.val_data.resolve()),
            "rounds": args.rounds,
            "recovery_epochs": args.recovery_epochs,
            "probe_repeats": args.repeats,
            "probe_samples_per_repeat": args.samples_per_repeat,
            "probe_seed": args.seed,
            "ranking": "minimax(max(relative_hard_regret, normalized_js))",
            "field_weights": weight_config(),
            "recovery_method": RECOVERY_METHOD,
            "recovery_teacher": "current_model_before_deletion",
            "recovery_train_scope": "student_language_layer_i_minus_1_only",
            "recovery_loss": "field_weighted_KL(q_teacher_i||q_student_i_minus_1)",
        },
    )
    environment = environment_record()
    write_json(args.run_dir / "environment.json", environment)
    (args.run_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(rows)} fixed probe rows from "
        f"{len(val_samples)} validation samples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
