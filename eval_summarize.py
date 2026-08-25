"""Summarize atomic field metrics in a generation-evaluation directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .utils.io_utils import write_csv, write_json
from .utils.task_metrics import summarize_predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.eval_dir / "predictions.jsonl"
    summary = summarize_predictions(read_jsonl(predictions_path))
    write_json(args.eval_dir / "atomic_metrics.json", summary)
    write_csv(
        args.eval_dir / "per_field.csv",
        [
            {
                "field": name,
                **values,
            }
            for name, values in summary["fields"].items()
        ],
    )
    print(
        f"Atomic macro accuracy: {summary['macro_field_accuracy']:.6f} "
        f"({summary['samples']} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
