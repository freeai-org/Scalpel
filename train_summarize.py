"""Summarize an append-only boundary-recovery training log."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from .utils.io_utils import write_json


COMPONENT_FIELDS = ("boundary_weighted_kl",)


def read_loss_rows(path: Path) -> list[dict[str, Any]]:
    """Read a fixed-length snapshot of an append-only Trainer log."""

    rows: list[dict[str, Any]] = []
    snapshot_size = path.stat().st_size
    descriptor = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        remaining = snapshot_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks).decode("utf-8")
    for line_number, line in enumerate(snapshot.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "step" not in row:
            raise ValueError(f"{path}:{line_number} has no step")
        rows.append(row)
    if not rows:
        raise ValueError(f"No loss records found in {path}")
    return rows


def latest_record_per_step(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Use the latest replay of a step while reporting duplicated step IDs."""

    by_step: dict[int, dict[str, Any]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        step = int(row["step"])
        by_step[step] = row
        counts[step] = counts.get(step, 0) + 1
    duplicate_steps = sorted(step for step, count in counts.items() if count > 1)
    return [by_step[step] for step in sorted(by_step)], duplicate_steps


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Return the ordinary-least-squares slope, or zero for one point."""

    if len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return 0.0
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    ) / denominator


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build global and first/last-window statistics from loss records."""

    loss_fields = (
        "loss",
        *COMPONENT_FIELDS,
        "grad_norm",
        "learning_rate",
    )
    loss_rows = [
        row
        for row in rows
        if "epoch" in row and all(field in row for field in loss_fields)
    ]
    if not loss_rows:
        raise ValueError("No complete loss-component records found")
    unique_rows, duplicate_steps = latest_record_per_step(loss_rows)
    window_size = max(1, math.ceil(len(unique_rows) * 0.1))
    first_window = unique_rows[:window_size]
    last_window = unique_rows[-window_size:]
    steps = [float(row["step"]) for row in unique_rows]

    fields: dict[str, dict[str, float | None]] = {}
    nonfinite: dict[str, int] = {}
    for field in loss_fields:
        values = [float(row[field]) for row in unique_rows]
        nonfinite[field] = sum(not math.isfinite(value) for value in values)
        finite_pairs = [
            (step, value)
            for step, value in zip(steps, values)
            if math.isfinite(value)
        ]
        finite_steps = [pair[0] for pair in finite_pairs]
        finite_values = [pair[1] for pair in finite_pairs]
        if not finite_values:
            continue
        first_values = [
            float(row[field])
            for row in first_window
            if math.isfinite(float(row[field]))
        ]
        last_values = [
            float(row[field])
            for row in last_window
            if math.isfinite(float(row[field]))
        ]
        first_window_mean = (
            statistics.fmean(first_values) if first_values else None
        )
        last_window_mean = (
            statistics.fmean(last_values) if last_values else None
        )
        fields[field] = {
            "first": finite_values[0],
            "last": finite_values[-1],
            "min": min(finite_values),
            "max": max(finite_values),
            "mean": statistics.fmean(finite_values),
            "first_window_mean": first_window_mean,
            "last_window_mean": last_window_mean,
            "last_minus_first_window": (
                last_window_mean - first_window_mean
                if first_window_mean is not None and last_window_mean is not None
                else None
            ),
            "slope_per_step": linear_slope(finite_steps, finite_values),
        }

    decomposition_errors = [
        abs(
            float(row["loss"])
            - sum(float(row[field]) for field in COMPONENT_FIELDS)
        )
        for row in unique_rows
    ]
    finite_decomposition_errors = [
        value for value in decomposition_errors if math.isfinite(value)
    ]
    return {
        "raw_records": len(rows),
        "skipped_non_loss_records": len(rows) - len(loss_rows),
        "loss_component_records": len(loss_rows),
        "unique_steps": len(unique_rows),
        "duplicate_steps": duplicate_steps,
        "first_step": int(unique_rows[0]["step"]),
        "last_step": int(unique_rows[-1]["step"]),
        "first_epoch": float(unique_rows[0]["epoch"]),
        "last_epoch": float(unique_rows[-1]["epoch"]),
        "window_definition": "first and last ceil(10%) unique step records",
        "window_records": window_size,
        "loss_components": list(COMPONENT_FIELDS),
        "fields": fields,
        "nonfinite_counts": nonfinite,
        "max_loss_decomposition_abs_error": (
            max(finite_decomposition_errors)
            if finite_decomposition_errors
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.loss_log.with_name("loss_summary.json")
    summary = summarize_rows(read_loss_rows(args.loss_log))
    write_json(output, summary)
    print(f"Training loss summary saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
