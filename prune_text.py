"""Rank one-layer deletions on the fixed 4x20 ordered text probe."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch

from .utils.data import HighwayDataset
from .utils.io_utils import load_layer_state, write_csv, write_json
from .utils.model_ops import bypass_layer, get_layers, load_model, load_processor
from .utils.text_metrics import (
    TextMetricAccumulator,
    prepare_scoring_sample,
    score_hidden_states,
)


def candidate_layers(layer_count: int, min_layer: int, requested: str) -> list[int]:
    available = list(range(min_layer, layer_count))
    if not requested:
        return available
    selected = [int(value) for value in requested.split(",") if value.strip()]
    invalid = [value for value in selected if value not in available]
    if invalid:
        raise ValueError(f"Invalid candidate layers: {invalid}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--min-layer", type=int, default=3)
    parser.add_argument("--candidate-layers", default="")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    return parser.parse_args()


def _restore_progress(
    progress_path: Path,
    signature: dict[str, Any],
    candidates: list[int],
) -> tuple[int, TextMetricAccumulator, dict[int, TextMetricAccumulator]]:
    if not progress_path.exists():
        return (
            0,
            TextMetricAccumulator(),
            {layer: TextMetricAccumulator() for layer in candidates},
        )
    import json

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("signature") != signature:
        raise RuntimeError(
            f"Probe progress belongs to a different configuration: {progress_path}"
        )
    candidate_states = progress.get("candidates", {})
    return (
        int(progress.get("next_sample", 0)),
        TextMetricAccumulator.from_state_dict(progress.get("baseline", {})),
        {
            layer: TextMetricAccumulator.from_state_dict(
                candidate_states.get(str(layer), {})
            )
            for layer in candidates
        },
    )


def main() -> int:
    args = parse_args()
    if args.max_length < 2 or args.logit_chunk_size < 1:
        raise ValueError("--max-length must be >= 2 and chunk size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    processor = load_processor(args.model)
    dataset = HighwayDataset(args.data, processor, max_length=args.max_length)
    model = load_model(args.model, device, dtype, args.attention)
    model.eval()
    state = load_layer_state(args.model, len(get_layers(model)))
    original_ids = [int(value) for value in state["original_layer_ids"]]
    candidates = candidate_layers(
        len(original_ids),
        args.min_layer,
        args.candidate_layers,
    )
    if not candidates:
        raise ValueError("No candidate layers remain")

    signature = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "samples": len(dataset),
        "candidates": candidates,
        "max_length": args.max_length,
        "max_answer_tokens": args.max_answer_tokens,
        "logit_chunk_size": args.logit_chunk_size,
    }
    progress_path = args.output_dir / "progress.json"
    next_sample, baseline, by_candidate = _restore_progress(
        progress_path,
        signature,
        candidates,
    )
    started = time.monotonic()
    for sample_index in range(next_sample, len(dataset)):
        prepared = dataset[sample_index]
        language_inputs, positions, targets = prepare_scoring_sample(
            model,
            prepared,
            device,
            args.max_answer_tokens,
        )
        group = str(dataset.samples[sample_index].get("group") or "unknown")

        baseline_hidden = model.model.language_model(
            **language_inputs.as_kwargs()
        ).last_hidden_state
        baseline.update(
            group,
            *score_hidden_states(
                model,
                baseline_hidden,
                positions,
                targets,
                args.logit_chunk_size,
            ),
        )
        del baseline_hidden

        for layer_index in candidates:
            with bypass_layer(model, layer_index):
                candidate_hidden = model.model.language_model(
                    **language_inputs.as_kwargs()
                ).last_hidden_state
            by_candidate[layer_index].update(
                group,
                *score_hidden_states(
                    model,
                    candidate_hidden,
                    positions,
                    targets,
                    args.logit_chunk_size,
                ),
            )
            del candidate_hidden

        write_json(
            progress_path,
            {
                "signature": signature,
                "next_sample": sample_index + 1,
                "baseline": baseline.state_dict(),
                "candidates": {
                    str(layer): accumulator.state_dict()
                    for layer, accumulator in by_candidate.items()
                },
            },
        )
        elapsed = time.monotonic() - started
        print(
            f"[prune-text] sample {sample_index + 1}/{len(dataset)}; "
            f"candidates={len(candidates)}; elapsed={elapsed:.1f}s",
            flush=True,
        )
        del prepared, language_inputs, positions, targets

    baseline_summary = baseline.summary()
    write_json(args.output_dir / "baseline_metrics.json", baseline_summary)
    rows = []
    baseline_nll = float(baseline_summary["mean_nll"])
    baseline_score = float(baseline_summary["leaderboard_score"])
    for layer_index in candidates:
        candidate_summary = by_candidate[layer_index].summary()
        candidate_nll = float(candidate_summary["mean_nll"])
        candidate_score = float(candidate_summary["leaderboard_score"])
        relative_nll_increase = max(0.0, candidate_nll - baseline_nll) / max(
            abs(baseline_nll), 1e-12
        )
        relative_score_drop = max(0.0, baseline_score - candidate_score) / max(
            abs(baseline_score), 1e-12
        )
        row = {
            "current_layer": layer_index,
            "original_layer": original_ids[layer_index],
            "baseline_ppl": baseline_summary["ppl"],
            "candidate_ppl": candidate_summary["ppl"],
            "baseline_leaderboard_score": baseline_score,
            "candidate_leaderboard_score": candidate_score,
            "relative_nll_increase": relative_nll_increase,
            "relative_leaderboard_drop": relative_score_drop,
            "pruning_risk": max(relative_nll_increase, relative_score_drop),
            "candidate_metrics": candidate_summary,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["pruning_risk"]),
            float(row["relative_nll_increase"]),
            float(row["relative_leaderboard_drop"]),
            int(row["original_layer"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    write_csv(
        args.output_dir / "layer_leaderboard.csv",
        [
            {key: value for key, value in row.items() if key != "candidate_metrics"}
            for row in rows
        ],
    )
    write_json(args.output_dir / "layer_leaderboard.json", rows)
    write_json(
        args.output_dir / "selected_layer.json",
        {
            "ranking": (
                "pruning_risk, relative_nll_increase, "
                "relative_leaderboard_drop, original_layer"
            ),
            "selected": rows[0],
            "candidate_count": len(rows),
            "original_layer_ids": original_ids,
            "probe_samples": len(dataset),
        },
    )
    print(
        f"[prune-text] selected current={rows[0]['current_layer']} "
        f"original={rows[0]['original_layer']} "
        f"risk={rows[0]['pruning_risk']:.6f}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
