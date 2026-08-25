"""Probe and rank deletable layers with generation accuracy and normalized JS."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from sft_scripts.utils.sft_io import (
    get_conversation_text,
    load_sft_samples,
    resolve_image_path,
)
from sft_scripts.utils.sft_json import extract_json_from_text

from .utils.data import HighwayDataset, move_sample, supervised_positions
from .utils.io_utils import append_jsonl, load_layer_state, write_csv, write_json
from .utils.metrics import normalized_jensen_shannon
from .utils.model_ops import (
    bypass_layer,
    get_layers,
    load_model,
    load_processor,
    prepare_language_inputs,
    temporary_delete_layer,
)
from .utils.task_metrics import hard_regret, pruning_risk, summarize_predictions


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_generation_inputs(
    processor: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> Any:
    message_batch = []
    for sample in samples:
        user_text, _ = get_conversation_text(sample)
        image_path = resolve_image_path(str(sample["image"]))
        message_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        )
    messages = message_batch[0] if len(message_batch) == 1 else message_batch
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "processor_kwargs": {"padding": "longest"},
    }
    chat_template = getattr(processor, "chat_template", "")
    if isinstance(chat_template, str) and "enable_thinking" in chat_template:
        template_kwargs["enable_thinking"] = False
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        template_kwargs.pop("enable_thinking")
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    return inputs.to(device)


@torch.inference_mode()
def generate_with_bypass(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    layer_index: int,
    device: torch.device,
    max_new_tokens: int,
) -> list[tuple[str, dict[str, Any] | None]]:
    inputs = build_generation_inputs(processor, samples, device)
    previous_use_cache = bool(model.config.use_cache)
    model.config.use_cache = True
    try:
        with temporary_delete_layer(model, layer_index):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        model.config.use_cache = previous_use_cache
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_texts = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [
        (raw_text, extract_json_from_text(raw_text))
        for raw_text in raw_texts
    ]


@torch.inference_mode()
def sample_normalized_js(
    reference_model: Any,
    candidate_model: Any,
    dataset: HighwayDataset,
    sample_index: int,
    layer_index: int,
    device: torch.device,
    chunk_size: int,
) -> float:
    sample = move_sample(dataset[sample_index], device)
    positions, _ = supervised_positions(
        sample.labels,
        max_tokens=max(1, int((sample.labels[:, 1:] != -100).sum().item())),
    )
    token_weights = sample.loss_weights[:, 1:][:, positions]

    reference_inputs = prepare_language_inputs(
        reference_model,
        sample.model_inputs,
    )
    if reference_model is candidate_model:
        candidate_inputs = reference_inputs
    else:
        candidate_inputs = prepare_language_inputs(
            candidate_model,
            sample.model_inputs,
        )
    reference_hidden = reference_model.model.language_model(
        **reference_inputs.as_kwargs()
    ).last_hidden_state
    with bypass_layer(candidate_model, layer_index):
        candidate_hidden = candidate_model.model.language_model(
            **candidate_inputs.as_kwargs()
        ).last_hidden_state

    weighted_js_sum = 0.0
    weight_sum = 0.0
    for start in range(0, positions.numel(), chunk_size):
        chunk_positions = positions[start : start + chunk_size]
        chunk_weights = token_weights[:, start : start + chunk_size]
        reference_logits = reference_model.lm_head(
            reference_hidden[:, chunk_positions, :]
        )
        candidate_logits = candidate_model.lm_head(
            candidate_hidden[:, chunk_positions, :]
        )
        chunk_js = normalized_jensen_shannon(
            reference_logits,
            candidate_logits,
            chunk_weights,
        )
        chunk_weight = float(chunk_weights.sum().item())
        weighted_js_sum += float(chunk_js.item()) * chunk_weight
        weight_sum += chunk_weight
        del reference_logits, candidate_logits
    return weighted_js_sum / max(weight_sum, 1e-12)


def repeat_summaries(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if len(manifest_rows) != len(prediction_rows):
        raise ValueError("manifest and predictions have different row counts")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for manifest, prediction in zip(manifest_rows, prediction_rows):
        grouped[int(manifest["repeat_id"])].append(prediction)
    return {
        repeat_id: summarize_predictions(rows)
        for repeat_id, rows in sorted(grouped.items())
    }


def load_reference_probe_predictions(
    manifest_rows: list[dict[str, Any]],
    baseline_predictions_path: Path,
) -> list[dict[str, Any]]:
    by_index = {
        int(row["index"]): row
        for row in read_jsonl(baseline_predictions_path)
    }
    missing = sorted(
        {
            int(row["index_in_source"])
            for row in manifest_rows
            if int(row["index_in_source"]) not in by_index
        }
    )
    if missing:
        raise ValueError(
            f"Baseline predictions are missing {len(missing)} probe indices"
        )
    return [
        by_index[int(manifest["index_in_source"])]
        for manifest in manifest_rows
    ]


def candidate_layers(
    layer_count: int,
    min_layer: int,
    requested: str,
) -> list[int]:
    available = list(range(min_layer, layer_count))
    if not requested:
        return available
    selected = [int(value) for value in requested.split(",") if value.strip()]
    invalid = [value for value in selected if value not in available]
    if invalid:
        raise ValueError(f"Invalid candidate layers: {invalid}")
    return selected


def write_candidate_predictions(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    layer_index: int,
    original_layer_id: int,
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if output_path.exists():
        rows = read_jsonl(output_path)
        if len(rows) == len(manifest_rows):
            return rows
        output_path.unlink()

    rows: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        try:
            outputs = generate_with_bypass(
                model,
                processor,
                batch_samples,
                layer_index,
                device,
                max_new_tokens,
            )
        except Exception:
            if len(batch_samples) == 1:
                raise
            print(
                f"[prune-probe] layer={layer_index} batch generation failed; "
                "retrying one sample at a time"
            )
            outputs = []
            for sample in batch_samples:
                outputs.extend(
                    generate_with_bypass(
                        model,
                        processor,
                        [sample],
                        layer_index,
                        device,
                        max_new_tokens,
                    )
                )
        for offset, (raw_prediction, prediction) in enumerate(outputs):
            manifest = manifest_rows[start + offset]
            row = {
                **manifest,
                "current_layer": layer_index,
                "original_layer": original_layer_id,
                "ground_truth": manifest["ground_truth"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
            }
            append_jsonl(output_path, row)
            rows.append(row)
        print(
            f"[prune-probe] layer={layer_index} generated "
            f"{min(start + batch_size, len(samples))}/{len(samples)}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--min-layer", type=int, default=3)
    parser.add_argument("--candidate-layers", default="")
    parser.add_argument("--max-probe-rows", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--js-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    manifest_rows = read_jsonl(args.probe_manifest)
    if args.max_probe_rows:
        manifest_rows = manifest_rows[: args.max_probe_rows]
    all_val_samples = load_sft_samples(args.val_data)
    samples = [
        all_val_samples[int(row["index_in_source"])]
        for row in manifest_rows
    ]
    reference_predictions = load_reference_probe_predictions(
        manifest_rows,
        args.baseline_predictions,
    )
    reference_repeats = repeat_summaries(
        manifest_rows,
        reference_predictions,
    )

    processor = load_processor(args.candidate_model)
    candidate_model = load_model(
        args.candidate_model,
        device,
        dtype,
        args.attention,
    )
    same_model = args.reference_model.resolve() == args.candidate_model.resolve()
    reference_model = (
        candidate_model
        if same_model
        else load_model(
            args.reference_model,
            device,
            dtype,
            args.attention,
        )
    )
    candidate_model.eval()
    reference_model.eval()
    state = load_layer_state(
        args.candidate_model,
        len(get_layers(candidate_model)),
    )
    original_ids = [int(value) for value in state["original_layer_ids"]]
    candidates = candidate_layers(
        len(original_ids),
        args.min_layer,
        args.candidate_layers,
    )
    dataset = HighwayDataset(args.val_data, processor, max_length=2560)
    layer_rows = []
    repeat_rows = []
    soft_rows_path = args.output_dir / "sample_soft_metrics.jsonl"
    if soft_rows_path.exists():
        soft_rows_path.unlink()

    for candidate_position, layer_index in enumerate(candidates, start=1):
        started = time.monotonic()
        original_layer_id = original_ids[layer_index]
        layer_dir = (
            args.output_dir
            / "layers"
            / f"current_{layer_index:02d}_original_{original_layer_id:02d}"
        )
        layer_dir.mkdir(parents=True, exist_ok=True)
        predictions = write_candidate_predictions(
            model=candidate_model,
            processor=processor,
            samples=samples,
            manifest_rows=manifest_rows,
            layer_index=layer_index,
            original_layer_id=original_layer_id,
            output_path=layer_dir / "predictions.jsonl",
            batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        candidate_repeats = repeat_summaries(manifest_rows, predictions)

        js_by_row = []
        js_cache: dict[int, float] = {}
        for row_number, manifest in enumerate(manifest_rows, start=1):
            source_index = int(manifest["index_in_source"])
            if source_index not in js_cache:
                js_cache[source_index] = sample_normalized_js(
                    reference_model,
                    candidate_model,
                    dataset,
                    source_index,
                    layer_index,
                    device,
                    args.js_chunk_size,
                )
            normalized_js = js_cache[source_index]
            js_by_row.append(normalized_js)
            append_jsonl(
                soft_rows_path,
                {
                    "probe_row_id": manifest["probe_row_id"],
                    "repeat_id": manifest["repeat_id"],
                    "source_index": source_index,
                    "current_layer": layer_index,
                    "original_layer": original_layer_id,
                    "normalized_js": normalized_js,
                },
            )
            print(
                f"[prune-probe] layer={layer_index} soft "
                f"{row_number}/{len(manifest_rows)}"
            )

        js_by_repeat: dict[int, list[float]] = defaultdict(list)
        for manifest, value in zip(manifest_rows, js_by_row):
            js_by_repeat[int(manifest["repeat_id"])].append(value)
        candidate_repeat_metrics = []
        for repeat_id in sorted(candidate_repeats):
            reference_accuracy = float(
                reference_repeats[repeat_id]["macro_field_accuracy"]
            )
            candidate_accuracy = float(
                candidate_repeats[repeat_id]["macro_field_accuracy"]
            )
            regret = hard_regret(reference_accuracy, candidate_accuracy)
            normalized_js = _mean(js_by_repeat[repeat_id])
            risk = pruning_risk(regret, normalized_js)
            row = {
                "repeat_id": repeat_id,
                "current_layer": layer_index,
                "original_layer": original_layer_id,
                "reference_macro_accuracy": reference_accuracy,
                "candidate_macro_accuracy": candidate_accuracy,
                "relative_hard_regret": regret,
                "normalized_js": normalized_js,
                "pruning_risk": risk,
            }
            candidate_repeat_metrics.append(row)
            repeat_rows.append(row)
        write_json(
            layer_dir / "metrics.json",
            {
                "candidate": summarize_predictions(predictions),
                "reference_repeats": reference_repeats,
                "repeat_metrics": candidate_repeat_metrics,
            },
        )
        summary = {
            "current_layer": layer_index,
            "original_layer": original_layer_id,
            "mean_reference_macro_accuracy": _mean(
                row["reference_macro_accuracy"]
                for row in candidate_repeat_metrics
            ),
            "mean_candidate_macro_accuracy": _mean(
                row["candidate_macro_accuracy"]
                for row in candidate_repeat_metrics
            ),
            "mean_relative_hard_regret": _mean(
                row["relative_hard_regret"]
                for row in candidate_repeat_metrics
            ),
            "std_relative_hard_regret": _std(
                row["relative_hard_regret"]
                for row in candidate_repeat_metrics
            ),
            "mean_normalized_js": _mean(
                row["normalized_js"]
                for row in candidate_repeat_metrics
            ),
            "std_normalized_js": _std(
                row["normalized_js"]
                for row in candidate_repeat_metrics
            ),
            "mean_pruning_risk": _mean(
                row["pruning_risk"]
                for row in candidate_repeat_metrics
            ),
            "std_pruning_risk": _std(
                row["pruning_risk"]
                for row in candidate_repeat_metrics
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        layer_rows.append(summary)
        write_json(layer_dir / "summary.json", summary)
        write_csv(args.output_dir / "layer_metrics.partial.csv", layer_rows)
        print(
            f"[prune-probe] candidate {candidate_position}/{len(candidates)} "
            f"done: current={layer_index}, original={original_layer_id}, "
            f"risk={summary['mean_pruning_risk']:.6f}"
        )

    layer_rows.sort(
        key=lambda row: (
            float(row["mean_pruning_risk"]),
            float(row["mean_relative_hard_regret"]),
            float(row["mean_normalized_js"]),
            int(row["original_layer"]),
        )
    )
    for rank, row in enumerate(layer_rows, start=1):
        row["rank"] = rank
    write_csv(args.output_dir / "layer_metrics.csv", layer_rows)
    write_csv(args.output_dir / "repeat_metrics.csv", repeat_rows)
    write_json(
        args.output_dir / "selected_layer.json",
        {
            "ranking": (
                "mean_pruning_risk, mean_relative_hard_regret, "
                "mean_normalized_js, original_layer"
            ),
            "selected": layer_rows[0],
            "candidate_count": len(layer_rows),
            "original_layer_ids": original_ids,
        },
    )
    if reference_model is not candidate_model:
        del reference_model
    del candidate_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[prune-probe] selected: {layer_rows[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
