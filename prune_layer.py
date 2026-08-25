"""Physically delete one selected language layer and save an auditable model."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .utils.io_utils import load_layer_state, write_json
from .utils.model_ops import (
    get_layers,
    load_model,
    load_processor,
    physical_delete_layer,
    save_pruned_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_model.exists():
        raise FileExistsError(f"Output model already exists: {args.output_model}")
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    processor = load_processor(args.model)
    model = load_model(args.model, device, dtype, args.attention)
    state = load_layer_state(args.model, len(get_layers(model)))
    original_ids = [int(value) for value in state["original_layer_ids"]]
    deleted_original_layer, new_ids = physical_delete_layer(
        model,
        args.layer,
        original_ids,
    )
    round_number = len(state.get("deleted_original_layers", [])) + 1
    deletion_record = {
        "round": round_number,
        "source_model": str(args.model.resolve()),
        "output_model": str(args.output_model.resolve()),
        "deleted_current_layer": args.layer,
        "deleted_original_layer": deleted_original_layer,
        "layers_before": len(original_ids),
        "layers_after": len(new_ids),
    }
    new_state = {
        **state,
        "original_layer_ids": new_ids,
        "deleted_original_layers": [
            *state.get("deleted_original_layers", []),
            deleted_original_layer,
        ],
        "rounds": [*state.get("rounds", []), deletion_record],
    }
    save_pruned_model(model, processor, args.output_model, new_state)
    write_json(args.record, deletion_record)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"Deleted current layer {args.layer} (original {deleted_original_layer}); "
        f"saved {len(new_ids)}-layer model to {args.output_model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
