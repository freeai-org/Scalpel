"""Stream one prompt through the baseline and Round-7 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__:
    from .utils.streaming_eval import StreamingEvalConfig, stream_model
else:
    # Also support `python eval.py` from this directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Scalpel.utils.streaming_eval import StreamingEvalConfig, stream_model


SCALPEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCALPEL_ROOT.parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "B-RVPO/models/vl_local"
DEFAULT_ROUND7 = (
    SCALPEL_ROOT
    / "results/qwen3vl2b_mixture_prune10_bs2_ga16"
    / "models/round_07/post_recovery_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and compare Baseline versus Round 7 on one prompt."
    )
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--round7-model", type=Path, default=DEFAULT_ROUND7)
    parser.add_argument("--prompt", help="Question to send to both models")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one unmeasured token before timing each model.",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


def read_prompt(value: str | None) -> str:
    if value is not None:
        return value
    print("Prompt: ", end="", flush=True)
    return sys.stdin.readline().strip()


def main() -> int:
    args = parse_args()
    prompt = read_prompt(args.prompt)
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In WSL, run `wsl --shutdown` from Windows "
            "PowerShell, reopen WSL, and retry."
        )
    config = StreamingEvalConfig(
        device=device,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        attention=args.attention,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
    )
    baseline = stream_model(
        "Baseline", args.baseline_model.resolve(), prompt, config
    )
    round7 = stream_model("Round 7", args.round7_model.resolve(), prompt, config)

    print(f"\n{'=' * 28} Speed comparison {'=' * 28}")
    print(
        f"Baseline: {baseline.layers} layers, "
        f"{baseline.tokens_per_second:.2f} tokens/s"
    )
    print(
        f"Round 7 : {round7.layers} layers, "
        f"{round7.tokens_per_second:.2f} tokens/s"
    )
    speedup = round7.tokens_per_second / max(
        baseline.tokens_per_second, 1e-12
    )
    print(f"Round 7 throughput speedup: {speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
