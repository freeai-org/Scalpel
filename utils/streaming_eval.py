"""Reusable model loading, streaming generation, and timing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from threading import Thread
import time
from typing import Any

import torch
from transformers import TextIteratorStreamer

from .model_ops import get_layers, load_model, load_processor


@dataclass(slots=True, frozen=True)
class StreamingEvalConfig:
    device: torch.device
    dtype: torch.dtype
    attention: str = "sdpa"
    max_new_tokens: int = 1024
    temperature: float = 1.0
    warmup: bool = True


@dataclass(slots=True, frozen=True)
class GenerationResult:
    label: str
    layers: int
    parameters: int
    load_seconds: float
    first_text_seconds: float
    generation_seconds: float
    output_tokens: int
    tokens_per_second: float
    decode_tokens_per_second: float
    peak_cuda_mib: float


def _build_inputs(processor: Any, prompt: str, device: torch.device) -> Any:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    chat_template = getattr(processor, "chat_template", "")
    if isinstance(chat_template, str) and "enable_thinking" in chat_template:
        template_kwargs["enable_thinking"] = False
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        template_kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    return inputs.to(device)


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def _warm_up(model: Any, inputs: Any, pad_token_id: int) -> None:
    model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )


def stream_model(
    label: str,
    model_path: Path,
    prompt: str,
    config: StreamingEvalConfig,
) -> GenerationResult:
    """Load one model, stream its answer, report speed, then release it."""
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    load_started = time.perf_counter()
    processor = load_processor(model_path)
    model = load_model(
        model_path, config.device, config.dtype, config.attention
    )
    model.eval()
    model.config.use_cache = True
    load_seconds = time.perf_counter() - load_started
    layers = len(get_layers(model))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    inputs = _build_inputs(processor, prompt, config.device)
    input_width = int(inputs.input_ids.shape[-1])
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    if config.warmup:
        _warm_up(model, inputs, int(pad_token_id))
    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
        torch.cuda.reset_peak_memory_stats(config.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        timeout=120.0,
    )
    outputs: list[torch.Tensor] = []
    errors: list[BaseException] = []
    worker_finished_at: list[float] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                outputs.append(
                    model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=True,
                        temperature=config.temperature,
                        use_cache=True,
                        pad_token_id=int(pad_token_id),
                    )
                )
        except BaseException as exc:  # propagate worker failures to the caller
            errors.append(exc)
            streamer.end()
        finally:
            worker_finished_at.append(time.perf_counter())

    print(f"\n{'=' * 24} {label} ({layers} layers) {'=' * 24}")
    started = time.perf_counter()
    first_text_at: float | None = None
    worker = Thread(target=generate, daemon=True)
    worker.start()
    for text in streamer:
        if text and first_text_at is None:
            first_text_at = time.perf_counter()
        print(text, end="", flush=True)
    worker.join()
    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
    print()

    if errors:
        del worker, generate, streamer, outputs, inputs, model, processor
        _release_cuda_cache()
        raise RuntimeError(f"{label} generation failed") from errors[0]
    if not outputs:
        del worker, generate, streamer, inputs, model, processor
        _release_cuda_cache()
        raise RuntimeError(f"{label} produced no output tensor")

    output_tokens = int(outputs[0].shape[-1] - input_width)
    generation_seconds = worker_finished_at[0] - started
    first_text_seconds = (
        first_text_at - started if first_text_at is not None else generation_seconds
    )
    decode_seconds = max(generation_seconds - first_text_seconds, 1e-12)
    result = GenerationResult(
        label=label,
        layers=layers,
        parameters=parameters,
        load_seconds=load_seconds,
        first_text_seconds=first_text_seconds,
        generation_seconds=generation_seconds,
        output_tokens=output_tokens,
        tokens_per_second=output_tokens / max(generation_seconds, 1e-12),
        decode_tokens_per_second=max(output_tokens - 1, 0) / decode_seconds,
        peak_cuda_mib=(
            torch.cuda.max_memory_allocated(config.device) / 2**20
            if config.device.type == "cuda"
            else 0.0
        ),
    )
    print(
        f"[{label} speed] load={result.load_seconds:.2f}s | "
        f"first_text={result.first_text_seconds:.3f}s | "
        f"generation={result.generation_seconds:.2f}s | "
        f"tokens={result.output_tokens} | "
        f"tokens/s={result.tokens_per_second:.2f} | "
        f"decode_tokens/s={result.decode_tokens_per_second:.2f} | "
        f"peak_cuda={result.peak_cuda_mib:.0f} MiB",
        flush=True,
    )
    del worker, generate, streamer, outputs, inputs, model, processor
    _release_cuda_cache()
    return result
