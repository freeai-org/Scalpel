"""Field-aware token weights shared by probe metrics and recovery training.

The constants intentionally match the project's weighted JSON SFT recipe.  Keeping
the implementation here avoids importing a command-line training script from the
Highway package and gives the probe and distillation exactly the same weighting.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch


FORMAT_WEIGHT = 0.5
DEFAULT_WEIGHT = 1.0
ENV_WEIGHT = 1.0
POSTURE_ACTION_WEIGHT = 3.0
CAT_PRESENCE_WEIGHT = 2.0


def _set_span_weight(
    char_weights: list[float],
    start: int,
    end: int,
    weight: float,
) -> None:
    for index in range(max(0, start), min(len(char_weights), end)):
        char_weights[index] = max(char_weights[index], weight)


def _set_regex_weight(
    text: str,
    char_weights: list[float],
    pattern: str,
    weight: float,
) -> None:
    for match in re.finditer(pattern, text):
        _set_span_weight(char_weights, match.start(), match.end(), weight)


def build_char_weights(assistant_text: str) -> list[float]:
    """Return the established 0.5/1/2/3 character-level field weights."""

    char_weights = [DEFAULT_WEIGHT] * len(assistant_text)
    for index, character in enumerate(assistant_text):
        if character in '{}[]:,"':
            char_weights[index] = FORMAT_WEIGHT

    _set_regex_weight(
        assistant_text,
        char_weights,
        r'"cats_visible"\s*:\s*(?:-?\d+|"[^"]*")',
        CAT_PRESENCE_WEIGHT,
    )
    for pattern in [
        r'"action"\s*:\s*"[^"]*"',
        r'"overall_body"\s*:\s*"[^"]*"',
        r'"fur_state"\s*:\s*"[^"]*"',
        r'"ears"\s*:\s*\{[^{}]*\}',
        r'"tail"\s*:\s*\{[^{}]*\}',
        r'"face"\s*:\s*\{[^{}]*\}',
        r'"eyelid"\s*:\s*"[^"]*"',
        r'"mouth"\s*:\s*"[^"]*"',
    ]:
        _set_regex_weight(
            assistant_text,
            char_weights,
            pattern,
            POSTURE_ACTION_WEIGHT,
        )
    for pattern in [
        r'"lighting"\s*:\s*"[^"]*"',
        r'"other_beings"\s*:\s*\[[^\]]*\]',
        r'"nearby_anchors"\s*:\s*\[[^\]]*\]',
        r'"nearby_beings"\s*:\s*\[[^\]]*\]',
        r'"attention_to"\s*:\s*\{[^{}]*\}',
        r'"interactions"\s*:\s*\[[\s\S]*?\]\s*,\s*"environment_anomalies"',
        r'"environment_anomalies"\s*:\s*\[[\s\S]*?\]\s*,\s*"summary"',
        r'"abnormalities"\s*:\s*\[[^\]]*\]',
        r'"is_partially_occluded"\s*:\s*(?:true|false|null|"[^"]*")',
    ]:
        _set_regex_weight(
            assistant_text,
            char_weights,
            pattern,
            ENV_WEIGHT,
        )
    return char_weights


def token_weights_from_offsets(
    assistant_text: str,
    offsets: Iterable[tuple[int, int] | list[int]],
) -> torch.Tensor:
    """Project character weights onto tokenizer offsets."""

    char_weights = build_char_weights(assistant_text)
    token_weights = []
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if end <= start:
            token_weights.append(DEFAULT_WEIGHT)
        else:
            token_weights.append(max(char_weights[start:end]))
    return torch.tensor(token_weights, dtype=torch.float32)


def weight_config() -> dict[str, float]:
    """Return a serializable audit record for experiment configs."""

    return {
        "format": FORMAT_WEIGHT,
        "default": DEFAULT_WEIGHT,
        "environment": ENV_WEIGHT,
        "posture_action": POSTURE_ACTION_WEIGHT,
        "cat_presence": CAT_PRESENCE_WEIGHT,
    }
