"""Teacher-forced text metrics shared by layer ranking and validation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .data import PreparedSample, move_sample, supervised_positions
from .mixture_data import GROUP_ORDER
from .model_ops import prepare_language_inputs


@dataclass(slots=True)
class TokenMetric:
    nll_sum: float = 0.0
    correct: int = 0
    tokens: int = 0
    samples: int = 0

    def update(self, nll_sum: float, correct: int, tokens: int) -> None:
        self.nll_sum += float(nll_sum)
        self.correct += int(correct)
        self.tokens += int(tokens)
        self.samples += 1


class TextMetricAccumulator:
    def __init__(self) -> None:
        self.total = TokenMetric()
        self.groups: dict[str, TokenMetric] = defaultdict(TokenMetric)

    def update(self, group: str, nll_sum: float, correct: int, tokens: int) -> None:
        self.total.update(nll_sum, correct, tokens)
        self.groups[group].update(nll_sum, correct, tokens)

    @staticmethod
    def _metric_state(metric: TokenMetric) -> dict[str, Any]:
        return {
            "nll_sum": metric.nll_sum,
            "correct": metric.correct,
            "tokens": metric.tokens,
            "samples": metric.samples,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "total": self._metric_state(self.total),
            "groups": {
                group: self._metric_state(metric)
                for group, metric in self.groups.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TextMetricAccumulator":
        accumulator = cls()
        for name, value in state.get("total", {}).items():
            setattr(accumulator.total, name, value)
        for group, metric_state in state.get("groups", {}).items():
            metric = accumulator.groups[group]
            for name, value in metric_state.items():
                setattr(metric, name, value)
        return accumulator

    @staticmethod
    def _summary(metric: TokenMetric) -> dict[str, Any]:
        mean_nll = metric.nll_sum / max(metric.tokens, 1)
        return {
            "samples": metric.samples,
            "tokens": metric.tokens,
            "mean_nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 80.0)),
            "token_accuracy": metric.correct / max(metric.tokens, 1),
        }

    def summary(self) -> dict[str, Any]:
        group_summaries = {
            group: self._summary(self.groups[group])
            for group in GROUP_ORDER
        }
        present_accuracies = [
            metrics["token_accuracy"]
            for metrics in group_summaries.values()
            if metrics["tokens"]
        ]
        # An internal, model-comparable leaderboard score.  Each of the four
        # groups gets equal weight because the fixed probe contains 20 rows per
        # group.  It is intentionally not presented as an external benchmark.
        leaderboard_score = (
            100.0 * sum(present_accuracies) / max(len(present_accuracies), 1)
        )
        return {
            **self._summary(self.total),
            "leaderboard_score": leaderboard_score,
            "leaderboard_definition": "100 * macro_group_teacher_forced_token_accuracy",
            "groups": group_summaries,
        }


@torch.inference_mode()
def score_hidden_states(
    model: Any,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
) -> tuple[float, int, int]:
    """Project only selected answer positions, keeping vocab memory bounded."""

    nll_sum = 0.0
    correct = 0
    token_count = int(positions.numel())
    for start in range(0, token_count, chunk_size):
        chunk_positions = positions[start : start + chunk_size]
        chunk_targets = targets[:, start : start + chunk_size]
        logits = model.lm_head(hidden_states[:, chunk_positions, :]).float()
        nll_sum += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk_targets.reshape(-1),
                reduction="sum",
            ).item()
        )
        correct += int((logits.argmax(dim=-1) == chunk_targets).sum().item())
        del logits
    return nll_sum, correct, token_count


@torch.inference_mode()
def prepare_scoring_sample(
    model: Any,
    sample: PreparedSample,
    device: torch.device,
    max_tokens: int,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    moved = move_sample(sample, device)
    available = int((moved.labels[:, 1:] != -100).sum().item())
    positions, targets = supervised_positions(
        moved.labels,
        max_tokens=available if max_tokens <= 0 else min(available, max_tokens),
    )
    return prepare_language_inputs(model, moved.model_inputs), positions, targets
