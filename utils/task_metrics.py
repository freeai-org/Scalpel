"""Hard task metrics used to rank layer-deletion candidates."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from sft_scripts.utils.sft_metrics import (
    behavior_match,
    cat_count_match,
    cat_presence_match,
    cats_visible,
    compare_behavior_fields,
)


ATOMIC_BEHAVIOR_FIELDS = [
    "location_on",
    "vertical_position",
    "action",
    "overall_body",
    "ears",
    "tail",
    "face",
    "fur_state",
]
DIAGNOSTIC_BEHAVIOR_FIELDS = [*ATOMIC_BEHAVIOR_FIELDS, "posture_full"]
MACRO_FIELDS = ["cat_presence", "cat_count", *ATOMIC_BEHAVIOR_FIELDS]


@dataclass(slots=True)
class MetricCounter:
    correct: int = 0
    total: int = 0

    def update(self, is_correct: bool, applicable: bool = True) -> None:
        if not applicable:
            return
        self.total += 1
        self.correct += int(is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(slots=True)
class TaskMetricAccumulator:
    """Accumulate metrics while penalizing missing predicted cats."""

    counters: dict[str, MetricCounter] = field(
        default_factory=lambda: {
            name: MetricCounter()
            for name in [
                "cat_presence",
                "cat_count",
                *DIAGNOSTIC_BEHAVIOR_FIELDS,
                "strict_behavior",
                "parse_success",
            ]
        }
    )

    def update(
        self,
        ground_truth: dict[str, Any],
        prediction: dict[str, Any] | None,
    ) -> None:
        self.counters["parse_success"].update(prediction is not None)
        self.counters["cat_presence"].update(
            cat_presence_match(ground_truth, prediction)
        )
        self.counters["cat_count"].update(cat_count_match(ground_truth, prediction))

        gt_has_cat = cats_visible(ground_truth) > 0
        self.counters["strict_behavior"].update(
            behavior_match(ground_truth, prediction),
            applicable=gt_has_cat,
        )
        if not gt_has_cat:
            return

        gt_cats = ground_truth.get("cats", [])
        pred_cats = prediction.get("cats", []) if prediction else []
        for cat_index, gt_cat in enumerate(gt_cats):
            if cat_index < len(pred_cats):
                field_matches = compare_behavior_fields(
                    gt_cat,
                    pred_cats[cat_index],
                )
            else:
                field_matches = {
                    name: False for name in DIAGNOSTIC_BEHAVIOR_FIELDS
                }
            for name in DIAGNOSTIC_BEHAVIOR_FIELDS:
                self.counters[name].update(bool(field_matches[name]))

    def summary(self) -> dict[str, Any]:
        fields = {
            name: {
                "accuracy": counter.accuracy,
                "correct": counter.correct,
                "total": counter.total,
            }
            for name, counter in self.counters.items()
        }
        macro_accuracy = statistics.fmean(
            fields[name]["accuracy"] for name in MACRO_FIELDS
        )
        return {
            "macro_field_accuracy": float(macro_accuracy),
            "macro_fields": list(MACRO_FIELDS),
            "fields": fields,
        }


def summarize_predictions(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    accumulator = TaskMetricAccumulator()
    count = 0
    for row in rows:
        ground_truth = row.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValueError("Every scored row must contain a ground_truth object")
        prediction = row.get("prediction")
        accumulator.update(
            ground_truth,
            prediction if isinstance(prediction, dict) else None,
        )
        count += 1
    summary = accumulator.summary()
    summary["samples"] = count
    return summary


def hard_regret(reference_accuracy: float, candidate_accuracy: float) -> float:
    """Return bounded relative accuracy degradation; improvements receive zero."""

    if not 0.0 <= reference_accuracy <= 1.0:
        raise ValueError("reference_accuracy must be in [0, 1]")
    if not 0.0 <= candidate_accuracy <= 1.0:
        raise ValueError("candidate_accuracy must be in [0, 1]")
    denominator = max(reference_accuracy, 1e-12)
    return min(1.0, max(0.0, (reference_accuracy - candidate_accuracy) / denominator))


def pruning_risk(relative_hard_regret: float, normalized_js: float) -> float:
    """Chebyshev scalarization: minimize the worst normalized degradation."""

    if not 0.0 <= relative_hard_regret <= 1.0:
        raise ValueError("relative_hard_regret must be in [0, 1]")
    if not 0.0 <= normalized_js <= 1.0:
        raise ValueError("normalized_js must be in [0, 1]")
    return max(relative_hard_regret, normalized_js)
