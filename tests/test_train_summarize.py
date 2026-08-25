from __future__ import annotations

import math
import unittest

from highway.train_summarize import summarize_rows


class TrainSummarizeTest(unittest.TestCase):
    def test_summary_uses_latest_replayed_step(self) -> None:
        rows = [
            self.row(5, 0.8, 0.6, 0.2),
            self.row(10, 0.7, 0.5, 0.2),
            self.row(10, 0.6, 0.45, 0.15),
            self.row(15, 0.5, 0.4, 0.1),
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["raw_records"], 4)
        self.assertEqual(summary["unique_steps"], 3)
        self.assertEqual(summary["duplicate_steps"], [10])
        self.assertEqual(summary["last_step"], 15)
        self.assertAlmostEqual(summary["fields"]["loss"]["mean"], 0.6333333333)
        self.assertAlmostEqual(
            summary["max_loss_decomposition_abs_error"],
            0.0,
        )

    def test_nonfinite_values_are_counted(self) -> None:
        rows = [
            self.row(5, 0.8, 0.6, 0.2),
            self.row(10, math.nan, 0.5, 0.2),
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["nonfinite_counts"]["loss"], 1)
        self.assertEqual(summary["fields"]["loss"]["first"], 0.8)

    def test_final_runtime_record_without_loss_is_skipped(self) -> None:
        rows = [
            self.row(5, 0.8, 0.6, 0.2),
            {
                "step": 5,
                "epoch": 1.0,
                "train_runtime": 12.3,
                "train_samples_per_second": 1.5,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["raw_records"], 2)
        self.assertEqual(summary["loss_component_records"], 1)
        self.assertEqual(summary["skipped_non_loss_records"], 1)
        self.assertEqual(summary["duplicate_steps"], [])
        self.assertEqual(summary["fields"]["loss"]["last"], 0.8)

    @staticmethod
    def row(step: int, loss: float, hard: float, soft: float) -> dict[str, float]:
        return {
            "step": step,
            "epoch": step / 100.0,
            "loss": loss,
            "hard_weighted_ce": hard,
            "soft_weighted_kl": soft,
            "grad_norm": 0.3,
            "learning_rate": 1e-4,
        }


if __name__ == "__main__":
    unittest.main()
