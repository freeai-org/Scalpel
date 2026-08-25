import argparse
import json
import tempfile
import unittest
from pathlib import Path

from highway.eval_generation import (
    ModelTarget,
    checkpoint_for_round,
    create_probe_manifest,
    plan_tasks,
    summarize,
)
from highway.eval_logits import probe_sample_rows


class EvalGenerationTests(unittest.TestCase):
    def test_current_layout_checkpoint_and_fixed_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_run = root / "source"
            checkpoint_root = root / "models"
            round_dir = source_run / "round_01"
            round_dir.mkdir(parents=True)
            (round_dir / "round_summary.json").write_text("{}", encoding="utf-8")
            checkpoint = checkpoint_root / "round_01" / "post_recovery_model"
            checkpoint.mkdir(parents=True)
            probe_dir = source_run / "probe"
            probe_dir.mkdir()
            probe_rows = [
                {
                    "index_in_source": 7,
                    "repeat_id": 1,
                    "position_in_repeat": 1,
                },
                {
                    "index_in_source": 7,
                    "repeat_id": 2,
                    "position_in_repeat": 1,
                },
            ]
            (probe_dir / "probe_10x10.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in probe_rows),
                encoding="utf-8",
            )

            self.assertEqual(
                checkpoint_for_round(source_run, checkpoint_root, 1),
                checkpoint,
            )
            manifest = create_probe_manifest(
                source_run,
                1,
                root / "val.json",
                root / "manifests",
            )
            self.assertEqual(manifest.name, "probe_round_01_fixed.jsonl")
            logits_rows = probe_sample_rows(source_run, 1, max_samples=0)
            self.assertEqual(
                [row["sample_index"] for row in logits_rows],
                [7, 7],
            )
            self.assertEqual(
                [row["probe_repeat"] for row in logits_rows],
                [1, 2],
            )

    def test_legacy_probe_manifest_preserves_repeat_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_run = root / "source"
            (source_run / "round_01").mkdir(parents=True)
            (source_run / "round_01" / "probe_selected.json").write_text(
                json.dumps({"sample_subsets": [[3, 5], [5, 8]]}),
                encoding="utf-8",
            )

            manifest = create_probe_manifest(
                source_run,
                1,
                root / "val.json",
                root / "manifests",
            )
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual([row["index_in_source"] for row in rows], [3, 5, 5, 8])
            self.assertEqual([row["probe_repeat"] for row in rows], [1, 1, 2, 2])

    def test_summary_reports_metric_delta_against_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(output_dir=root)
            base = ModelTarget("base_28_layers", root / "base", 0)
            pruned = ModelTarget("round_01_layers_27", root / "round01", 1)
            tasks = [
                argparse.Namespace(
                    dataset="full_val",
                    val_json=root / "val.json",
                    model=base,
                    reference_label="base_28_layers",
                ),
                argparse.Namespace(
                    dataset="full_val",
                    val_json=root / "val.json",
                    model=pruned,
                    reference_label="base_28_layers",
                ),
            ]
            for label, accuracy in [("base_28_layers", 0.9), ("round_01_layers_27", 0.8)]:
                metrics_dir = root / "full_val" / label
                metrics_dir.mkdir(parents=True)
                (metrics_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "total_samples": 10,
                            "cat_recognition_accuracy": accuracy,
                            "cat_count_accuracy": accuracy,
                            "behavior_analysis_accuracy": accuracy,
                            "behavior_accuracy_on_gt_cat_images": accuracy,
                            "speed": {},
                            "paths": {},
                        }
                    ),
                    encoding="utf-8",
                )

            summary_path = summarize(args, tasks)
            lines = summary_path.read_text(encoding="utf-8").splitlines()

            self.assertIn("delta_cat_recognition_accuracy", lines[0])
            self.assertIn("-0.09999999999999998", lines[2])

    def test_plan_probe_tasks_use_previous_round_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_run = root / "source"
            for round_number in (1, 2):
                round_dir = source_run / f"round_{round_number:02d}"
                round_dir.mkdir(parents=True)
                (round_dir / "probe_selected.json").write_text(
                    json.dumps({"sample_subsets": [[round_number]]}),
                    encoding="utf-8",
                )
            args = argparse.Namespace(
                skip_full=True,
                skip_probe=False,
                accepted_rounds=2,
                output_dir=root / "eval",
                source_run_dir=source_run,
                val_json=root / "val.json",
            )
            targets = [
                ModelTarget("base_28_layers", root / "base", 0),
                ModelTarget("round_01_layers_27", root / "r1", 1),
                ModelTarget("round_02_layers_26", root / "r2", 2),
            ]

            tasks = plan_tasks(args, targets)
            reference_pairs = [
                (task.dataset, task.model.label, task.reference_label)
                for task in tasks
            ]

            self.assertIn(
                ("probe_round_01_fixed", "round_01_layers_27", "base_28_layers"),
                reference_pairs,
            )
            self.assertIn(
                ("probe_round_02_fixed", "round_02_layers_26", "round_01_layers_27"),
                reference_pairs,
            )


if __name__ == "__main__":
    unittest.main()
