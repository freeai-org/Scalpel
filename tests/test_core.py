from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from highway.utils.data import repeated_random_subsets, supervised_positions
from highway.utils.io_utils import load_layer_state, normalize_path, write_csv
from highway.utils.metrics import (
    compare_logits,
    distribution_kl,
    field_weighted_kd_loss,
    normalized_jensen_shannon,
)
from highway.utils.task_metrics import hard_regret, pruning_risk, summarize_predictions
from highway.utils.training_collator import DynamicKDCollator
from highway.utils.model_ops import (
    bypass_layer,
    configure_trainable_layers,
    enable_generation_cache,
    get_layers,
    physical_delete_layer,
    temporary_delete_layer,
)
from highway.prune_highway import (
    generated_model_ready,
    quarantine_incomplete_model,
)


class DummyLayer(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.self_attn = SimpleNamespace(layer_idx=index)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.linear(hidden_states)


class DummyLanguageModel(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(index) for index in range(count)])
        self.config = SimpleNamespace(num_hidden_layers=count)


class DummyCore(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.language_model = DummyLanguageModel(count)


class DummyModel(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.model = DummyCore(count)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=count),
            num_hidden_layers=count,
        )


class CoreTests(unittest.TestCase):
    def test_repeated_subsets_are_reproducible(self) -> None:
        first = repeated_random_subsets(list(range(20)), 5, 10, 42)
        second = repeated_random_subsets(list(range(20)), 5, 10, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(len(set(row)) == 10 for row in first))

    def test_supervised_positions_apply_causal_shift(self) -> None:
        labels = torch.tensor([[-100, -100, 7, 8, 9]])
        positions, targets = supervised_positions(labels, max_tokens=8)
        self.assertEqual(positions.tolist(), [1, 2, 3])
        self.assertEqual(targets.tolist(), [[7, 8, 9]])

    def test_bypass_restores_original_layer(self) -> None:
        model = DummyModel(4)
        original = get_layers(model)[2]
        with bypass_layer(model, 2):
            value = torch.randn(1, 2, 4)
            self.assertTrue(torch.equal(get_layers(model)[2](value), value))
        self.assertIs(get_layers(model)[2], original)

    def test_physical_delete_keeps_original_id_mapping(self) -> None:
        model = DummyModel(5)
        deleted, mapping = physical_delete_layer(model, 2, [0, 1, 2, 3, 4])
        self.assertEqual(deleted, 2)
        self.assertEqual(mapping, [0, 1, 3, 4])
        self.assertEqual(len(get_layers(model)), 4)
        self.assertEqual(model.config.text_config.num_hidden_layers, 4)
        self.assertEqual(
            [layer.self_attn.layer_idx for layer in get_layers(model)],
            [0, 1, 2, 3],
        )

    def test_temporary_delete_restores_layers_config_and_cache_indices(self) -> None:
        model = DummyModel(5)
        original_layers = list(get_layers(model))
        with temporary_delete_layer(model, 2):
            self.assertEqual(len(get_layers(model)), 4)
            self.assertEqual(model.config.text_config.num_hidden_layers, 4)
            self.assertEqual(
                [layer.self_attn.layer_idx for layer in get_layers(model)],
                [0, 1, 2, 3],
            )
        self.assertEqual(list(get_layers(model)), original_layers)
        self.assertEqual(model.config.text_config.num_hidden_layers, 5)
        self.assertEqual(
            [layer.self_attn.layer_idx for layer in get_layers(model)],
            [0, 1, 2, 3, 4],
        )

    def test_generation_cache_is_enabled_in_persisted_configs(self) -> None:
        model = DummyModel(3)
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.generation_config = SimpleNamespace(use_cache=False)
        enable_generation_cache(model)
        self.assertTrue(model.config.use_cache)
        self.assertTrue(model.config.text_config.use_cache)
        self.assertTrue(model.generation_config.use_cache)

    def test_previous_and_prefix_trainable_modes(self) -> None:
        previous = DummyModel(4)
        indices, count = configure_trainable_layers(previous, 2, "previous")
        self.assertEqual(indices, [1])
        self.assertGreater(count, 0)
        self.assertFalse(any(p.requires_grad for p in get_layers(previous)[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in get_layers(previous)[1].parameters()))

        prefix = DummyModel(4)
        indices, _ = configure_trainable_layers(prefix, 2, "prefix")
        self.assertEqual(indices, [0, 1])
        self.assertTrue(all(p.requires_grad for p in get_layers(prefix)[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in get_layers(prefix)[2].parameters()))

    def test_probability_metrics(self) -> None:
        reference = torch.tensor([[[3.0, 1.0], [0.5, 2.5]]])
        identical_kl = distribution_kl(reference, reference, temperature=2.0)
        self.assertAlmostEqual(float(identical_kl), 0.0, places=6)
        result = compare_logits(reference, reference, torch.tensor([[0, 1]]), 1.0)
        self.assertAlmostEqual(result.ce_delta, 0.0, places=6)
        self.assertEqual(result.top1_agreement, 1.0)
        identical_js = normalized_jensen_shannon(reference, reference)
        self.assertAlmostEqual(float(identical_js), 0.0, places=6)
        opposite = normalized_jensen_shannon(
            torch.tensor([[[100.0, -100.0]]]),
            torch.tensor([[[-100.0, 100.0]]]),
        )
        self.assertAlmostEqual(float(opposite), 1.0, places=5)

    def test_task_risk_is_bounded_and_does_not_hide_one_bad_axis(self) -> None:
        self.assertAlmostEqual(hard_regret(0.8, 0.72), 0.1)
        self.assertEqual(hard_regret(0.8, 0.9), 0.0)
        self.assertEqual(pruning_risk(0.0, 0.4), 0.4)
        self.assertEqual(pruning_risk(0.3, 0.0), 0.3)

    def test_weighted_kd_has_zero_soft_loss_for_identical_distributions(self) -> None:
        logits = torch.tensor([[[3.0, 1.0], [0.5, 2.5]]])
        losses = field_weighted_kd_loss(
            logits,
            logits,
            torch.tensor([[0, 1]]),
            torch.tensor([[1.0, 3.0]]),
        )
        self.assertAlmostEqual(float(losses.soft_kl), 0.0, places=6)
        self.assertAlmostEqual(
            float(losses.total),
            float(losses.hard_ce),
            places=6,
        )

    def test_batched_weighted_kd_matches_mean_of_individual_samples(self) -> None:
        torch.manual_seed(7)
        student = torch.randn(2, 4, 11)
        teacher = torch.randn(2, 4, 11)
        labels = torch.tensor(
            [
                [1, 2, 3, -100],
                [4, 5, -100, -100],
            ]
        )
        weights = torch.tensor(
            [
                [1.0, 3.0, 0.5, 1.0],
                [2.0, 1.0, 1.0, 1.0],
            ]
        )

        batched = field_weighted_kd_loss(
            student,
            teacher,
            labels,
            weights,
        )
        individual = [
            field_weighted_kd_loss(
                student[index : index + 1],
                teacher[index : index + 1],
                labels[index : index + 1],
                weights[index : index + 1],
            )
            for index in range(2)
        ]

        self.assertTrue(
            torch.allclose(
                batched.hard_ce,
                torch.stack([loss.hard_ce for loss in individual]).mean(),
            )
        )
        self.assertTrue(
            torch.allclose(
                batched.soft_kl,
                torch.stack([loss.soft_kl for loss in individual]).mean(),
            )
        )
        self.assertTrue(
            torch.allclose(
                batched.total,
                torch.stack([loss.total for loss in individual]).mean(),
            )
        )

    def test_dynamic_kd_collator_pads_text_and_concatenates_images(self) -> None:
        collator = DynamicKDCollator(pad_token_id=99, max_length=8)
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, 3]),
                "loss_weights": torch.tensor([1.0, 2.0, 3.0]),
                "pixel_values": torch.ones(2, 4),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 5]),
                "loss_weights": torch.tensor([1.0, 2.0]),
                "pixel_values": torch.zeros(3, 4),
                "image_grid_thw": torch.tensor([[1, 1, 3]]),
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(int(batch["input_ids"][1, 2]), 99)
        self.assertEqual(int(batch["labels"][1, 2]), -100)
        self.assertEqual(tuple(batch["pixel_values"].shape), (5, 4))
        self.assertEqual(tuple(batch["image_grid_thw"].shape), (2, 3))

    def test_task_macro_uses_atomic_fields_and_penalizes_missing_cat(self) -> None:
        ground_truth = {
            "cats_visible": 1,
            "cats": [
                {
                    "location_on": "地面",
                    "vertical_position": "地面",
                    "action": "静止",
                    "posture": {
                        "overall_body": "站立",
                        "ears": {"visible": True, "position": "正常竖立"},
                        "tail": {"visible": True, "position": "下垂"},
                        "face": {
                            "visible": True,
                            "eyelid": "睁开",
                            "mouth": "闭合",
                        },
                        "fur_state": "正常",
                    },
                }
            ],
        }
        summary = summarize_predictions(
            [{"ground_truth": ground_truth, "prediction": None}]
        )
        self.assertEqual(summary["macro_field_accuracy"], 0.0)
        self.assertEqual(summary["fields"]["action"]["total"], 1)

    def test_io_helpers_and_layer_state(self) -> None:
        self.assertEqual(normalize_path(r"D:\models\x"), Path("/mnt/d/models/x"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            write_csv(path, [{"layer": 1, "delta": 0.2}])
            self.assertIn("layer,delta", path.read_text(encoding="utf-8"))
            state = load_layer_state(Path(directory), 3)
            self.assertEqual(state["original_layer_ids"], [0, 1, 2])

            state_path = Path(directory) / "highway_state.json"
            state_path.write_text(
                '{"original_layer_ids": [0, 2], "deleted_original_layers": [1]}',
                encoding="utf-8",
            )
            self.assertEqual(load_layer_state(Path(directory), 2)["original_layer_ids"], [0, 2])
            with self.assertRaises(ValueError):
                load_layer_state(Path(directory), 3)

    def test_generated_model_requires_weights_config_and_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            self.assertFalse(generated_model_ready(model))
            (model / "highway_state.json").write_text("{}", encoding="utf-8")
            self.assertTrue(generated_model_ready(model))

    def test_incomplete_model_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "partial.tmp").write_text("partial", encoding="utf-8")
            quarantine = quarantine_incomplete_model(model)
            self.assertIsNotNone(quarantine)
            self.assertFalse(model.exists())
            self.assertTrue(quarantine.exists())


if __name__ == "__main__":
    unittest.main()
