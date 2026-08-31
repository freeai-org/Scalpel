from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from highway.utils.mixture_data import (
    GROUP_ORDER,
    MIXTURE_ORDER_INTERLEAVED,
    build_token_balanced_partitions,
    enforce_effective_batch_groups,
    fixed_group_sample_refs,
    load_fixed_group_sample,
    load_mixture_partition,
    proportionally_interleave_refs,
    scan_parquet_refs,
)
from highway.utils.sft_dataset import WeightedSFTDataset
from highway.data_recipes.build_scalpel_mix import (
    SourcePlan,
    TokenCounter,
    normalized_rows,
    transform_agieval,
    transform_medical_qa,
    transform_native_qa,
    validate_manifest,
)


def write_fixture(path: Path) -> None:
    rows = []
    for group_index, group in enumerate(GROUP_ORDER):
        for row_index in range(12):
            sample_id = f"{group}-{row_index}"
            conversations = [
                {"role": "user", "content": f"question {sample_id}"},
                {"role": "assistant", "content": f"answer {sample_id}"},
            ]
            rows.append(
                {
                    "id": sample_id,
                    "source": f"source-{group}",
                    "split": "default:train",
                    "group": group,
                    "language": "en",
                    "image": "",
                    "user": conversations[0]["content"],
                    "assistant": conversations[1]["content"],
                    "conversations_json": json.dumps(conversations),
                    "meta_json": "{}",
                    "token_estimate": 10 + group_index * 3 + row_index,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        length = 4 if len(messages) == 1 else 8
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


class MixtureDataTests(unittest.TestCase):
    def test_effective_batch_group_invariant_drops_partial_tail(self) -> None:
        samples = [
            {"group": group}
            for group in (
                "english", "chinese", "math", "code",
                "english", "chinese", "math", "code",
                "english",
            )
        ]
        retained, audit = enforce_effective_batch_groups(
            samples,
            effective_batch_size=4,
            required_groups=("math", "code"),
            drop_incomplete=True,
        )
        self.assertEqual(len(retained), 8)
        self.assertEqual(audit["dropped_tail_rows"], 1)
        self.assertEqual(audit["validated_effective_batches"], 2)

    def test_effective_batch_group_invariant_rejects_missing_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required"):
            enforce_effective_batch_groups(
                [{"group": "english"}, {"group": "math"}],
                effective_batch_size=2,
                required_groups=("math", "code"),
                drop_incomplete=False,
            )

    def test_partitions_are_deterministic_disjoint_and_group_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            _, grouped = scan_parquet_refs(root)
            first = build_token_balanced_partitions(grouped, parts=3, seed=17)
            second = build_token_balanced_partitions(grouped, parts=3, seed=17)
            self.assertEqual(first, second)
            all_ids = [ref.sample_id for part in first for ref in part]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(len(all_ids), 48)
            for part in first:
                groups = [ref.group for ref in part]
                self.assertEqual(groups, sorted(groups, key=GROUP_ORDER.index))
            for group in GROUP_ORDER:
                totals = [
                    sum(ref.token_estimate for ref in part if ref.group == group)
                    for part in first
                ]
                self.assertLessEqual(max(totals) - min(totals), 30)

    def test_fixed_sample_and_loaded_partition_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            _, grouped = scan_parquet_refs(root)
            refs = fixed_group_sample_refs(grouped, samples_per_group=2, seed=23)
            self.assertEqual(
                [ref.group for ref in refs],
                [group for group in GROUP_ORDER for _ in range(2)],
            )
            samples, stats = load_fixed_group_sample(
                root,
                samples_per_group=2,
                seed=23,
            )
            self.assertEqual([sample["group"] for sample in samples], [ref.group for ref in refs])
            self.assertEqual(stats["rows"], 8)
            partition, partition_stats = load_mixture_partition(
                root,
                part_index=2,
                parts=3,
                seed=23,
            )
            self.assertEqual(
                [sample["group"] for sample in partition],
                sorted([sample["group"] for sample in partition], key=GROUP_ORDER.index),
            )
            self.assertEqual(partition_stats["rows"], len(partition))

    def test_proportional_interleave_is_deterministic_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            _, grouped = scan_parquet_refs(root)
            grouped_partition = build_token_balanced_partitions(
                grouped,
                parts=3,
                seed=23,
            )[1]

            first = proportionally_interleave_refs(grouped_partition)
            second = proportionally_interleave_refs(grouped_partition)

            self.assertEqual(first, second)
            self.assertCountEqual(first, grouped_partition)
            for start in range(0, len(first), len(GROUP_ORDER)):
                self.assertEqual(
                    {ref.group for ref in first[start : start + len(GROUP_ORDER)]},
                    set(GROUP_ORDER),
                )

            samples, stats = load_mixture_partition(
                root,
                part_index=2,
                parts=3,
                seed=23,
                sample_order=MIXTURE_ORDER_INTERLEAVED,
            )
            self.assertEqual(
                stats["sample_order"],
                MIXTURE_ORDER_INTERLEAVED,
            )
            self.assertEqual(
                {sample["group"] for sample in samples[: len(GROUP_ORDER)]},
                set(GROUP_ORDER),
            )

    def test_weighted_dataset_accepts_text_only_rows(self) -> None:
        samples = [
            {
                "image": "",
                "group": "english",
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            }
        ]
        item = WeightedSFTDataset(samples, FakeProcessor(), max_length=16)[0]
        self.assertNotIn("pixel_values", item)
        self.assertEqual(item["input_ids"].numel(), 8)
        self.assertEqual(int((item["labels"] != -100).sum()), 4)
        self.assertEqual(int(item["group_id"]), 0)

    def test_weighted_dataset_refuses_to_truncate_assistant(self) -> None:
        samples = [
            {
                "id": "too-long",
                "source": "fixture",
                "image": "",
                "group": "english",
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            }
        ]
        dataset = WeightedSFTDataset(samples, FakeProcessor(), max_length=7)
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            dataset[0]

    def test_native_qa_transforms_keep_complete_answers(self) -> None:
        agieval = transform_agieval(
            {
                "passage": None,
                "question": "Two plus two?",
                "options": ["(A) 3", "(B) 4"],
                "label": "B",
                "other": {"solution": "Two plus two equals four."},
                "explanation": "Two plus two equals four.",
            },
            {"language": "en"},
        )
        self.assertIsNotNone(agieval)
        assert agieval is not None
        self.assertIn("A. 3", agieval[0])
        self.assertNotIn("A. (A)", agieval[0])
        self.assertEqual(agieval[1], "Answer: B\n\nTwo plus two equals four.")

        medical = transform_medical_qa(
            {
                "Question": "What is the diagnosis?",
                "Complex_CoT": "private generated reasoning",
                "Response": "The complete final answer.",
            },
            {"language": "en"},
        )
        self.assertEqual(
            medical,
            ("What is the diagnosis?", "The complete final answer."),
        )
        legal = transform_native_qa(
            {"question": "Q: Q: What is due process?", "answer": "A: A: A legal safeguard."},
            {"strip_qa_prefixes": True},
        )
        self.assertEqual(legal, ("What is due process?", "A legal safeguard."))

    def test_native_only_manifest_rejects_continuation_data(self) -> None:
        manifest = {
            "require_native_instruction_pairs": True,
            "mixture": {
                "english": {
                    "ratio": 1.0,
                    "sources": [
                        {
                            "name": "pseudo",
                            "transform": "text_continuation",
                            "weight": 1.0,
                        }
                    ],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "Native-instruction-only"):
            validate_manifest(manifest)

    def test_builder_drops_whole_overlength_qa_without_slicing(self) -> None:
        source = {
            "name": "fixture",
            "repo": "fixture/repo",
            "transform": "prompt_response",
            "language": "en",
        }
        plan = SourcePlan("english", 1.0, source, token_budget=1000)
        raw_rows = iter(
            [
                ("default:train", {"question": "short", "answer": "answer"}),
                ("default:train", {"question": "long", "answer": "x" * 100}),
            ]
        )
        counters: dict[str, int] = {}
        with patch(
            "highway.data_recipes.build_scalpel_mix.iter_dataset_rows",
            return_value=raw_rows,
        ):
            rows = list(
                normalized_rows(
                    plan,
                    TokenCounter(),
                    max_samples=0,
                    max_sequence_tokens=20,
                    counters=counters,
                )
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conversations"][1]["content"], "answer")
        self.assertEqual(counters["overlength"], 1)


if __name__ == "__main__":
    unittest.main()
