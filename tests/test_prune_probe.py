from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from highway.utils.model_ops import load_processor
from highway.prune_probe import (
    build_generation_inputs,
    write_candidate_predictions,
)


class DummyBatch:
    def to(self, device):
        return self


class RecordingProcessor:
    def __init__(self, chat_template: str) -> None:
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return DummyBatch()


class PruneProbeInputTests(unittest.TestCase):
    def test_processor_uses_left_padding_for_batched_generation(self) -> None:
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(padding_side="right"),
        )
        with patch(
            "highway.utils.model_ops.AutoProcessor.from_pretrained",
            return_value=processor,
        ):
            loaded = load_processor(Path("/tmp/model"))

        self.assertIs(loaded, processor)
        self.assertEqual(loaded.tokenizer.padding_side, "left")

    def test_variable_length_batches_request_explicit_longest_padding(self) -> None:
        processor = RecordingProcessor(chat_template="plain template")
        samples = [
            {
                "image": "/tmp/a.jpg",
                "conversations": [
                    {"role": "user", "content": "ignored"},
                    {"role": "assistant", "content": "{}"},
                ],
            },
            {
                "image": "/tmp/b.jpg",
                "conversations": [
                    {"role": "user", "content": "ignored"},
                    {"role": "assistant", "content": "{}"},
                ],
            },
        ]

        build_generation_inputs(processor, samples, device="cpu")

        _, kwargs = processor.calls[0]
        self.assertEqual(
            kwargs["processor_kwargs"],
            {"padding": "longest"},
        )
        self.assertNotIn("enable_thinking", kwargs)

    def test_thinking_flag_is_only_passed_when_template_declares_it(self) -> None:
        processor = RecordingProcessor(
            chat_template="{% if enable_thinking %}think{% endif %}"
        )
        sample = {
            "image": "/tmp/a.jpg",
            "conversations": [
                {"role": "user", "content": "ignored"},
                {"role": "assistant", "content": "{}"},
            ],
        }

        build_generation_inputs(processor, [sample], device="cpu")

        _, kwargs = processor.calls[0]
        self.assertIs(kwargs["enable_thinking"], False)

    def test_candidate_generation_retries_failed_batch_as_single_items(self) -> None:
        samples = [
            {"image": "/tmp/a.jpg"},
            {"image": "/tmp/b.jpg"},
        ]
        manifests = [
            {
                "probe_row_id": index,
                "repeat_id": 1,
                "position_in_repeat": index + 1,
                "index_in_source": index,
                "image": sample["image"],
                "ground_truth": {"cats_visible": 0, "cats": []},
            }
            for index, sample in enumerate(samples)
        ]

        def fake_generate(model, processor, batch, *args, **kwargs):
            if len(batch) > 1:
                raise ValueError("variable image-token lengths")
            return [('{"cats_visible": 0, "cats": []}', {"cats_visible": 0, "cats": []})]

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.jsonl"
            with patch(
                "highway.prune_probe.generate_with_bypass",
                side_effect=fake_generate,
            ):
                rows = write_candidate_predictions(
                    model=object(),
                    processor=object(),
                    samples=samples,
                    manifest_rows=manifests,
                    layer_index=3,
                    original_layer_id=3,
                    output_path=output_path,
                    batch_size=2,
                    max_new_tokens=64,
                    device="cpu",
                )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["prediction"] is not None for row in rows))


if __name__ == "__main__":
    unittest.main()
