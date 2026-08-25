"""Training log callback without importing the project's LoRA utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class LossJsonlCallback(TrainerCallback):
    """Append numeric Trainer logs to JSONL for boundary-loss summaries."""

    def __init__(self, log_path: Path, reset: bool = True) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.log_path.exists():
            self.log_path.unlink()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{
                key: value
                for key, value in logs.items()
                if isinstance(value, (int, float))
            },
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
