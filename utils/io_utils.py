"""日志、表格与路径工具。"""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/mnt/d/models/soulgard-vl-bestloss-step6215")
DEFAULT_MODEL_ROOT = Path("/mnt/d/models/highway")
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "sft_data/sft_cat_train.json"
DEFAULT_VAL_DATA = PROJECT_ROOT / "sft_data/sft_cat_val.json"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "highway/runs"


def normalize_path(value: str | Path) -> Path:
    """同时接受 Linux 路径和 ``D:\\models`` 形式的 Windows 路径。"""

    text = str(value)
    if len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        drive = text[0].lower()
        text = f"/mnt/{drive}/{text[3:].replace(chr(92), '/')}"
    return Path(text).expanduser().resolve()


def make_run_id(prefix: str = "highway") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_model_state(path: Path, state: dict[str, Any]) -> None:
    """状态文件先写临时文件，避免训练中断留下半截 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_layer_state(model_path: Path, layer_count: int) -> dict[str, Any]:
    """读取删层状态；原始模型没有状态文件时生成默认 0..N-1 映射。"""

    state_path = model_path / "highway_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        original_ids = [int(value) for value in state["original_layer_ids"]]
        if len(original_ids) != layer_count:
            raise ValueError(f"{state_path} 的层号映射与模型层数不一致")
        return state
    return {
        "original_model": str(model_path),
        "original_layer_ids": list(range(layer_count)),
        "deleted_original_layers": [],
        "rounds": [],
    }
