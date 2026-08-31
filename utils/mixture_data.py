"""Deterministic access to the Scalpel Parquet mixture.

The source Parquet files stay immutable.  This module builds virtual
token-balanced partitions and can expose either the legacy group-blocked order
or a proportionally interleaved order without writing another dataset copy.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GROUP_ORDER = ("english", "chinese", "math", "code")
GROUP_TO_ID = {group: index for index, group in enumerate(GROUP_ORDER)}
MIXTURE_ORDER_GROUPED = "grouped"
MIXTURE_ORDER_INTERLEAVED = "proportional_interleave_v1"
MIXTURE_ORDERS = (MIXTURE_ORDER_GROUPED, MIXTURE_ORDER_INTERLEAVED)
PARQUET_COLUMNS = (
    "id",
    "source",
    "split",
    "group",
    "language",
    "image",
    "user",
    "assistant",
    "conversations_json",
    "meta_json",
    "token_estimate",
)


@dataclass(frozen=True, slots=True)
class RowRef:
    file_index: int
    row_index: int
    sample_id: str
    group: str
    token_estimate: int


def parquet_files(path: Path) -> list[Path]:
    path = Path(path)
    files = [path] if path.is_file() and path.suffix == ".parquet" else []
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return files


def scan_parquet_refs(path: Path) -> tuple[list[Path], dict[str, list[RowRef]]]:
    """Read only the small columns needed to plan sampling and partitioning."""

    import pyarrow.parquet as pq

    files = parquet_files(path)
    grouped = {group: [] for group in GROUP_ORDER}
    for file_index, file_path in enumerate(files):
        metadata = pq.read_table(
            file_path,
            columns=["id", "group", "token_estimate"],
        ).to_pydict()
        for row_index, (sample_id, group, tokens) in enumerate(
            zip(metadata["id"], metadata["group"], metadata["token_estimate"])
        ):
            if group not in grouped:
                raise ValueError(f"Unknown mixture group {group!r} in {file_path}")
            grouped[group].append(
                RowRef(
                    file_index=file_index,
                    row_index=row_index,
                    sample_id=str(sample_id),
                    group=str(group),
                    token_estimate=int(tokens),
                )
            )
    return files, grouped


def _group_seed(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_token_balanced_partitions(
    grouped_refs: dict[str, list[RowRef]],
    parts: int,
    seed: int,
) -> list[list[RowRef]]:
    """Split every group across ``parts`` while balancing estimated tokens.

    Rows are shuffled deterministically inside each group, then greedily sent
    to the currently lightest partition for that group.  Concatenating groups
    in :data:`GROUP_ORDER` gives every partition the requested curriculum while
    keeping its token mixture close to the full 65/20/10/5 distribution.
    """

    if parts < 1:
        raise ValueError("parts must be positive")
    by_part = [{group: [] for group in GROUP_ORDER} for _ in range(parts)]
    for group in GROUP_ORDER:
        refs = list(grouped_refs.get(group, []))
        random.Random(_group_seed(seed, group)).shuffle(refs)
        token_heap = [(0, part_index) for part_index in range(parts)]
        heapq.heapify(token_heap)
        for ref in refs:
            token_total, part_index = heapq.heappop(token_heap)
            by_part[part_index][group].append(ref)
            heapq.heappush(
                token_heap,
                (token_total + ref.token_estimate, part_index),
            )
    return [
        [ref for group in GROUP_ORDER for ref in group_refs[group]]
        for group_refs in by_part
    ]


def proportionally_interleave_refs(refs: Iterable[RowRef]) -> list[RowRef]:
    """Interleave groups while keeping every prefix close to the row mixture.

    Rows inside each group retain their existing deterministic shuffled order.
    At each position, the group furthest below its expected cumulative count is
    emitted.  Therefore short windows track the partition's row proportions,
    while every row is still consumed exactly once.
    """

    grouped = {group: [] for group in GROUP_ORDER}
    for ref in refs:
        if ref.group not in grouped:
            raise ValueError(f"Unknown mixture group {ref.group!r}")
        grouped[ref.group].append(ref)

    totals = {group: len(grouped[group]) for group in GROUP_ORDER}
    total_rows = sum(totals.values())
    if total_rows == 0:
        return []

    emitted = {group: 0 for group in GROUP_ORDER}
    ordered: list[RowRef] = []
    for position in range(1, total_rows + 1):
        candidates = [
            group
            for group in GROUP_ORDER
            if emitted[group] < totals[group]
        ]
        selected = max(
            candidates,
            key=lambda group: (
                position * totals[group] - emitted[group] * total_rows,
                -GROUP_TO_ID[group],
            ),
        )
        ordered.append(grouped[selected][emitted[selected]])
        emitted[selected] += 1
    return ordered


def enforce_effective_batch_groups(
    samples: list[dict[str, Any]],
    *,
    effective_batch_size: int,
    required_groups: Iterable[str],
    drop_incomplete: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate required groups in every effective batch.

    When requested, the final incomplete accumulation window is removed so
    the invariant also holds for the last optimizer step.
    """

    if effective_batch_size < 1:
        raise ValueError("effective_batch_size must be positive")
    required = tuple(dict.fromkeys(str(group) for group in required_groups))
    unknown = [group for group in required if group not in GROUP_ORDER]
    if unknown:
        raise ValueError(f"Unknown required mixture groups: {unknown}")

    original_rows = len(samples)
    remainder = original_rows % effective_batch_size
    if remainder and drop_incomplete:
        samples = samples[: original_rows - remainder]

    checked_batches = 0
    for start in range(0, len(samples), effective_batch_size):
        batch = samples[start : start + effective_batch_size]
        if len(batch) < effective_batch_size and drop_incomplete:
            raise AssertionError("Incomplete effective batch survived trimming")
        present = {str(sample.get("group") or "") for sample in batch}
        missing = [group for group in required if group not in present]
        if missing:
            raise ValueError(
                "Effective batch is missing required mixture groups: "
                f"batch_index={checked_batches}, size={len(batch)}, "
                f"missing={missing}, present={sorted(present)}"
            )
        checked_batches += 1

    return samples, {
        "effective_batch_size": effective_batch_size,
        "required_groups": list(required),
        "drop_incomplete": drop_incomplete,
        "original_rows": original_rows,
        "retained_rows": len(samples),
        "dropped_tail_rows": original_rows - len(samples),
        "validated_effective_batches": checked_batches,
    }


def fixed_group_sample_refs(
    grouped_refs: dict[str, list[RowRef]],
    samples_per_group: int,
    seed: int,
) -> list[RowRef]:
    """Select a fixed random sample per group and return it in curriculum order."""

    if samples_per_group < 1:
        raise ValueError("samples_per_group must be positive")
    selected = []
    for group in GROUP_ORDER:
        refs = grouped_refs.get(group, [])
        if len(refs) < samples_per_group:
            raise ValueError(
                f"Group {group!r} has {len(refs)} rows; "
                f"cannot sample {samples_per_group}"
            )
        generator = random.Random(_group_seed(seed, group))
        selected.extend(generator.sample(refs, samples_per_group))
    return selected


def _parquet_row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    conversations_value = row.get("conversations_json")
    if conversations_value:
        conversations = json.loads(conversations_value)
    else:
        conversations = [
            {"role": "user", "content": str(row.get("user") or "")},
            {"role": "assistant", "content": str(row.get("assistant") or "")},
        ]
    meta_value = row.get("meta_json")
    return {
        "id": str(row.get("id") or ""),
        "source": str(row.get("source") or ""),
        "split": str(row.get("split") or ""),
        "group": str(row.get("group") or ""),
        "language": str(row.get("language") or ""),
        "image": str(row.get("image") or ""),
        "conversations": conversations,
        "token_estimate": int(row.get("token_estimate") or 0),
        "meta": json.loads(meta_value) if meta_value else {},
    }


def load_parquet_refs(files: list[Path], refs: Iterable[RowRef]) -> list[dict[str, Any]]:
    """Load selected rows and preserve the exact order of ``refs``."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    ordered_refs = list(refs)
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    refs_by_file: dict[int, list[int]] = {}
    for ref in ordered_refs:
        refs_by_file.setdefault(ref.file_index, []).append(ref.row_index)
    for file_index, row_indices in refs_by_file.items():
        unique_indices = sorted(set(row_indices))
        table = pq.read_table(files[file_index], columns=list(PARQUET_COLUMNS))
        selected = table.take(pa.array(unique_indices, type=pa.int64())).to_pylist()
        for row_index, row in zip(unique_indices, selected):
            rows_by_key[(file_index, row_index)] = _parquet_row_to_sample(row)
    return [rows_by_key[(ref.file_index, ref.row_index)] for ref in ordered_refs]


def summarize_refs(refs: Iterable[RowRef]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    total_rows = 0
    total_tokens = 0
    max_token_estimate = 0
    for ref in refs:
        counts[ref.group] += 1
        tokens[ref.group] += ref.token_estimate
        total_rows += 1
        total_tokens += ref.token_estimate
        max_token_estimate = max(max_token_estimate, ref.token_estimate)
    return {
        "rows": total_rows,
        "token_estimate": total_tokens,
        "max_token_estimate": max_token_estimate,
        "group_order": list(GROUP_ORDER),
        "groups": {
            group: {
                "rows": counts[group],
                "row_ratio": counts[group] / max(total_rows, 1),
                "token_estimate": tokens[group],
                "token_ratio": tokens[group] / max(total_tokens, 1),
            }
            for group in GROUP_ORDER
        },
    }


def load_mixture_partition(
    path: Path,
    *,
    part_index: int,
    parts: int,
    seed: int,
    max_samples: int | None = None,
    sample_order: str = MIXTURE_ORDER_GROUPED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one 1-based deterministic training partition."""

    if not 1 <= part_index <= parts:
        raise ValueError(f"part_index must be in [1, {parts}]")
    if sample_order not in MIXTURE_ORDERS:
        raise ValueError(
            f"sample_order must be one of {MIXTURE_ORDERS}, got {sample_order!r}"
        )
    files, grouped = scan_parquet_refs(path)
    refs = build_token_balanced_partitions(grouped, parts, seed)[part_index - 1]
    if sample_order == MIXTURE_ORDER_INTERLEAVED:
        refs = proportionally_interleave_refs(refs)
    if max_samples is not None:
        refs = refs[:max_samples]
    stats = summarize_refs(refs)
    stats["sample_order"] = sample_order
    manifest_path = path.parent / "manifest.lock.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        stats["dataset_version"] = manifest.get("version")
        stats["native_instruction_pairs_only"] = bool(
            manifest.get("require_native_instruction_pairs", False)
        )
        stats["manifest_max_sequence_tokens"] = int(
            manifest.get("max_sequence_tokens", 0)
        )
    return load_parquet_refs(files, refs), stats


def load_fixed_group_sample(
    path: Path,
    *,
    samples_per_group: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files, grouped = scan_parquet_refs(path)
    refs = fixed_group_sample_refs(grouped, samples_per_group, seed)
    return load_parquet_refs(files, refs), summarize_refs(refs)


def partition_summaries(path: Path, parts: int, seed: int) -> list[dict[str, Any]]:
    _, grouped = scan_parquet_refs(path)
    return [
        {"part_index": index, **summarize_refs(refs)}
        for index, refs in enumerate(
            build_token_balanced_partitions(grouped, parts, seed),
            start=1,
        )
    ]


def write_samples_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_training_samples(
    path: Path,
    max_samples: int | None = None,
    *,
    part_index: int = 0,
    parts: int = 1,
    seed: int = 42,
    mixture_order: str = MIXTURE_ORDER_GROUPED,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load legacy JSON data or one virtual Parquet mixture partition."""

    path = Path(path)
    is_parquet = path.suffix == ".parquet" or (
        path.is_dir() and any(path.glob("*.parquet"))
    )
    if is_parquet:
        resolved_part = part_index or 1
        return load_mixture_partition(
            path,
            part_index=resolved_part,
            parts=parts,
            seed=seed,
            max_samples=max_samples,
            sample_order=mixture_order,
        )

    from sft_scripts.utils.sft_io import load_sft_samples

    samples = load_sft_samples(path)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples, None
