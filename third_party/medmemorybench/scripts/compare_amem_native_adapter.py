#!/usr/bin/env python3
"""Compare matched A-MEM native and MedMemoryBench-adapter LoCoMo results."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        _mean([values[rng.randrange(size)] for _ in range(size)])
        for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _cohen_kappa(first: list[bool], second: list[bool]) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    size = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / size
    first_positive = sum(first) / size
    second_positive = sum(second) / size
    expected = (
        first_positive * second_positive
        + (1.0 - first_positive) * (1.0 - second_positive)
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def compare(
    native_path: Path,
    adapter_query_path: Path,
    adapter_memory_path: Path,
    *,
    native_initial_log: Path | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    native = json.loads(native_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_query_path.read_text(encoding="utf-8"))
    adapter_memory = json.loads(adapter_memory_path.read_text(encoding="utf-8"))
    sample_id = str(native["config"]["sample_id"])

    native_rows = native["queries"]
    adapter_rows = [
        row for row in adapter["queries"] if str(row.get("context_id")) == sample_id
    ]
    native_by_id = {str(row["query_id"]): row for row in native_rows}
    adapter_by_id = {str(row["query_id"]): row for row in adapter_rows}
    if len(native_by_id) != len(native_rows) or len(adapter_by_id) != len(adapter_rows):
        raise ValueError("Duplicate query IDs in native or adapter results")
    if set(native_by_id) != set(adapter_by_id):
        raise ValueError(
            "Native/adapter query coverage differs: "
            f"native_only={sorted(set(native_by_id) - set(adapter_by_id))[:5]}, "
            f"adapter_only={sorted(set(adapter_by_id) - set(native_by_id))[:5]}"
        )

    query_ids = sorted(native_by_id)
    native_scores = [float(native_by_id[key]["score"]) for key in query_ids]
    adapter_scores = [float(adapter_by_id[key]["score"]) for key in query_ids]
    score_deltas = [n - a for n, a in zip(native_scores, adapter_scores)]
    native_correct = [bool(native_by_id[key]["is_correct"]) for key in query_ids]
    adapter_correct = [bool(adapter_by_id[key]["is_correct"]) for key in query_ids]
    delta_ci = _bootstrap_mean_ci(score_deltas, samples=bootstrap_samples, seed=seed)

    by_type: dict[str, list[str]] = defaultdict(list)
    for key in query_ids:
        by_type[str(native_by_id[key]["query_type"])].append(key)

    adapter_units = [
        unit
        for unit in adapter_memory.get("units", [])
        if str(unit.get("context_id")) == sample_id
    ]
    if len(adapter_units) != 1:
        raise ValueError(
            f"Expected one adapter memory unit for {sample_id}, found {len(adapter_units)}"
        )
    adapter_unit = adapter_units[0]
    original_native_build_seconds = None
    if native_initial_log is not None:
        initial_text = native_initial_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r'"loaded_memory_cache"\s*:\s*false.*?'
            r'"total_memory_construction_time"\s*:\s*([0-9.eE+-]+)',
            initial_text,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(
                f"Could not recover original native build time from {native_initial_log}"
            )
        original_native_build_seconds = float(match.group(1))

    return {
        "contract": "amem.native_adapter_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "inputs": {
            "native": str(native_path.resolve()),
            "adapter_query": str(adapter_query_path.resolve()),
            "adapter_memory": str(adapter_memory_path.resolve()),
            "native_initial_log": (
                str(native_initial_log.resolve()) if native_initial_log is not None else None
            ),
        },
        "coverage": {
            "queries": len(query_ids),
            "native_empty_retrievals": sum(
                int(row.get("retrieved_count", 0)) <= 0 for row in native_rows
            ),
            "adapter_empty_retrievals": sum(
                int(row.get("retrieved_count", 0)) <= 0 for row in adapter_rows
            ),
        },
        "scores": {
            "native_mean_official_f1": _mean(native_scores),
            "adapter_mean_official_f1": _mean(adapter_scores),
            "native_minus_adapter_mean_f1": _mean(score_deltas),
            "native_minus_adapter_bootstrap_ci95": list(delta_ci),
            "native_threshold_accuracy": _mean([float(value) for value in native_correct]),
            "adapter_threshold_accuracy": _mean([float(value) for value in adapter_correct]),
            "threshold_agreement": _mean(
                [float(a == b) for a, b in zip(native_correct, adapter_correct)]
            ),
            "threshold_cohen_kappa": _cohen_kappa(native_correct, adapter_correct),
            "contingency": {
                "both_correct": sum(a and b for a, b in zip(native_correct, adapter_correct)),
                "native_only_correct": sum(
                    a and not b for a, b in zip(native_correct, adapter_correct)
                ),
                "adapter_only_correct": sum(
                    not a and b for a, b in zip(native_correct, adapter_correct)
                ),
                "both_incorrect": sum(
                    not a and not b for a, b in zip(native_correct, adapter_correct)
                ),
            },
            "by_type": {
                query_type: {
                    "queries": len(keys),
                    "native_mean_official_f1": _mean(
                        [float(native_by_id[key]["score"]) for key in keys]
                    ),
                    "adapter_mean_official_f1": _mean(
                        [float(adapter_by_id[key]["score"]) for key in keys]
                    ),
                    "native_threshold_accuracy": _mean(
                        [float(bool(native_by_id[key]["is_correct"])) for key in keys]
                    ),
                    "adapter_threshold_accuracy": _mean(
                        [float(bool(adapter_by_id[key]["is_correct"])) for key in keys]
                    ),
                }
                for query_type, keys in sorted(by_type.items())
            },
        },
        "memory_protocol": {
            "native_ingestion": native["implementation"]["ingestion"],
            "native_sessions": native["coverage"]["sessions"],
            "native_turns": native["coverage"]["turns"],
            "native_memories": native["coverage"]["memories"],
            "native_original_memory_construction_seconds": original_native_build_seconds,
            "native_cache_load_seconds_for_this_query_run": native["timing"][
                "memory_construction_seconds"
            ],
            "native_loaded_cache_for_this_query_run": native["coverage"][
                "loaded_memory_cache"
            ],
            "adapter_sessions": adapter_unit["session_count"],
            "adapter_chunk_count": adapter_unit["chunk_count"],
            "adapter_chunk_size_config": adapter_unit["chunk_size_config"],
            "adapter_total_stored_chunks": adapter_unit["total_stored_chunks"],
            "adapter_memory_construction_seconds": adapter_unit["total_time"],
        },
        "interpretation": (
            "The reader, raw query, top-k, model, embedding model, dataset, and metric "
            "are matched. Ingestion is intentionally not identical: native stores one "
            "A-MEM note per dialogue turn, while the MedMemoryBench adapter batches sessions "
            "using the dataset memory_chunk_size. Therefore score agreement is strong "
            "cross-protocol evidence, not proof of protocol identity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--adapter-query", type=Path, required=True)
    parser.add_argument("--adapter-memory", type=Path, required=True)
    parser.add_argument("--native-initial-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = compare(
        args.native,
        args.adapter_query,
        args.adapter_memory,
        native_initial_log=args.native_initial_log,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
