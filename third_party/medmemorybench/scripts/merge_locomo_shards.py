#!/usr/bin/env python3
"""Strictly merge independent-sample LoCoMo result shards."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_medmemorybench_shards import (
    _merge_usage,
    _read_one,
    _validate_identity,
)


def _query_summary(queries: list[dict[str, Any]], memory_seconds: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        grouped[str(row["query_type"])].append(row)

    by_type = {}
    for query_type, rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(bool(row["is_correct"]) for row in rows)
        by_type[query_type] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "avg_score": sum(float(row["score"]) for row in rows) / total,
        }

    total = len(queries)
    correct = sum(bool(row["is_correct"]) for row in queries)
    query_seconds = sum(float(row.get("query_time", 0.0)) for row in queries)
    return {
        "total_queries": total,
        "correct_count": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "overall_avg_score": (
            sum(float(row["score"]) for row in queries) / total if total else 0.0
        ),
        "by_type": by_type,
        "efficiency": {
            "total_memory_construction_time": memory_seconds,
            "total_query_time": query_seconds,
            "avg_memory_construction_time": memory_seconds / total if total else 0.0,
            "avg_query_time": query_seconds / total if total else 0.0,
        },
    }


def _query_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    query_id = str(row["query_id"])
    try:
        ordinal = int(query_id.rsplit("_q", 1)[1])
    except (IndexError, ValueError):
        ordinal = -1
    return str(row["context_id"]), ordinal, query_id


def merge(shard_dirs: list[Path], output_dir: Path, expected_queries: int | None) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("At least one shard directory is required")

    result_files, result_payloads = zip(*(_read_one(path, "result") for path in shard_dirs))
    query_files, query_payloads = zip(*(_read_one(path, "query_answer") for path in shard_dirs))
    memory_files, memory_payloads = zip(*(_read_one(path, "memory_build") for path in shard_dirs))
    method, model, dataset = _validate_identity(list(result_payloads))
    _validate_identity(list(query_payloads))
    _validate_identity(list(memory_payloads))
    if dataset != "locomo":
        raise ValueError(f"Expected LoCoMo shards, found dataset={dataset!r}")

    queries: list[dict[str, Any]] = []
    by_context: dict[str, Any] = {}
    seen_query_ids: set[str] = set()
    for payload in query_payloads:
        local_contexts = payload.get("by_context", {})
        overlap = set(by_context).intersection(local_contexts)
        if overlap:
            raise ValueError(f"Duplicate LoCoMo samples across shards: {sorted(overlap)}")
        by_context.update(local_contexts)
        allowed_pairs = {
            (str(context_id), str(query_id))
            for context_id, context in local_contexts.items()
            for query_id in context.get("query_ids", [])
        }
        for row in payload.get("queries", []):
            query_id = str(row["query_id"])
            context_id = str(row.get("context_id", ""))
            if (context_id, query_id) not in allowed_pairs:
                raise ValueError(
                    f"Query is absent from shard by_context manifest: {(context_id, query_id)}"
                )
            if query_id in seen_query_ids:
                raise ValueError(f"Duplicate LoCoMo query ID across shards: {query_id}")
            seen_query_ids.add(query_id)
            queries.append(dict(row, context_id=context_id))

    queries.sort(key=_query_sort_key)
    if expected_queries is not None and len(queries) != expected_queries:
        raise ValueError(f"Coverage mismatch: expected {expected_queries}, found {len(queries)}")

    units = [unit for payload in memory_payloads for unit in payload.get("units", [])]
    unit_contexts = [str(unit.get("context_id", "")) for unit in units]
    if len(unit_contexts) != len(set(unit_contexts)):
        raise ValueError("Duplicate LoCoMo memory-build sample across shards")
    if set(unit_contexts) != set(by_context):
        raise ValueError(
            f"Memory/query sample mismatch: memory={sorted(unit_contexts)}, "
            f"query={sorted(by_context)}"
        )
    units.sort(key=lambda unit: str(unit.get("context_id", "")))
    memory_seconds = sum(float(unit.get("total_time", 0.0)) for unit in units)
    summary = _query_summary(queries, memory_seconds)
    build_summary = {
        "total_units": len(units),
        "total_sessions": sum(int(unit.get("session_count", 0)) for unit in units),
        "total_memory_chunks": sum(int(unit.get("chunk_count", 0)) for unit in units),
        "total_time": memory_seconds,
        "total_entries": sum(int(unit.get("total_entries", 0)) for unit in units),
        "total_stored_chunks": sum(
            int(unit.get("total_stored_chunks", 0)) for unit in units
        ),
        "avg_time_per_unit": memory_seconds / len(units) if units else 0.0,
    }

    sample_ids = sorted(by_context)
    config = copy.deepcopy(result_payloads[0].get("config", {}))
    config.setdefault("dataset_config", {}).setdefault("evaluation", {})[
        "sample_ids"
    ] = sample_ids
    merged_result = {
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "start_time": min(str(payload["start_time"]) for payload in result_payloads),
        "end_time": max(str(payload["end_time"]) for payload in result_payloads),
        "duration_seconds": sum(
            float(payload.get("duration_seconds", 0.0)) for payload in result_payloads
        ),
        "summary": {key: value for key, value in summary.items() if key != "efficiency"},
        "efficiency": summary["efficiency"],
        "memory_build_summary": build_summary,
        "llm_usage": _merge_usage(list(result_payloads)),
        "config": config,
    }
    merged_queries = {
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "summary": {
            "total_queries": len(queries),
            "correct_count": summary["correct_count"],
            "total_query_time": summary["efficiency"]["total_query_time"],
            "avg_query_time": summary["efficiency"]["avg_query_time"],
            "avg_retrieved_count": (
                sum(int(row.get("retrieved_count", 0)) for row in queries) / len(queries)
                if queries else 0.0
            ),
        },
        "by_context": dict(sorted(by_context.items())),
        "queries": queries,
    }
    merged_memory = {
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "summary": build_summary,
        "memory_chunk_size": memory_payloads[0].get("memory_chunk_size"),
        "total_units": len(units),
        "units": units,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "result": output_dir / "locomo_merged_result.json",
        "query_answer": output_dir / "locomo_merged_query_answer.json",
        "memory_build": output_dir / "locomo_merged_memory_build.json",
    }
    for name, payload in (
        ("result", merged_result),
        ("query_answer", merged_queries),
        ("memory_build", merged_memory),
    ):
        outputs[name].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "contract": "medmemorybench.locomo_sample_shard_merge.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "sample_ids": sample_ids,
        "query_count": len(queries),
        "shards": [
            {
                "root": str(root.resolve()),
                "result": str(result.resolve()),
                "query_answer": str(query.resolve()),
                "memory_build": str(memory.resolve()),
            }
            for root, result, query, memory in zip(
                shard_dirs, result_files, query_files, memory_files
            )
        ],
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(merge(args.shard_dir, args.output_dir, args.expected_queries), indent=2))
