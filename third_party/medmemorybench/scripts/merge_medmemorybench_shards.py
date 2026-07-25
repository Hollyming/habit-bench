#!/usr/bin/env python3
"""Strictly merge independent-persona MedMemoryBench result shards."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_one(root: Path, suffix: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(root.rglob(f"*_{suffix}.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one *_{suffix}.json below {root}, found {matches}")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def _validate_identity(payloads: list[dict[str, Any]]) -> tuple[str, str, str]:
    fields = ("method_name", "model_name", "dataset_name")
    identity = tuple(payloads[0][field] for field in fields)
    for payload in payloads[1:]:
        candidate = tuple(payload[field] for field in fields)
        if candidate != identity:
            raise ValueError(f"Shard identity mismatch: {identity} != {candidate}")
    return identity  # type: ignore[return-value]


def _normalized_shard_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only fields that are expected to differ between persona shards."""
    normalized = copy.deepcopy(config)
    normalized.pop("total_personas", None)
    evaluation = normalized.get("dataset_config", {}).get("evaluation", {})
    evaluation.pop("persona_ids", None)
    return normalized


def _validate_config_identity(payloads: list[dict[str, Any]]) -> None:
    expected = _normalized_shard_config(payloads[0]["config"])
    for index, payload in enumerate(payloads[1:], start=2):
        candidate = _normalized_shard_config(payload["config"])
        if candidate != expected:
            raise ValueError(
                "Shard config mismatch after excluding persona-only fields: "
                f"shard 1 != shard {index}"
            )


def _query_summary(queries: list[dict[str, Any]], memory_seconds: float) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_type[query["query_type"]].append(query)

    type_summary: dict[str, Any] = {}
    for query_type, rows in sorted(by_type.items()):
        total = len(rows)
        correct = sum(bool(row["is_correct"]) for row in rows)
        stats: dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "avg_score": sum(float(row["score"]) for row in rows) / total,
        }
        if query_type == "multi_hop_clinical_deduction":
            details = [row.get("evaluation_details", {}) for row in rows]
            for source, target in (
                ("ncr_score", "avg_ncr"),
                ("crc_score", "avg_crc"),
                ("cc_score", "avg_cc"),
            ):
                stats[target] = sum(float(detail.get(source, 0.0)) for detail in details) / total
            node_rows = [
                node
                for detail in details
                for node in detail.get("node_validations", [])
            ]
            if node_rows:
                stats["node_mention_rate"] = sum(bool(node.get("mentioned")) for node in node_rows) / len(node_rows)
                stats["node_causal_rate"] = sum(bool(node.get("causal_link_correct")) for node in node_rows) / len(node_rows)
                stats["total_nodes_validated"] = len(node_rows)
        type_summary[query_type] = stats

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
        "by_type": type_summary,
        "efficiency": {
            "total_memory_construction_time": memory_seconds,
            "total_query_time": query_seconds,
            "avg_memory_construction_time": memory_seconds / total if total else 0.0,
            "avg_query_time": query_seconds / total if total else 0.0,
        },
    }


def _merge_usage(result_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for phase in ("memorize_phase", "query_phase", "total"):
        rows = [payload.get("llm_usage", {}).get(phase, {}) for payload in result_payloads]
        calls = sum(int(row.get("call_count", 0)) for row in rows)
        latency = sum(float(row.get("total_latency", 0.0)) for row in rows)
        merged[phase] = {
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
            "call_count": calls,
            "total_latency": latency,
            "avg_latency": latency / calls if calls else 0.0,
        }
    return merged


def merge(shard_dirs: list[Path], output_dir: Path, expected_queries: int | None) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("At least one shard directory is required")

    result_files, result_payloads = zip(*(_read_one(path, "result") for path in shard_dirs))
    query_files, query_payloads = zip(*(_read_one(path, "query_answer") for path in shard_dirs))
    memory_files, memory_payloads = zip(*(_read_one(path, "memory_build") for path in shard_dirs))
    method, model, dataset = _validate_identity(list(result_payloads))
    _validate_config_identity(list(result_payloads))
    _validate_identity(list(query_payloads))
    _validate_identity(list(memory_payloads))

    queries: list[dict[str, Any]] = []
    seen_queries: set[tuple[int, str]] = set()
    by_context: dict[str, Any] = {}
    for payload in query_payloads:
        ordered_keys: list[tuple[int, str]] = []
        for context_id, context in payload.get("by_context", {}).items():
            if context_id in by_context:
                raise ValueError(f"Duplicate context/persona across shards: {context_id}")
            by_context[context_id] = context
            ordered_keys.extend(
                (int(context_id), str(query_id))
                for query_id in context.get("query_ids", [])
            )
        payload_queries = payload.get("queries", [])
        if len(ordered_keys) != len(payload_queries):
            raise ValueError(
                "Cannot recover persona IDs: by_context/query list length mismatch "
                f"({len(ordered_keys)} != {len(payload_queries)})"
            )
        explicit_contexts = all(query.get("context_id") is not None for query in payload_queries)
        expected_key_set = set(ordered_keys)
        for query, recovered_key in zip(payload_queries, ordered_keys):
            query_id = str(query["query_id"])
            context_id, expected_query_id = recovered_key
            if explicit_contexts:
                context_id = int(query["context_id"])
                if (context_id, query_id) not in expected_key_set:
                    raise ValueError(
                        "Explicit persona/query pair is absent from by_context: "
                        f"{(context_id, query_id)}"
                    )
            elif query_id != expected_query_id:
                raise ValueError(
                    "Cannot recover persona IDs: query order differs from by_context "
                    f"({query_id} != {expected_query_id})"
                )
            composite_key = (context_id, query_id)
            if composite_key in seen_queries:
                raise ValueError(f"Duplicate persona/query pair across shards: {composite_key}")
            seen_queries.add(composite_key)
            query_with_context = dict(query)
            query_with_context["context_id"] = context_id
            queries.append(query_with_context)

    queries.sort(key=lambda row: (int(row["context_id"]), str(row["query_id"])))
    if expected_queries is not None and len(queries) != expected_queries:
        raise ValueError(f"Coverage mismatch: expected {expected_queries}, found {len(queries)}")

    units = [unit for payload in memory_payloads for unit in payload.get("units", [])]
    units.sort(key=lambda row: (int(row.get("context_id", -1)), str(row.get("unit_id", ""))))
    memory_seconds = sum(float(unit.get("total_time", 0.0)) for unit in units)
    summary = _query_summary(queries, memory_seconds)

    method_stats: dict[str, dict[str, float | int]] = {}
    for payload in result_payloads:
        for name, row in payload.get("memory_build_summary", {}).get("by_method", {}).items():
            target = method_stats.setdefault(name, {"count": 0, "time_cost": 0.0, "passages": 0})
            target["count"] += int(row.get("count", 0))
            target["time_cost"] += float(row.get("time_cost", 0.0))
            target["passages"] += int(row.get("passages", 0))
    build_summary = {
        "total_units": len(units),
        "total_sessions": sum(int(unit.get("session_count", 0)) for unit in units),
        "total_time": memory_seconds,
        "avg_time_per_session": 0.0,
        "total_passages": sum(int(unit.get("total_passages", 0)) for unit in units),
        "total_memory_entries": sum(
            len(session.get("memory_entries", []))
            for unit in units
            for session in unit.get("session_builds", [])
        ),
        "by_method": method_stats,
    }
    if build_summary["total_sessions"]:
        build_summary["avg_time_per_session"] = memory_seconds / build_summary["total_sessions"]

    config = copy.deepcopy(result_payloads[0]["config"])
    persona_ids = sorted(int(context_id) for context_id in by_context)
    config["total_personas"] = len(persona_ids)
    config.setdefault("dataset_config", {}).setdefault("evaluation", {})["persona_ids"] = persona_ids
    duration = sum(float(payload.get("duration_seconds", 0.0)) for payload in result_payloads)
    merged_result = {
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "start_time": min(str(payload["start_time"]) for payload in result_payloads),
        "end_time": max(str(payload["end_time"]) for payload in result_payloads),
        "duration_seconds": duration,
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
                sum(int(query.get("retrieved_count", 0)) for query in queries) / len(queries)
                if queries else 0.0
            ),
        },
        "by_context": dict(sorted(by_context.items(), key=lambda item: int(item[0]))),
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
        "result": output_dir / "medmemorybench_merged_result.json",
        "query_answer": output_dir / "medmemorybench_merged_query_answer.json",
        "memory_build": output_dir / "medmemorybench_merged_memory_build.json",
    }
    for name, payload in (
        ("result", merged_result),
        ("query_answer", merged_queries),
        ("memory_build", merged_memory),
    ):
        outputs[name].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "contract": "medmemorybench.independent_shard_merge.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_name": method,
        "model_name": model,
        "dataset_name": dataset,
        "persona_ids": persona_ids,
        "query_count": len(queries),
        "shards": [
            {
                "root": str(root.resolve()),
                "result": str(result.resolve()),
                "query_answer": str(query.resolve()),
                "memory_build": str(memory.resolve()),
            }
            for root, result, query, memory in zip(shard_dirs, result_files, query_files, memory_files)
        ],
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
