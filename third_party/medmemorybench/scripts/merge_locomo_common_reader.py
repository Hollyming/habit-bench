#!/usr/bin/env python3
"""Strictly merge common-reader LoCoMo query shards."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rescore_locomo_common_reader import summarize


def merge(shards: list[Path], output: Path, expected_total: int) -> dict[str, Any]:
    if not shards:
        raise ValueError("At least one shard is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    if any(item.get("contract") != "medmemorybench.locomo_common_reader.v1" for item in payloads):
        raise ValueError("Unexpected common-reader contract")
    if any(item.get("complete") is not True for item in payloads):
        raise ValueError("All common-reader shards must be complete")

    source_hashes = {item.get("source", {}).get("sha256") for item in payloads}
    source_totals = {item.get("source", {}).get("total_queries") for item in payloads}
    methods = {item.get("method_name") for item in payloads}
    if len(source_hashes) != 1 or source_totals != {expected_total} or len(methods) != 1:
        raise ValueError("Source, total-query, or method identity mismatch")

    configs = [dict(item["common_reader"]["config"]) for item in payloads]
    shard_count = int(configs[0].pop("shard_count"))
    indices = {int(config.pop("shard_index")) for config in configs}
    for config in configs[1:]:
        config.pop("shard_count", None)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Common-reader config mismatch")
    if shard_count != len(shards) or indices != set(range(shard_count)):
        raise ValueError(
            f"Shard coverage mismatch: count={shard_count}, indices={sorted(indices)}"
        )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_path = Path(payloads[0]["source"]["path"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_ids = [str(row["query_id"]) for row in source_payload.get("queries", [])]
    if len(source_ids) != expected_total or len(source_ids) != len(set(source_ids)):
        raise ValueError("Source query order is incomplete or duplicated")
    order = {query_id: index for index, query_id in enumerate(source_ids)}
    for payload in payloads:
        for row in payload.get("queries", []):
            query_id = str(row["query_id"])
            if query_id in seen:
                raise ValueError(f"Duplicate query ID: {query_id}")
            if query_id not in order:
                raise ValueError(f"Unknown query ID: {query_id}")
            seen.add(query_id)
            rows.append(row)
    if len(rows) != expected_total or seen != set(source_ids):
        raise ValueError(f"Query coverage mismatch: expected {expected_total}, found {len(rows)}")
    rows.sort(key=lambda row: order[str(row["query_id"])])

    merged_config = dict(configs[0])
    merged_config["shard_count"] = shard_count
    result = {
        "contract": "medmemorybench.locomo_common_reader_merge.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "dataset_name": "locomo",
        "method_name": next(iter(methods)),
        "source": payloads[0]["source"],
        "common_reader": {
            "config": merged_config,
            "elapsed_seconds_sum": sum(
                float(item["common_reader"].get("elapsed_seconds", 0.0))
                for item in payloads
            ),
            "summary": summarize(rows),
        },
        "shards": [str(path.resolve()) for path in shards],
        "queries": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output.resolve()),
        "queries": len(rows),
        "summary": result["common_reader"]["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(merge(args.shard, args.output, args.expected_total), indent=2))
