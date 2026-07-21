#!/usr/bin/env python
"""Run O-Mem on user-disjoint shards and merge retrieved memory contexts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sum_memory_llm_stats(runtimes: List[Dict[str, Any]]) -> Dict[str, int]:
    fields = [
        "calls_attempted",
        "calls_succeeded",
        "invalid_json_responses",
        "raw_invalid_json_responses",
        "transport_failures",
        "json_schema_calls",
        "json_object_calls",
        "schema_contract_failures",
        "topic_merge_calls",
        "topic_merge_repairs",
        "topic_merge_parse_fallbacks",
        "topic_merge_unresolved",
    ]
    return {
        field: sum(runtime.get("memory_llm_stats", {}).get(field, 0) for runtime in runtimes)
        for field in fields
    }


def run(args: argparse.Namespace) -> None:
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")

    payload = read_json(args.input)
    probe_order = [probe["probe_id"] for probe in payload["probes"]]
    users = sorted({probe["user_id"] for probe in payload["probes"]})
    work_dir = args.output.parent / "omem_shards"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    processes = []
    for shard_index in range(args.shard_count):
        shard_users = {
            user_id
            for user_index, user_id in enumerate(users)
            if user_index % args.shard_count == shard_index
        }
        shard_dir = work_dir / f"shard_{shard_index:02d}_of_{args.shard_count:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_input = shard_dir / "official_input.json"
        shard_output = shard_dir / "raw_predictions.jsonl"
        shard_payload = {
            **payload,
            "probes": [probe for probe in payload["probes"] if probe["user_id"] in shard_users],
            "sessions_by_user": {
                user_id: payload["sessions_by_user"][user_id] for user_id in sorted(shard_users)
            },
        }
        write_json(shard_input, shard_payload)
        command = [
            sys.executable,
            "-m",
            "eval.official_adapters.omem",
            "--input",
            str(shard_input),
            "--output",
            str(shard_output),
            "--topn",
            str(args.topn),
            "--drop-threshold",
            str(args.drop_threshold),
        ]
        processes.append((shard_index, shard_dir, shard_output, command, subprocess.Popen(command)))

    failures = []
    for shard_index, _, shard_output, command, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append(
                {
                    "shard_index": shard_index,
                    "returncode": returncode,
                    "command": command,
                    "output": str(shard_output),
                }
            )
    if failures:
        write_json(work_dir / "shard_failures.json", {"failures": failures})
        raise RuntimeError(f"O-Mem shards failed: {failures}")

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    shard_runtimes = []
    shard_configs = []
    for _, shard_dir, shard_output, _, _ in processes:
        for row in read_jsonl(shard_output):
            probe_id = row["probe_id"]
            if probe_id in predictions_by_probe:
                raise RuntimeError(f"Duplicate O-Mem prediction for {probe_id}")
            predictions_by_probe[probe_id] = row
        shard_runtimes.append(read_json(shard_dir / "omem_runtime.json"))
        shard_configs.append(read_json(shard_dir / "omem_config.json"))

    missing = [probe_id for probe_id in probe_order if probe_id not in predictions_by_probe]
    extra = sorted(set(predictions_by_probe) - set(probe_order))
    if missing or extra:
        raise RuntimeError(f"O-Mem shard coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    merged_config = dict(shard_configs[0])
    merged_config.update(
        {
            "sharded": True,
            "shard_count": args.shard_count,
            "shard_work_dir": str(work_dir),
        }
    )
    write_json(args.output.parent / "omem_config.json", merged_config)

    merged_runtime = {
        "sharded": True,
        "shard_count": args.shard_count,
        "elapsed_sec": round(time.time() - started, 3),
        "messages_added": sum(runtime.get("messages_added", 0) for runtime in shard_runtimes),
        "total_predictions": len(predictions_by_probe),
        "message_understanding_enabled": True,
        "persona_update_enabled": True,
        "memory_llm_json_output_contract": True,
        "memory_llm_stats": sum_memory_llm_stats(shard_runtimes),
        "shard_runtimes": shard_runtimes,
    }
    write_json(args.output.parent / "omem_runtime.json", merged_runtime)
    write_jsonl(args.output, [predictions_by_probe[probe_id] for probe_id in probe_order])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=int(os.getenv("HABITBENCH_OMEM_SHARDS", "4")),
    )
    parser.add_argument("--topn", type=int, default=12)
    parser.add_argument("--drop-threshold", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
