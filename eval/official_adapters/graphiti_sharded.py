#!/usr/bin/env python
"""Run Graphiti user shards concurrently and merge retrieved memory contexts."""

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


def run(args: argparse.Namespace) -> None:
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")

    payload = read_json(args.input)
    probe_order = [probe["probe_id"] for probe in payload["probes"]]
    work_dir = args.output.parent / "graphiti_shards"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    processes = []
    for shard_index in range(args.shard_count):
        shard_dir = work_dir / f"shard_{shard_index:02d}_of_{args.shard_count:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_output = shard_dir / "raw_predictions.jsonl"
        command = [
            sys.executable,
            "-m",
            "eval.official_adapters.graphiti",
            "--input",
            str(args.input),
            "--output",
            str(shard_output),
            "--topk",
            str(args.topk),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(args.shard_count),
        ]
        processes.append((shard_index, shard_dir, shard_output, command, subprocess.Popen(command)))

    failures = []
    for shard_index, shard_dir, shard_output, command, process in processes:
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
        raise RuntimeError(f"Graphiti shards failed: {failures}")

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    shard_runtimes = []
    shard_configs = []
    for shard_index, shard_dir, shard_output, _, _ in processes:
        for row in read_jsonl(shard_output):
            probe_id = row["probe_id"]
            if probe_id in predictions_by_probe:
                raise RuntimeError(f"Duplicate Graphiti prediction for {probe_id}")
            predictions_by_probe[probe_id] = row
        runtime_path = shard_dir / "graphiti_runtime.json"
        config_path = shard_dir / "graphiti_config.json"
        shard_runtimes.append(read_json(runtime_path))
        shard_configs.append(read_json(config_path))

    missing = [probe_id for probe_id in probe_order if probe_id not in predictions_by_probe]
    extra = sorted(set(predictions_by_probe) - set(probe_order))
    if missing or extra:
        raise RuntimeError(f"Graphiti shard coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    merged_config = dict(shard_configs[0])
    merged_config.update(
        {
            "sharded": True,
            "shard_count": args.shard_count,
            "shard_work_dir": str(work_dir),
            "db_path": "per-shard; see shard_work_dir",
        }
    )
    write_json(args.output.parent / "graphiti_config.json", merged_config)

    add_stats_rows = [runtime.get("add_stats", {}) for runtime in shard_runtimes]
    merged_runtime = {
        "sharded": True,
        "shard_count": args.shard_count,
        "elapsed_sec": round(time.time() - started, 3),
        "total_predictions": len(predictions_by_probe),
        "add_stats": {
            "episodes_attempted": sum(row.get("episodes_attempted", 0) for row in add_stats_rows),
            "episodes_added": sum(row.get("episodes_added", 0) for row in add_stats_rows),
            "add_failure_count": sum(row.get("add_failure_count", 0) for row in add_stats_rows),
            "add_elapsed_sec_sum": round(sum(row.get("add_elapsed_sec", 0.0) for row in add_stats_rows), 3),
        },
        "shard_runtimes": shard_runtimes,
    }
    write_json(args.output.parent / "graphiti_runtime.json", merged_runtime)
    write_jsonl(args.output, [predictions_by_probe[probe_id] for probe_id in probe_order])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("HABITBENCH_GRAPHITI_SHARDS", "4")))
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
