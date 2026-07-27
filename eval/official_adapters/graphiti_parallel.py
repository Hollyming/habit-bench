#!/usr/bin/env python
"""Run isolated Graphiti user workers concurrently and merge their contexts.

Each child keeps the official, chronological ``Graphiti.add_episode`` and
``Graphiti.search_`` lifecycle. Parallelism is only across independent users:
every child owns a separate Kuzu database while all children share the local
vLLM endpoint, allowing vLLM to batch their requests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def child_command(
    args: argparse.Namespace,
    *,
    worker_index: int,
    worker_count: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eval.official_adapters.graphiti",
        "--input",
        str(args.input),
        "--output",
        str(output),
        "--topk",
        str(args.topk),
        "--shard-index",
        str(worker_index),
        "--shard-count",
        str(worker_count),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.continue_on_add_error:
        command.append("--continue-on-add-error")
    return command


def run(args: argparse.Namespace) -> None:
    if args.user_workers < 1:
        raise ValueError("--user-workers must be positive")

    payload = read_json(args.input)
    user_count = len(payload.get("sessions_by_user", {}))
    probe_order = [probe["probe_id"] for probe in payload.get("probes", [])]
    if user_count == 0:
        write_jsonl(args.output, [])
        write_json(
            args.output.parent / "graphiti_runtime.json",
            {
                "parallel_users": True,
                "user_workers_requested": args.user_workers,
                "user_workers_effective": 0,
                "elapsed_sec": 0.0,
                "total_predictions": 0,
            },
        )
        return

    worker_count = min(args.user_workers, user_count)
    work_dir = args.output.parent / "graphiti_user_workers"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    processes: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        worker_dir = work_dir / f"worker_{worker_index:02d}_of_{worker_count:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_output = worker_dir / "memory_contexts.jsonl"
        command = child_command(
            args,
            worker_index=worker_index,
            worker_count=worker_count,
            output=worker_output,
        )
        processes.append(
            {
                "worker_index": worker_index,
                "dir": worker_dir,
                "output": worker_output,
                "command": command,
                "process": subprocess.Popen(command),
            }
        )

    failures: list[dict[str, Any]] = []
    for record in processes:
        returncode = record["process"].wait()
        record["returncode"] = returncode
        if returncode != 0:
            failures.append(
                {
                    "worker_index": record["worker_index"],
                    "returncode": returncode,
                    "command": record["command"],
                    "output": str(record["output"]),
                }
            )
    if failures:
        write_json(work_dir / "worker_failures.json", {"failures": failures})
        raise RuntimeError(f"Graphiti user workers failed: {failures}")

    predictions_by_probe: dict[str, dict[str, Any]] = {}
    worker_runtimes: list[dict[str, Any]] = []
    worker_configs: list[dict[str, Any]] = []
    for record in processes:
        for row in read_jsonl(record["output"]):
            probe_id = row["probe_id"]
            if probe_id in predictions_by_probe:
                raise RuntimeError(f"Duplicate Graphiti prediction for {probe_id}")
            predictions_by_probe[probe_id] = row
        worker_runtimes.append(
            read_json(record["dir"] / "graphiti_runtime.json")
        )
        worker_configs.append(
            read_json(record["dir"] / "graphiti_config.json")
        )

    missing = [probe_id for probe_id in probe_order if probe_id not in predictions_by_probe]
    extra = sorted(set(predictions_by_probe) - set(probe_order))
    if missing or extra:
        raise RuntimeError(
            "Graphiti parallel coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    merged_config = dict(worker_configs[0])
    merged_config.update(
        {
            "parallel_users": True,
            "user_workers_requested": args.user_workers,
            "user_workers_effective": worker_count,
            "worker_store_root": str(work_dir),
            "db_path": "per-user-worker; see worker_store_root",
        }
    )
    write_json(args.output.parent / "graphiti_config.json", merged_config)

    add_stats_rows = [runtime.get("add_stats", {}) for runtime in worker_runtimes]
    aggregate_worker_elapsed = sum(
        float(row.get("wall_elapsed_sec", 0.0)) for row in add_stats_rows
    )
    elapsed = time.perf_counter() - started
    merged_runtime = {
        "parallel_users": True,
        "user_workers_requested": args.user_workers,
        "user_workers_effective": worker_count,
        "elapsed_sec": round(elapsed, 3),
        "aggregate_worker_elapsed_sec": round(aggregate_worker_elapsed, 3),
        "parallelism_ratio": (
            round(aggregate_worker_elapsed / elapsed, 3) if elapsed else 0.0
        ),
        "total_predictions": len(predictions_by_probe),
        "add_stats": {
            "episodes_attempted": sum(
                int(row.get("episodes_attempted", 0)) for row in add_stats_rows
            ),
            "episodes_added": sum(
                int(row.get("episodes_added", 0)) for row in add_stats_rows
            ),
            "add_failure_count": sum(
                int(row.get("add_failure_count", 0)) for row in add_stats_rows
            ),
            "add_elapsed_sec_sum": round(
                sum(float(row.get("add_elapsed_sec", 0.0)) for row in add_stats_rows),
                3,
            ),
        },
        "worker_runtimes": worker_runtimes,
    }
    write_json(args.output.parent / "graphiti_runtime.json", merged_runtime)
    write_jsonl(
        args.output,
        [predictions_by_probe[probe_id] for probe_id in probe_order],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--user-workers",
        type=int,
        default=int(os.getenv("HABITBENCH_GRAPHITI_USER_WORKERS", "4")),
        help="Concurrent isolated user processes sharing one vLLM endpoint",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "25")),
    )
    parser.add_argument(
        "--continue-on-add-error",
        action="store_true",
        help="Diagnostic only; formal runs keep this disabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
