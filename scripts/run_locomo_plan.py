#!/usr/bin/env python3
"""Run a resumable LoCoMo plan on one H-cluster Replica.

Each local H200 gets one Qwen3-8B vLLM server.  A shared GPFS atomic-directory
queue lets two 8-GPU Replicas process method/sample units dynamically.  The
actual benchmark logic remains the vendored MedMemoryBench LoCoMo evaluator;
this launcher only handles scheduling, isolation, logging, and resumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multigpu_plan import (  # noqa: E402
    DistributedTaskCoordinator,
    _benchmark_server,
    _public_worker_record,
    _source_env_file,
    _start_server,
    _stop_server,
    _utc_now,
    _write_json,
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"LoCoMo plan is empty: {path}")
    required = {
        "task_id",
        "method",
        "dataset_name",
        "sample_id",
        "dataset_file",
        "method_config",
        "output_dir",
        "shard_index",
        "shard_count",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"LoCoMo plan is missing columns: {sorted(missing)}")
    expected_ids = list(range(len(rows)))
    actual_ids = [int(row["task_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("LoCoMo task_id values must be contiguous and ordered")
    for row in rows:
        if row["dataset_name"] != "locomo":
            raise ValueError(f"Unexpected dataset in LoCoMo plan: {row['dataset_name']}")
    return rows


def _task_marker(row: dict[str, str]) -> Path:
    return Path(row["output_dir"]).expanduser().resolve() / "locomo_task_result.json"


def _is_completed(row: dict[str, str]) -> bool:
    marker = _task_marker(row)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "succeeded"


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] command={' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def _run_task(
    row: dict[str, str],
    worker: dict[str, Any],
    base_env: dict[str, str],
    attempts: int,
) -> dict[str, Any]:
    output_dir = Path(row["output_dir"]).expanduser().resolve()
    label = f"{row['method']}/{row['sample_id']}"
    if _is_completed(row):
        print(f"{_utc_now()} locomo_task_skip task={label}", flush=True)
        return {
            "status": "skipped_completed",
            "checkpoint_status": "succeeded",
            "task_id": int(row["task_id"]),
            "method": row["method"],
            "sample_id": row["sample_id"],
            "output_dir": str(output_dir),
            "finished_at": _utc_now(),
        }

    task_env = dict(base_env)
    task_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            # vLLM owns the worker GPU; BGE-M3 and method-side state use CPU.
            "CUDA_VISIBLE_DEVICES": "",
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": worker["base_url"],
            "OMP_NUM_THREADS": base_env.get("HABITBENCH_ADAPTER_CPU_THREADS", "2"),
            "MKL_NUM_THREADS": base_env.get("HABITBENCH_ADAPTER_CPU_THREADS", "2"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command_base = [
        str(Path(base_env.get("PYTHON_BIN", sys.executable)).expanduser()),
        str(PROJECT_ROOT / "scripts/run_locomo_task.py"),
        "--method",
        row["method"],
        "--sample-id",
        row["sample_id"],
        "--dataset-file",
        row["dataset_file"],
        "--output-dir",
        str(output_dir),
        "--base-url",
        worker["base_url"],
        "--model-name",
        base_env.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
        "--llm-model-path",
        base_env.get("HABITBENCH_LLM_MODEL", "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/Qwen3-8B"),
        "--embedding-model-path",
        base_env.get("HABITBENCH_EMBED_MODEL", "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/bge-m3"),
        "--med-repo",
        base_env.get("HABITBENCH_MEDMEMORYBENCH_ROOT", str(PROJECT_ROOT / "third_party/medmemorybench")),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "task.stdout.log"
    started = time.perf_counter()
    last_returncode: int | None = None
    # A retry coordinator reuses the same method/sample output directories so
    # successful results remain in place. Failed directories can contain
    # partial state from an earlier attempt; offset the attempt number so the
    # evaluator gets a fresh ``attempt-N`` state directory instead of
    # accidentally reopening that partial state.
    prior_failure_offset = 0
    prior_marker = _task_marker(row)
    if prior_marker.is_file():
        try:
            prior_payload = json.loads(prior_marker.read_text(encoding="utf-8"))
            if prior_payload.get("status") == "failed":
                prior_failure_offset = max(int(prior_payload.get("attempt", 1)), 1)
        except (OSError, ValueError, json.JSONDecodeError):
            prior_failure_offset = 1
    for attempt in range(1, attempts + 1):
        effective_attempt = attempt + prior_failure_offset
        command = command_base + ["--attempt", str(effective_attempt)]
        print(
            f"{_utc_now()} locomo_task_start task={label} worker={worker['worker_index']} gpu={worker['gpu']} attempt={effective_attempt} ({attempt}/{attempts})",
            flush=True,
        )
        last_returncode = _run_command(command, cwd=PROJECT_ROOT, env=task_env, log_path=log_path)
        if last_returncode == 0 and _is_completed(row):
            return {
                "status": "succeeded",
                "task_id": int(row["task_id"]),
                "method": row["method"],
                "sample_id": row["sample_id"],
                "output_dir": str(output_dir),
                "worker_index": int(worker["worker_index"]),
                "gpu": worker["gpu"],
                "attempt": effective_attempt,
                "returncode": 0,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "finished_at": _utc_now(),
            }
        print(
            f"{_utc_now()} locomo_task_attempt_failed task={label} attempt={attempt} returncode={last_returncode}",
            file=sys.stderr,
            flush=True,
        )
    marker = _task_marker(row)
    error = "task subprocess failed without a successful completion marker"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            error = str(payload.get("error", error))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "status": "failed",
        "task_id": int(row["task_id"]),
        "method": row["method"],
        "sample_id": row["sample_id"],
        "output_dir": str(output_dir),
        "worker_index": int(worker["worker_index"]),
        "gpu": worker["gpu"],
        "attempt": attempts + prior_failure_offset,
        "returncode": last_returncode,
        "error": error,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "finished_at": _utc_now(),
    }


def _aggregate_completed(rows: list[dict[str, str]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        marker = _task_marker(row)
        if marker.is_file():
            try:
                records.append(json.loads(marker.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                records.append({"status": "invalid", "method": row["method"], "sample_id": row["sample_id"]})
        else:
            records.append({"status": "missing", "method": row["method"], "sample_id": row["sample_id"]})
    by_method: dict[str, dict[str, Any]] = {}
    for record in records:
        method = str(record.get("method", "unknown"))
        method_out = by_method.setdefault(method, {"tasks": 0, "succeeded": 0, "mean_official_f1": 0.0, "query_count": 0})
        method_out["tasks"] += 1
        if record.get("status") == "succeeded":
            method_out["succeeded"] += 1
            official = record.get("official_locomo", {})
            summary = (official.get("summary") or {}) if isinstance(official, dict) else {}
            count = int(summary.get("total", 0) or 0)
            method_out["mean_official_f1"] += float(summary.get("mean_official_score", 0.0) or 0.0) * count
            method_out["query_count"] += count
    for item in by_method.values():
        if item["query_count"]:
            item["mean_official_f1"] /= item["query_count"]
    return {
        "contract_version": "habitbench.locomo_suite_summary.v1",
        "dataset_name": "locomo",
        "task_count": len(rows),
        "succeeded": sum(1 for record in records if record.get("status") == "succeeded"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
        "missing_or_invalid": sum(1 for record in records if record.get("status") not in {"succeeded", "failed"}),
        "methods": by_method,
        "tasks": records,
        "created_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated local GPU IDs")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--coordination-root", type=Path)
    parser.add_argument("--coordinator-id")
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--task-attempts", type=int, default=int(os.environ.get("HABITBENCH_LOCOMO_TASK_ATTEMPTS", "2")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.task_attempts < 1:
        raise ValueError("--task-attempts must be positive")
    plan = args.plan.expanduser().resolve()
    rows = _load_plan(plan)
    gpus = _split_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if not (0 <= args.replica_index < args.replica_count):
        raise ValueError("invalid replica index/count")
    if args.replica_count > 1 and not args.coordinator_id:
        raise ValueError("multi-Replica LoCoMo runs require --coordinator-id")

    base_env = _source_env_file(args.env_file.expanduser().resolve(), os.environ.copy())
    base_env["HABITBENCH_INFERENCE_BACKEND"] = "local-vllm"
    base_env["OPENAI_API_KEY"] = "dummy"
    base_env["PYTHONUNBUFFERED"] = "1"
    coordinator_id = args.coordinator_id or f"manual-{os.getpid()}"
    coordination_root = (args.coordination_root or plan.parent / "locomo_queue").expanduser().resolve()
    coordinator = DistributedTaskCoordinator(
        coordination_root,
        rows,
        coordinator_id=coordinator_id,
        plan_sha256=__import__("hashlib").sha256(plan.read_bytes()).hexdigest(),
        replica_count=args.replica_count,
    )
    runtime_path = (args.runtime_output or plan.parent / f"locomo_runtime.replica-{args.replica_index:03d}.json").expanduser().resolve()
    log_root = (args.log_root or plan.parent / "locomo_vllm_logs" / coordinator_id / f"replica-{args.replica_index:03d}").expanduser().resolve()
    runtime: dict[str, Any] = {
        "contract_version": "habitbench.locomo_runtime.v1",
        "status": "starting",
        "started_at": _utc_now(),
        "finished_at": None,
        "host": socket.gethostname(),
        "plan": str(plan),
        "replica_index": args.replica_index,
        "replica_count": args.replica_count,
        "gpus": gpus,
        "total_gpus": len(gpus) * args.replica_count,
        "task_count": len(rows),
        "task_attempts": args.task_attempts,
        "queue_root": str(coordinator.root),
        "servers": [],
    }
    _write_json(runtime_path, runtime)
    reusable = sum(_is_completed(row) for row in rows)
    print(f"{_utc_now()} locomo_suite_start replica={args.replica_index}/{args.replica_count} tasks={len(rows)} reusable={reusable} pending={len(rows)-reusable}", flush=True)

    workers: list[dict[str, Any]] = []
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(
                    _start_server,
                    index,
                    gpu,
                    args.port_base + index,
                    base_env,
                    log_root,
                )
                for index, gpu in enumerate(gpus)
            ]
            for future in as_completed(futures):
                worker = future.result()
                # Register the process before the throughput check so a
                # failing benchmark gate is still cleaned up in ``finally``.
                workers.append(worker)
                worker["throughput_gate"] = _benchmark_server(worker, base_env)
                if worker["throughput_gate"].get("status") != "pass":
                    raise RuntimeError(f"vLLM throughput gate failed for worker {worker['worker_index']}: {worker['throughput_gate']}")
                print(f"{_utc_now()} locomo_vllm_ready worker={worker['worker_index']} gpu={worker['gpu']} port={worker['port']}", flush=True)
        workers.sort(key=lambda item: int(item["worker_index"]))
        runtime["servers"] = [_public_worker_record(worker) for worker in workers]
        runtime["status"] = "running"
        _write_json(runtime_path, runtime)

        def worker_loop(worker: dict[str, Any]) -> int:
            local_failures = 0
            while True:
                claimed = coordinator.claim_next(
                    replica_index=args.replica_index,
                    worker_index=int(worker["worker_index"]),
                    host=socket.gethostname(),
                )
                if claimed is None:
                    return local_failures
                row, claim = claimed
                result = _run_task(row, worker, base_env, args.task_attempts)
                result["distributed_claim"] = claim
                if result.get("status") == "failed":
                    local_failures += 1
                state = coordinator.finish(claim, result)
                print(
                    f"{_utc_now()} locomo_queue_progress finished={state['finished']} total={state['task_count']} succeeded={state['succeeded']} failed={state['failed']}",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            failures = sum(future.result() for future in as_completed([executor.submit(worker_loop, worker) for worker in workers]))

        # Replica 0 owns the global summary, but waits for work still running on
        # the other Replica before declaring the suite complete.
        if args.replica_index == 0:
            deadline = time.monotonic() + float(base_env.get("HABITBENCH_LOCOMO_MERGE_WAIT_SEC", "172800"))
            while coordinator.aggregate_state()["finished"] < len(rows):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for all LoCoMo task markers")
                state = coordinator.aggregate_state()
                print(f"{_utc_now()} locomo_merge_wait finished={state['finished']} total={len(rows)}", flush=True)
                time.sleep(30)
            summary = _aggregate_completed(rows)
            summary["status"] = "succeeded" if summary["failed"] == 0 and summary["missing_or_invalid"] == 0 else "failed"
            _write_json(plan.parent / "locomo_suite_summary.json", summary)
            if summary["status"] != "succeeded":
                raise RuntimeError(f"LoCoMo suite has failed tasks: {summary['failed']}")
        runtime["status"] = "succeeded" if failures == 0 else "completed_with_task_failures"
        return 0 if failures == 0 else 1
    finally:
        for worker in workers:
            _stop_server(worker)
        runtime["servers"] = [_public_worker_record(worker) for worker in workers]
        runtime["finished_at"] = _utc_now()
        runtime["task_failures"] = failures
        if runtime.get("status") == "starting":
            runtime["status"] = "failed"
        _write_json(runtime_path, runtime)
        print(f"{_utc_now()} locomo_suite_finished replica={args.replica_index}/{args.replica_count} status={runtime['status']} failures={failures}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
