#!/usr/bin/env python
"""Merge every method/dataset group represented in a shard plan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.merge_shards import merge_shards
from eval.core.io import sha256_file, write_json
from eval.supplementary.merge_oracle import merge_oracle_shards


ORACLE_METHODS = {"oracle_evidence", "oracle_habit_state"}


def _combine_replica_runtimes(paths: list[Path]) -> dict:
    runtimes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runtimes:
        raise ValueError("No Replica runtimes were supplied")
    expected_count = int(runtimes[0].get("replica_count") or len(runtimes))
    indices = sorted(int(runtime.get("replica_index", -1)) for runtime in runtimes)
    if len(runtimes) != expected_count or indices != list(range(expected_count)):
        raise ValueError(
            f"Replica runtime coverage mismatch: expected={expected_count}, indices={indices}"
        )
    plan_hashes = {runtime.get("plan_sha256") for runtime in runtimes}
    if len(plan_hashes) != 1:
        raise ValueError(f"Replica runtime plan hashes differ: {plan_hashes}")
    configs = [runtime.get("config") for runtime in runtimes]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Replica runtime configs differ")

    combined_groups: list[dict] = []
    groups_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for runtime in runtimes:
        for group in runtime.get("groups", []):
            key = (
                str(group.get("method")),
                str(group.get("dataset_name")),
                str(Path(group.get("method_output_root", "")).resolve()),
            )
            groups_by_key.setdefault(key, []).append(group)
    for key, replica_groups in groups_by_key.items():
        first = replica_groups[0]
        starts = [group.get("started_at") for group in replica_groups if group.get("started_at")]
        finishes = [
            group.get("finished_at")
            for group in replica_groups
            if group.get("finished_at")
        ]
        started_at = min(starts) if starts else None
        finished_at = max(finishes) if finishes else None
        wall_clock_sec = None
        if started_at and finished_at:
            wall_clock_sec = round(
                (
                    datetime.fromisoformat(finished_at)
                    - datetime.fromisoformat(started_at)
                ).total_seconds(),
                3,
            )
        tasks = [task for group in replica_groups for task in group.get("tasks", [])]
        tasks.sort(key=lambda task: int(task.get("shard_index", -1)))
        combined_groups.append(
            {
                **{name: value for name, value in first.items() if name not in {"tasks", "status", "started_at", "finished_at", "wall_clock_sec"}},
                "started_at": started_at,
                "finished_at": finished_at,
                "wall_clock_sec": wall_clock_sec,
                "status": (
                    "succeeded"
                    if all(group.get("status") == "succeeded" for group in replica_groups)
                    else "failed"
                ),
                "tasks": tasks,
                "replica_group_count": len(replica_groups),
            }
        )

    starts = [runtime.get("started_at") for runtime in runtimes if runtime.get("started_at")]
    finishes = [runtime.get("finished_at") for runtime in runtimes if runtime.get("finished_at")]
    started_at = min(starts) if starts else None
    finished_at = max(finishes) if finishes else None
    wall_clock_sec = None
    if started_at and finished_at:
        wall_clock_sec = round(
            (
                datetime.fromisoformat(finished_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
            3,
        )
    return {
        "contract_version": "habitbench.multireplica_suite.v1",
        "status": (
            "succeeded"
            if all(runtime.get("status") == "succeeded" for runtime in runtimes)
            else "failed"
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_sec": wall_clock_sec,
        "host": [runtime.get("host") for runtime in runtimes],
        "plan": runtimes[0].get("plan"),
        "plan_sha256": runtimes[0].get("plan_sha256"),
        "plan_manifest": runtimes[0].get("plan_manifest"),
        "launcher": runtimes[0].get("launcher"),
        "replica_count": expected_count,
        "gpu_count": sum(int(runtime.get("gpu_count") or 0) for runtime in runtimes),
        "gpus": {str(runtime["replica_index"]): runtime.get("gpus") for runtime in runtimes},
        "task_count": sum(int(runtime.get("task_count") or 0) for runtime in runtimes),
        "global_task_count": runtimes[0].get("global_task_count"),
        "group_count": len(combined_groups),
        "config": configs[0],
        "preflight": [runtime.get("preflight") for runtime in runtimes],
        "replica_runtimes": [str(path) for path in paths],
        "groups": combined_groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON; defaults to PLAN directory/evaluation_summary.json.",
    )
    parser.add_argument(
        "--plan-manifest",
        type=Path,
        help="Plan metadata JSON; defaults to PLAN with .manifest.json suffix.",
    )
    parser.add_argument(
        "--replica-runtime-root",
        type=Path,
        help=(
            "Launch-scoped directory containing suite_runtime.replica-*.json; "
            "defaults to the plan directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    with args.plan.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    groups: dict[tuple[str, str, str, str, str, int], None] = {}
    for row in rows:
        key = (
            row["method"],
            row["dataset_name"],
            row["dataset_dir"],
            row.get("domain_filter", ""),
            row["method_output_root"],
            int(row["shard_count"]),
        )
        groups[key] = None

    suite_runtime_path = args.plan.resolve().parent / "suite_runtime.json"
    replica_runtime_root = (
        args.replica_runtime_root.expanduser().resolve()
        if args.replica_runtime_root
        else args.plan.resolve().parent
    )
    replica_paths = sorted(
        replica_runtime_root.glob("suite_runtime.replica-*-of-*.json")
    )
    if replica_paths:
        # RJob resumes reuse output roots. The current launch overwrites every
        # per-Replica runtime, while suite_runtime.json may belong to an older
        # launch, so fresh Replica evidence must take precedence.
        suite_runtime = _combine_replica_runtimes(replica_paths)
        write_json(suite_runtime_path, suite_runtime)
    elif suite_runtime_path.is_file():
        suite_runtime = json.loads(suite_runtime_path.read_text(encoding="utf-8"))
    else:
        suite_runtime = None
    plan_manifest_path = (
        args.plan_manifest.resolve()
        if args.plan_manifest
        else args.plan.resolve().with_suffix(".manifest.json")
    )
    plan_manifest = (
        json.loads(plan_manifest_path.read_text(encoding="utf-8"))
        if plan_manifest_path.is_file()
        else None
    )
    actual_plan_sha256 = sha256_file(args.plan)
    if (
        plan_manifest
        and plan_manifest.get("plan_sha256") != actual_plan_sha256
    ):
        raise ValueError(
            f"Plan hash no longer matches {plan_manifest_path}: "
            f"{actual_plan_sha256}"
        )
    observed_groups = {
        (
            row.get("method"),
            row.get("dataset_name"),
            str(Path(row.get("method_output_root", "")).resolve()),
        ): row
        for row in (suite_runtime or {}).get("groups", [])
    }

    results = []
    for (
        method,
        dataset_name,
        dataset_dir,
        domain_filter,
        method_output_root,
        shard_count,
    ) in groups:
        output_root = Path(method_output_root)
        if method in ORACLE_METHODS:
            manifest = merge_oracle_shards(
                dataset_dir=Path(dataset_dir),
                shard_root=output_root,
                output_dir=output_root / "merged",
                mode=method,
                expected_shards=shard_count,
                domain_filter=domain_filter or None,
            )
        else:
            manifest = merge_shards(
                Path(dataset_dir),
                output_root,
                output_root / "merged",
                method,
                shard_count,
                domain_filter or None,
            )
        observed = observed_groups.get(
            (method, dataset_name, str(output_root.resolve()))
        )
        results.append(
            {
                "method": method,
                "dataset_alias": dataset_name,
                "domain_filter": domain_filter or None,
                "dataset": manifest["dataset"]["name"],
                "dataset_manifest": manifest["dataset"],
                "output": str(output_root / "merged"),
                "shard_count": shard_count,
                "method_config": manifest.get("method_config"),
                "experiment_role": manifest.get("experiment_role"),
                "timing": {
                    **manifest["timing"],
                    "cluster_group_wall_clock_sec": (
                        observed.get("wall_clock_sec") if observed else None
                    ),
                    "cluster_group_started_at": (
                        observed.get("started_at") if observed else None
                    ),
                    "cluster_group_finished_at": (
                        observed.get("finished_at") if observed else None
                    ),
                },
                "result": manifest["result"],
            }
        )
    summary = {
        "contract_version": "habitbench.evaluation_summary.v1",
        "plan": str(args.plan.resolve()),
        "plan_sha256": actual_plan_sha256,
        "plan_manifest": (
            str(plan_manifest_path) if plan_manifest_path.is_file() else None
        ),
        "launcher": (plan_manifest or {}).get("launcher"),
        "suite_runtime": (
            str(suite_runtime_path) if suite_runtime_path.is_file() else None
        ),
        "execution": (
            {
                "status": suite_runtime.get("status"),
                "host": suite_runtime.get("host"),
                "gpu_count": suite_runtime.get("gpu_count"),
                "gpus": suite_runtime.get("gpus"),
                "config": suite_runtime.get("config"),
                "started_at": suite_runtime.get("started_at"),
                "finished_at": suite_runtime.get("finished_at"),
                "wall_clock_sec": suite_runtime.get("wall_clock_sec"),
            }
            if suite_runtime
            else None
        ),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "merge_wall_clock_sec": round(time.perf_counter() - started, 3),
        "groups": results,
    }
    summary_path = (
        args.summary.resolve()
        if args.summary
        else args.plan.resolve().parent / "evaluation_summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
