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
    suite_runtime = (
        json.loads(suite_runtime_path.read_text(encoding="utf-8"))
        if suite_runtime_path.is_file()
        else None
    )
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
