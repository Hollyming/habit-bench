#!/usr/bin/env python
"""Create the three balanced, eight-GPU plans for HABIT-Bench v3."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.dataset import load_dataset
from eval.core.io import sha256_file, write_json
from scripts.create_shard_plan import (
    BGE_M3_METHODS,
    PLAN_FIELDS,
    _bge_m3_snapshot,
    _git_state,
    _method_configs,
)


DEFAULT_SUITE_ROOT = PROJECT_ROOT / "results/habit_3domain_v3"
SHARDS = 8
NODE_COUNT = 3
REGULAR_METHODS = (
    "no_memory",
    "full_memory",
    "letta",
    "memrl",
    "mem0",
    "memos",
    "amem",
    "lightmem",
    "secom",
    "mirix",
    "omem",
    "graphiti",
)
ORACLE_METHODS = ("oracle_evidence", "oracle_habit_state")
ALL_METHODS = REGULAR_METHODS + ORACLE_METHODS
DATASETS = {
    "food": {
        "path": PROJECT_ROOT / "domain/food/food_habit_lifelines_stress_v4",
        "domain_filter": None,
    },
    "finance": {
        "path": PROJECT_ROOT
        / "domain/finance-software"
        / "habit_bench_multidogo_finance_software_scope_consistent_v1.3",
        "domain_filter": "finance",
    },
    "software": {
        "path": PROJECT_ROOT
        / "domain/finance-software"
        / "habit_bench_multidogo_finance_software_scope_consistent_v1.3",
        "domain_filter": "software",
    },
}

# Scheduling estimates only. They are deliberately stored in the experiment
# manifest and never used as benchmark results. Values combine v2 measured
# group wall times with conservative estimates for newly repaired methods.
ESTIMATED_HOURS = {
    "no_memory": {"food": 0.04, "finance": 0.03, "software": 0.02},
    "full_memory": {"food": 0.08, "finance": 0.07, "software": 0.05},
    "oracle_evidence": {"food": 0.05, "finance": 0.05, "software": 0.03},
    "oracle_habit_state": {"food": 0.05, "finance": 0.05, "software": 0.03},
    "letta": {"food": 0.22, "finance": 0.30, "software": 0.25},
    "memrl": {"food": 0.70, "finance": 1.30, "software": 1.35},
    "amem": {"food": 0.80, "finance": 3.00, "software": 3.00},
    "mem0": {"food": 1.35, "finance": 2.00, "software": 2.30},
    "memos": {"food": 1.20, "finance": 2.25, "software": 2.30},
    "lightmem": {"food": 1.50, "finance": 2.50, "software": 2.50},
    "secom": {"food": 1.50, "finance": 4.00, "software": 3.00},
    "mirix": {"food": 1.00, "finance": 3.00, "software": 3.00},
    "omem": {"food": 2.00, "finance": 5.00, "software": 4.00},
    "graphiti": {"food": 3.50, "finance": 12.00, "software": 12.00},
}


def _balanced_groups() -> list[list[dict[str, Any]]]:
    groups = [
        {
            "method": method,
            "dataset": dataset,
            "estimated_hours": ESTIMATED_HOURS[method][dataset],
        }
        for method in ALL_METHODS
        for dataset in DATASETS
    ]
    nodes: list[dict[str, Any]] = [
        {"estimated_hours": 0.0, "groups": []} for _ in range(NODE_COUNT)
    ]
    for group in sorted(
        groups,
        key=lambda item: (
            -float(item["estimated_hours"]),
            str(item["method"]),
            str(item["dataset"]),
        ),
    ):
        node = min(
            enumerate(nodes),
            key=lambda item: (float(item[1]["estimated_hours"]), item[0]),
        )[1]
        node["groups"].append(group)
        node["estimated_hours"] += float(group["estimated_hours"])
    return [
        sorted(
            node["groups"],
            key=lambda item: (
                float(item["estimated_hours"]),
                str(item["method"]),
                str(item["dataset"]),
            ),
        )
        for node in nodes
    ]


def _write_plan(
    suite_root: Path,
    node_index: int,
    groups: list[dict[str, Any]],
    dataset_manifests: dict[str, dict[str, Any]],
    method_configs: dict[str, dict | None],
) -> dict[str, Any]:
    node_name = f"node{node_index:02d}"
    node_root = suite_root / "plans" / node_name
    plan_path = node_root / "shard_plan.tsv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for group in groups:
        dataset_name = str(group["dataset"])
        method = str(group["method"])
        dataset = DATASETS[dataset_name]
        method_output_root = suite_root / dataset_name / method
        for shard_index in range(SHARDS):
            rows.append(
                {
                    "task_id": len(rows),
                    "method": method,
                    "dataset_name": dataset_name,
                    "dataset_dir": str(dataset["path"].resolve()),
                    "domain_filter": dataset["domain_filter"] or "",
                    "max_users": "",
                    "max_probes": "",
                    "method_output_root": str(method_output_root.resolve()),
                    "shard_index": shard_index,
                    "shard_count": SHARDS,
                }
            )
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PLAN_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    selected_methods = list(
        dict.fromkeys(str(group["method"]) for group in groups)
    )
    selected_datasets = list(
        dict.fromkeys(str(group["dataset"]) for group in groups)
    )
    job_name = f"hb3d-v3-{node_name}"
    manifest = {
        "contract_version": "habitbench.shard_plan.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "output_root": str(suite_root.resolve()),
        "task_count": len(rows),
        "shard_count": SHARDS,
        "methods": {
            method: (
                {
                    "diagnostic_only": True,
                    "experiment_role": "diagnostic_upper_bound",
                    "source": "eval.supplementary.oracle_controls",
                }
                if method in ORACLE_METHODS
                else method_configs[method]
            )
            for method in selected_methods
        },
        "models": {
            "embedding": (
                _bge_m3_snapshot()
                if any(method in BGE_M3_METHODS for method in selected_methods)
                else None
            )
        },
        "datasets": {
            dataset: dataset_manifests[dataset] for dataset in selected_datasets
        },
        "project": {"root": str(PROJECT_ROOT), **_git_state()},
        "launcher": {
            "clusterx_job_name": job_name,
            "cluster": "cluster-t",
            "queue": "queue-t-reserved-plm",
            "machine_type": "n3ls.ii.i60a",
            "gpus": 8,
            "cpus": 64,
            "memory_gib": 512,
            "shm_gib": 64,
            "node_plan": node_name,
            "estimated_hours": round(
                sum(float(group["estimated_hours"]) for group in groups), 3
            ),
        },
        "group_order": groups,
    }
    manifest_path = plan_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    return {
        "name": node_name,
        "clusterx_job": job_name,
        "plan": str(plan_path.resolve()),
        "plan_manifest": str(manifest_path.resolve()),
        "estimated_hours": manifest["launcher"]["estimated_hours"],
        "groups": groups,
        "task_count": len(rows),
    }


def run(args: argparse.Namespace) -> None:
    suite_root = args.suite_root.expanduser().resolve()
    if args.mark_submitted:
        experiment_path = suite_root / "experiment_manifest.json"
        if not experiment_path.is_file():
            raise FileNotFoundError(
                f"Cannot mark an experiment without its manifest: {experiment_path}"
            )
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        experiment["status"] = "submitted"
        experiment["submitted_at"] = datetime.now(timezone.utc).isoformat()
        write_json(experiment_path, experiment)
        print({"suite_root": str(suite_root), "status": "submitted"})
        return
    plans_root = suite_root / "plans"
    if plans_root.exists() and not args.force:
        raise FileExistsError(
            f"v3 plans already exist; pass --force only for an intentional rebuild: "
            f"{plans_root}"
        )
    dataset_manifests: dict[str, dict[str, Any]] = {}
    for name, dataset in DATASETS.items():
        bundle = load_dataset(
            dataset["path"], domain_filter=dataset["domain_filter"]
        )
        if int(bundle.manifest["users"]) < SHARDS:
            raise ValueError(f"{name} has fewer users than the {SHARDS} shards")
        dataset_manifests[name] = {
            "dataset_dir": str(dataset["path"].resolve()),
            "domain_filter": dataset["domain_filter"],
            "max_users": None,
            "max_probes": None,
            "manifest": bundle.manifest,
        }

    method_configs = _method_configs(list(REGULAR_METHODS))
    assignments = _balanced_groups()
    nodes = [
        _write_plan(
            suite_root,
            node_index,
            groups,
            dataset_manifests,
            method_configs,
        )
        for node_index, groups in enumerate(assignments, start=1)
    ]

    assignment_path = suite_root / "group_assignment.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "node",
                "order",
                "method",
                "dataset",
                "estimated_hours",
            ),
        )
        writer.writeheader()
        for node in nodes:
            for order, group in enumerate(node["groups"], start=1):
                writer.writerow(
                    {
                        "node": node["name"],
                        "order": order,
                        **group,
                    }
                )

    experiment_manifest = {
        "contract_version": "habitbench.cluster_experiment.v3",
        "experiment_name": "habit_3domain_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "suite_root": str(suite_root),
        "datasets": dataset_manifests,
        "methods": {
            "deployable_memory": [
                method
                for method in REGULAR_METHODS
                if method not in {"no_memory", "full_memory"}
            ],
            "controls": ["no_memory", "full_memory"],
            "diagnostic_upper_bounds": list(ORACLE_METHODS),
        },
        "shards_per_method_dataset": SHARDS,
        "total_gpus": 24,
        "total_groups": len(ALL_METHODS) * len(DATASETS),
        "total_shard_tasks": len(ALL_METHODS) * len(DATASETS) * SHARDS,
        "nodes": nodes,
        "group_assignment": str(assignment_path.resolve()),
        "scheduling": {
            "policy": (
                "LPT balancing across method/domain groups; ascending execution "
                "within each node so short groups finish before long-tail groups"
            ),
            "weights_are_benchmark_results": False,
            "estimated_hours": ESTIMATED_HOURS,
        },
        "supplementary": {
            "oracle_modes": list(ORACLE_METHODS),
            "offline_analysis": "scripts/run_supplementary_analysis.py",
            "bootstrap_samples": 10_000,
            "seed": 42,
            "human_audit": (
                "templates prepared only; scoring requires two completed human "
                "annotation files and is not fabricated automatically"
            ),
        },
    }
    write_json(suite_root / "experiment_manifest.json", experiment_manifest)
    print(
        {
            "suite_root": str(suite_root),
            "nodes": [
                {
                    "name": node["name"],
                    "groups": len(node["groups"]),
                    "estimated_hours": node["estimated_hours"],
                }
                for node in nodes
            ],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root", type=Path, default=DEFAULT_SUITE_ROOT
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mark-submitted", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
