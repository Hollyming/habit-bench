#!/usr/bin/env python
"""Create a portable task plan for user-sharded HABIT-Bench evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.dataset import load_dataset


DEFAULT_DATASETS = {
    "food": PROJECT_ROOT / "domain/food/food_habit_lifelines_stress",
    "finance_software": PROJECT_ROOT
    / "domain/finance-software/habit_bench_multidogo_finance_software_long_hard_diverse_v0_5",
}
PLAN_FIELDS = (
    "task_id",
    "method",
    "dataset_name",
    "dataset_dir",
    "method_output_root",
    "shard_index",
    "shard_count",
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_overrides(values: list[str]) -> dict[str, Path]:
    datasets = dict(DEFAULT_DATASETS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--dataset must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        datasets[name.strip()] = Path(raw_path).expanduser().resolve()
    return datasets


def create_plan(args: argparse.Namespace) -> list[dict[str, str | int]]:
    registry = json.loads((PROJECT_ROOT / "eval/methods.json").read_text(encoding="utf-8"))
    methods = _split_csv(args.methods)
    unknown_methods = set(methods) - set(registry)
    if not methods or unknown_methods:
        raise ValueError(f"Unknown or empty methods: {sorted(unknown_methods)}")

    datasets = _dataset_overrides(args.dataset)
    selected_datasets = _split_csv(args.datasets)
    unknown_datasets = set(selected_datasets) - set(datasets)
    if not selected_datasets or unknown_datasets:
        raise ValueError(f"Unknown or empty datasets: {sorted(unknown_datasets)}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    output_root = args.output_root.expanduser().resolve()
    rows: list[dict[str, str | int]] = []
    for dataset_name in selected_datasets:
        dataset_dir = datasets[dataset_name]
        user_count = int(load_dataset(dataset_dir).manifest["users"])
        if args.shards > user_count:
            raise ValueError(
                f"Dataset {dataset_name} has {user_count} users; cannot create {args.shards} nonempty shards"
            )
        for method in methods:
            method_output_root = output_root / dataset_name / method
            for shard_index in range(args.shards):
                rows.append(
                    {
                        "task_id": len(rows),
                        "method": method,
                        "dataset_name": dataset_name,
                        "dataset_dir": str(dataset_dir),
                        "method_output_root": str(method_output_root),
                        "shard_index": shard_index,
                        "shard_count": args.shards,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated methods. Controls run only when explicitly listed.",
    )
    parser.add_argument(
        "--datasets",
        default="food,finance_software",
        help="Comma-separated dataset aliases.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add or override a dataset alias; may be repeated.",
    )
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan.exists() and not args.force:
        raise FileExistsError(f"Plan already exists; pass --force to replace it: {args.plan}")
    rows = create_plan(args)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PLAN_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"plan": str(args.plan), "tasks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
