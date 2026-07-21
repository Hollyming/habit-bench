#!/usr/bin/env python
"""Merge every method/dataset group represented in a shard plan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.merge_shards import merge_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.plan.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    groups: dict[tuple[str, str, str, int], None] = {}
    for row in rows:
        key = (
            row["method"],
            row["dataset_dir"],
            row["method_output_root"],
            int(row["shard_count"]),
        )
        groups[key] = None

    results = []
    for method, dataset_dir, method_output_root, shard_count in groups:
        output_root = Path(method_output_root)
        manifest = merge_shards(
            Path(dataset_dir),
            output_root,
            output_root / "merged",
            method,
            shard_count,
        )
        results.append(
            {
                "method": method,
                "dataset": manifest["dataset"]["name"],
                "output": str(output_root / "merged"),
                "result": manifest["result"],
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
