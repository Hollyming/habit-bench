#!/usr/bin/env python
"""Run a shard plan on one multi-GPU host with one sequential worker per GPU."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="CUDA device ids, for example 0,1,2,3")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port-base", type=int, default=8100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = _split_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    with args.plan.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    task_ids = [int(row["task_id"]) for row in rows]
    assignments = [task_ids[index:: len(gpus)] for index in range(len(gpus))]

    def run_worker(worker_index: int) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[worker_index]
        env["HABITBENCH_VLLM_PORT"] = str(args.port_base + worker_index)
        for task_id in assignments[worker_index]:
            command = [
                "bash",
                str(PROJECT_ROOT / "scripts/run_shard_plan_task.sh"),
                str(args.plan.resolve()),
                str(task_id),
            ]
            if args.env_file:
                command.append(str(args.env_file.resolve()))
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_worker, index) for index in range(len(gpus))]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
