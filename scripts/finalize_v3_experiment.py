#!/usr/bin/env python
"""Record node completion and finalize v3 supplementary analyses exactly once."""

from __future__ import annotations

import argparse
import fcntl
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.io import write_json
from scripts.create_v3_experiment_plans import ALL_METHODS, DEFAULT_SUITE_ROOT


EXPECTED_NODES = ("node01", "node02", "node03")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> None:
    suite_root = args.suite_root.expanduser().resolve()
    if args.node not in EXPECTED_NODES:
        raise ValueError(f"Unknown v3 node name: {args.node}")
    plan_root = suite_root / "plans" / args.node
    summary_path = plan_root / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Node merge summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    execution = summary.get("execution") or {}
    if execution.get("status") != "succeeded":
        raise RuntimeError(
            f"{args.node} cannot be marked complete: "
            f"suite status={execution.get('status')!r}"
        )

    status_root = suite_root / "status"
    status_root.mkdir(parents=True, exist_ok=True)
    write_json(
        status_root / f"{args.node}.done.json",
        {
            "contract_version": "habitbench.v3_node_done.v1",
            "node": args.node,
            "host": socket.gethostname(),
            "completed_at": _now(),
            "evaluation_summary": str(summary_path),
            "groups": len(summary.get("groups") or []),
        },
    )
    missing = [
        node
        for node in EXPECTED_NODES
        if not (status_root / f"{node}.done.json").is_file()
    ]
    if missing:
        print(
            json.dumps(
                {
                    "node": args.node,
                    "status": "node_complete_waiting_for_peers",
                    "missing_nodes": missing,
                },
                indent=2,
            )
        )
        return

    lock_path = status_root / "finalize.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        supplementary_manifest = (
            suite_root / "supplementary/supplementary_manifest.json"
        )
        if supplementary_manifest.is_file():
            print(
                json.dumps(
                    {
                        "status": "already_finalized",
                        "manifest": str(supplementary_manifest),
                    },
                    indent=2,
                )
            )
            return

        experiment_path = suite_root / "experiment_manifest.json"
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        experiment["status"] = "primary_complete_supplementary_running"
        experiment["primary_completed_at"] = _now()
        write_json(experiment_path, experiment)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_supplementary_analysis.py"),
            "--suite-root",
            str(suite_root),
            "--output-root",
            str(suite_root / "supplementary"),
            "--domains",
            "food,finance,software",
            "--methods",
            ",".join(ALL_METHODS),
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ]
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except BaseException as exc:
            experiment["status"] = "primary_complete_supplementary_failed"
            experiment["supplementary_error_type"] = type(exc).__name__
            experiment["supplementary_error"] = str(exc)
            write_json(experiment_path, experiment)
            raise
        experiment["status"] = "complete"
        experiment["finished_at"] = _now()
        experiment["supplementary_manifest"] = str(
            supplementary_manifest.resolve()
        )
        experiment["human_audit_status"] = (
            "prepared_awaiting_two_independent_human_annotations"
        )
        write_json(experiment_path, experiment)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "suite_root": str(suite_root),
                    "supplementary_manifest": str(supplementary_manifest),
                },
                indent=2,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--node", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
