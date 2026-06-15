#!/usr/bin/env python
"""Run the guarded Lumia full official cycle with one local command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BUNDLE = Path("./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz")
LAUNCHER = Path("./scripts/lumia/launch_lumia_remote.py")


def run(command: List[str]) -> Dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-6000:],
        "stderr": completed.stderr[-6000:],
    }


def base_launcher_args(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(LAUNCHER),
        "--host",
        args.host,
        "--remote-dir",
        args.remote_dir,
        "--bundle",
        str(args.bundle),
        "--returned-root",
        str(args.returned_root),
        "--ssh",
        args.ssh,
        "--scp",
        args.scp,
        "--remote-log",
        args.remote_log,
        "--remote-pid",
        args.remote_pid,
        "--remote-job-id",
        args.remote_job_id,
        "--remote-exit-code",
        args.remote_exit_code,
        "--remote-batch-script",
        args.remote_batch_script,
    ]
    if args.remote_run_prefix:
        command.extend(["--remote-run-prefix", args.remote_run_prefix])
    if args.slurm_detached:
        command.append("--slurm-detached")
    if args.slurm_partition:
        command.extend(["--slurm-partition", args.slurm_partition])
    if args.slurm_gres:
        command.extend(["--slurm-gres", args.slurm_gres])
    if args.slurm_time:
        command.extend(["--slurm-time", args.slurm_time])
    if args.slurm_job_name:
        command.extend(["--slurm-job-name", args.slurm_job_name])
    for item in args.slurm_extra:
        command.extend(["--slurm-extra", item])
    if args.handoff:
        command.extend(["--handoff", str(args.handoff)])
    for item in args.remote_env:
        command.extend(["--remote-env", item])
    if args.reuse_server:
        command.append("--reuse-server")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--handoff", type=Path, default=None)
    parser.add_argument("--returned-root", type=Path, default=Path("runs/lumia_returned/habit-bench-lumia-full-official"))
    parser.add_argument("--ssh", default="ssh")
    parser.add_argument("--scp", default="scp")
    parser.add_argument("--remote-log", default="habitbench_remote_e2e.log")
    parser.add_argument("--remote-pid", default="habitbench_remote_e2e.pid")
    parser.add_argument("--remote-job-id", default="habitbench_remote_e2e.jobid")
    parser.add_argument("--remote-exit-code", default="habitbench_remote_e2e.exitcode")
    parser.add_argument("--remote-batch-script", default="habitbench_remote_e2e.sbatch")
    parser.add_argument("--remote-env", action="append", default=[])
    parser.add_argument("--remote-run-prefix", default="")
    parser.add_argument("--slurm-detached", action="store_true")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-gres", default="")
    parser.add_argument("--slurm-time", default="")
    parser.add_argument("--slurm-job-name", default="habitbench")
    parser.add_argument("--slurm-extra", action="append", default=[])
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--wait-timeout-sec", type=int, default=86400)
    parser.add_argument("--wait-poll-sec", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./runs/lumia_guarded_full_cycle_summary.json"),
    )
    parser.add_argument(
        "--launch-plan-out",
        type=Path,
        default=Path("./runs/lumia_guarded_full_cycle_launch_plan.json"),
    )
    parser.add_argument(
        "--wait-plan-out",
        type=Path,
        default=Path("./runs/lumia_guarded_full_cycle_wait_plan.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = base_launcher_args(args)
    launch_cmd = [
        *base,
        "--preflight-download-then-detached",
        "--out",
        str(args.launch_plan_out),
    ]
    wait_cmd = [
        *base,
        "--wait-and-fetch",
        "--wait-timeout-sec",
        str(args.wait_timeout_sec),
        "--wait-poll-sec",
        str(args.wait_poll_sec),
        "--out",
        str(args.wait_plan_out),
    ]
    if args.execute:
        launch_cmd.append("--execute")
        wait_cmd.append("--execute")

    steps = []
    launch = run(launch_cmd)
    steps.append({"name": "preflight_download_then_detached", **launch})
    if launch["returncode"] == 0:
        wait = run(wait_cmd)
        steps.append({"name": "wait_and_fetch", **wait})

    status = "pass" if steps and all(step["returncode"] == 0 for step in steps) else "fail"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "status": status,
        "host": args.host,
        "remote_dir": args.remote_dir,
        "bundle": str(args.bundle),
        "returned_root": str(args.returned_root),
        "wait_timeout_sec": args.wait_timeout_sec,
        "wait_poll_sec": args.wait_poll_sec,
        "remote_run_prefix": args.remote_run_prefix,
        "slurm_detached": args.slurm_detached,
        "slurm_partition": args.slurm_partition,
        "slurm_gres": args.slurm_gres,
        "slurm_time": args.slurm_time,
        "slurm_job_name": args.slurm_job_name,
        "slurm_extra": args.slurm_extra,
        "steps": steps,
        "notes": [
            "Step 1 starts the guarded detached full official run only after remote preflight and model-download audits pass.",
            "Step 2 imports results only after the detached exit-code file contains 0.",
            "Use --remote-run-prefix to route remote execution through Slurm srun while upload/status/fetch stay on the login host.",
            "Use --slurm-detached with --slurm-partition/--slurm-gres/--slurm-time to submit the long run via sbatch.",
            "Dry-run mode only prints child launcher command plans; it does not contact Lumia.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "mode": summary["mode"], "out": str(args.out), "steps": len(steps)}, indent=2))
    if status != "pass":
        raise SystemExit("Guarded Lumia full cycle failed")


if __name__ == "__main__":
    main()
