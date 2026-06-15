#!/usr/bin/env python
"""Upload the Lumia bundle, optionally run it remotely, fetch results, and audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BUNDLE = Path("./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz")
DEFAULT_PREFLIGHT_AUDIT = Path("./runs/lumia_preflight_import_audit.json")
REMOTE_MANIFESTS_SUBDIR = "runs/lumia_manifests"
REMOTE_RESULTS_SUBDIR = (
    "runs/"
    "habit_bench_balanced_v0_3_official_subset_90/full_official_results"
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bundle_env(args: argparse.Namespace) -> List[str]:
    manifest_path = args.bundle.with_suffix(args.bundle.suffix + ".manifest.json")
    env = []
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        tarball = manifest.get("tarball", {})
        if tarball.get("sha256"):
            env.append(f"HABITBENCH_BUNDLE_SHA256={tarball['sha256']}")
        if tarball.get("bytes") is not None:
            env.append(f"HABITBENCH_BUNDLE_BYTES={tarball['bytes']}")
        if manifest.get("file_count") is not None:
            env.append(f"HABITBENCH_BUNDLE_FILE_COUNT={manifest['file_count']}")
    env.append(f"HABITBENCH_BUNDLE_MANIFEST={args.remote_dir}/{args.bundle.name}.manifest.json")
    env.append("HABITBENCH_MODEL_PREFLIGHT_MANIFEST=runs/lumia_manifests/model_preflight_manifest_remote.json")
    env.append("HABITBENCH_MODEL_DOWNLOAD_MANIFEST=runs/lumia_manifests/model_download_manifest.json")
    return env


def shell_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def reset_local_subdir(local_root: Path, remote_subdir: str, execute: bool) -> Path:
    local_path = local_root / Path(remote_subdir)
    if execute and local_path.exists():
        shutil.rmtree(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path.parent


def command_prefix(value: str, name: str) -> List[str]:
    """Split command prefixes such as "scp -O" while tolerating Windows paths."""
    if not value.strip():
        raise ValueError(f"--{name} cannot be empty")
    parts = shlex.split(value, posix=(os.name != "nt"))
    cleaned = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            part = part[1:-1]
        cleaned.append(part)
    if not cleaned:
        raise ValueError(f"--{name} cannot be empty")
    return cleaned


def remote_bash_command(script: str) -> str:
    return "bash -lc " + shlex.quote(script)


def remote_runner_command(args: argparse.Namespace, script: str) -> str:
    if args.remote_run_prefix:
        return args.remote_run_prefix + " " + shlex.quote(script)
    return remote_bash_command(script)


def detached_runner_command(args: argparse.Namespace, script: str) -> str:
    if args.slurm_detached:
        return remote_bash_command(slurm_detached_script(args, script))
    return remote_runner_command(args, detached_script(args, script))


def fetch_remote_subdir(
    rows: List[Dict[str, Any]],
    scp_cmd: List[str],
    args: argparse.Namespace,
    local_root: Path,
    remote_subdir: str,
) -> None:
    local_parent = reset_local_subdir(local_root, remote_subdir, args.execute)
    remote_path = f"{args.host}:{args.remote_dir}/habit-bench-lumia-full-official/{remote_subdir}/"
    rows.append(run([*scp_cmd, "-r", remote_path, shell_path(local_parent)], args.execute))


def run(command: List[str], execute: bool) -> Dict[str, Any]:
    printable = " ".join(shlex.quote(part) for part in command)
    if not execute:
        return {"command": command, "printable": printable, "returncode": None, "skipped": True}
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "printable": printable,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-4000:],
    }


def remote_path_exists(
    rows: List[Dict[str, Any]],
    ssh_cmd: List[str],
    args: argparse.Namespace,
    remote_subdir: str,
) -> bool:
    remote_path = f"{args.remote_dir}/habit-bench-lumia-full-official/{remote_subdir}"
    check = f"test -e {shlex.quote(remote_path)}"
    row = run([*ssh_cmd, args.host, remote_bash_command(check)], args.execute)
    row["optional_probe"] = True
    rows.append(row)
    return (not args.execute) or row.get("returncode") == 0


def require_ok(row: Dict[str, Any]) -> None:
    if row.get("returncode") not in (0, None):
        raise RuntimeError(f"Command failed: {row['printable']}\n{row.get('stderr', '')}")


def require_preflight_pass(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preflight audit {path}. Run --preflight-only --execute and require status=pass before full run."
        )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("status") != "pass":
        raise RuntimeError(f"Preflight audit is not pass: {path}: status={data.get('status')}")
    return {"path": str(path), "status": data.get("status"), "checks": data.get("checks", {})}


def remote_status_script(args: argparse.Namespace) -> str:
    audit_path = (
        "habit-bench-lumia-full-official/runs/"
        "habit_bench_balanced_v0_3_official_subset_90/full_official_results/"
        "audit/full_official_audit.json"
    )
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
echo remote_dir={shlex.quote(args.remote_dir)}
job_id_arg={shlex.quote(args.remote_job_id)}
jobid=""
if [ -f "$job_id_arg" ]; then
  jobid=$(cat "$job_id_arg")
elif printf '%s' "$job_id_arg" | grep -Eq '^[0-9]+$'; then
  jobid="$job_id_arg"
fi
if [ -n "$jobid" ]; then
  if squeue -j "$jobid" -h 2>/dev/null | grep -q .; then
    echo status=running job_id="$jobid"
  else
    echo status=not_running job_id="$jobid"
  fi
elif [ -f {shlex.quote(args.remote_pid)} ]; then
  pid=$(cat {shlex.quote(args.remote_pid)})
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo status=running pid="$pid"
  else
    echo status=not_running pid="$pid"
  fi
else
  echo status=no_pid_file
fi
if [ -f {shlex.quote(args.remote_exit_code)} ]; then
  echo exit_code=$(cat {shlex.quote(args.remote_exit_code)})
fi
if [ -f {shlex.quote(args.remote_log)} ]; then
  echo log_path={shlex.quote(args.remote_log)}
  echo '--- log tail ---'
  tail -n 80 {shlex.quote(args.remote_log)}
fi
if [ -f {shlex.quote(audit_path)} ]; then
  echo '--- audit json ---'
  cat {shlex.quote(audit_path)}
fi
"""


def remote_status_json_script(args: argparse.Namespace) -> str:
    audit_path = (
        "habit-bench-lumia-full-official/runs/"
        "habit_bench_balanced_v0_3_official_subset_90/full_official_results/"
        "audit/full_official_audit.json"
    )
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
{remote_python_bootstrap()}
pid=""
job_id=""
running="unknown"
exit_code=""
audit_status=""
job_id_arg={shlex.quote(args.remote_job_id)}
if [ -f "$job_id_arg" ]; then
  job_id=$(cat "$job_id_arg")
elif printf '%s' "$job_id_arg" | grep -Eq '^[0-9]+$'; then
  job_id="$job_id_arg"
fi
if [ -n "$job_id" ]; then
  pid="slurm:$job_id"
  if squeue -j "$job_id" -h 2>/dev/null | grep -q .; then
    running="true"
  else
    running="false"
  fi
elif [ -f {shlex.quote(args.remote_pid)} ]; then
  pid=$(cat {shlex.quote(args.remote_pid)})
  if kill -0 "$pid" >/dev/null 2>&1; then
    running="true"
  else
    running="false"
  fi
else
  running="no_pid_file"
fi
if [ -f {shlex.quote(args.remote_exit_code)} ]; then
  exit_code=$(cat {shlex.quote(args.remote_exit_code)})
fi
if [ -f {shlex.quote(audit_path)} ]; then
  audit_status=$("${{PYTHON_BIN:-python}}" - <<'PY'
import json
from pathlib import Path
path = Path({audit_path!r})
try:
    print(json.loads(path.read_text(encoding="utf-8-sig")).get("status", ""))
except Exception:
    print("unreadable")
PY
)
fi
"${{PYTHON_BIN:-python}}" - <<PY
import json
payload = {{
  "remote_dir": {args.remote_dir!r},
  "pid": "$pid",
  "job_id": "$job_id",
  "running": "$running",
  "exit_code": "$exit_code",
  "audit_status": "$audit_status",
  "remote_log": {args.remote_log!r},
  "remote_exit_code": {args.remote_exit_code!r},
}}
print(json.dumps(payload, sort_keys=True))
PY
"""


def parse_remote_status(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("returncode") not in (0, None):
        return {"status": "probe_failed", "row": row}
    stdout = row.get("stdout", "").strip()
    if not stdout:
        return {"status": "dry_run" if row.get("skipped") else "empty_stdout", "row": row}
    last_line = stdout.splitlines()[-1]
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        return {"status": "unparseable_stdout", "stdout": stdout[-1000:], "row": row}
    return data


def wait_status_allows_fetch(wait_status: Dict[str, Any] | None, execute: bool) -> bool:
    if not execute:
        return True
    return (wait_status or {}).get("exit_code") == "0"


def wait_status_completion_reason(wait_status: Dict[str, Any] | None) -> str:
    status = wait_status or {}
    exit_code = status.get("exit_code")
    if exit_code == "0":
        return "remote_exit_code_zero"
    if exit_code:
        return f"remote_exit_code_nonzero:{exit_code}"
    running = status.get("running")
    if running == "true":
        return "remote_still_running"
    if running == "no_pid_file":
        return "remote_pid_file_missing"
    if running == "false":
        return "remote_not_running_without_exit_code"
    return f"remote_status_incomplete:{running or 'unknown'}"


def env_export_block(args: argparse.Namespace) -> str:
    env_exports = []
    for item in [*bundle_env(args), *args.remote_env]:
        if "=" not in item:
            raise ValueError(f"--remote-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if key == "PYTHON_BIN":
            continue
        env_exports.append(f"export {shlex.quote(key)}={shlex.quote(value)}")
    return "\n".join(env_exports)


def remote_user_env_export_block(args: argparse.Namespace) -> str:
    env_exports = []
    for item in args.remote_env:
        if "=" not in item:
            raise ValueError(f"--remote-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env_exports.append(f"export {shlex.quote(key)}={shlex.quote(value)}")
    return "\n".join(env_exports)


REMOTE_PYTHON_BOOTSTRAP = """
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "No python3 or python executable found on remote PATH" >&2
    exit 127
  fi
fi
export PYTHON_BIN
"""


def remote_python_bootstrap() -> str:
    return REMOTE_PYTHON_BOOTSTRAP


def remote_script(args: argparse.Namespace, bundle_name: str) -> str:
    env_block = env_export_block(args)
    user_env_block = remote_user_env_export_block(args)
    skip_download = "1" if args.skip_model_download else "0"
    reuse_server = "1" if args.reuse_server else "0"
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
{user_env_block}
{remote_python_bootstrap()}
sha256sum -c {shlex.quote(bundle_name)}.sha256
rm -rf habit-bench-lumia-full-official
tar -xzf {shlex.quote(bundle_name)}
cd habit-bench-lumia-full-official
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r ./requirements-official.txt
"$PYTHON_BIN" ./scripts/lumia/check_lumia_readiness.py
source ./scripts/lumia/lumia_env_example.sh
export HABITBENCH_SKIP_MODEL_DOWNLOAD={skip_download}
export HABITBENCH_REUSE_SERVER={reuse_server}
{env_block}
bash ./scripts/lumia/run_lumia_full_official_e2e.sh
"""


def remote_reuse_workspace_script(args: argparse.Namespace) -> str:
    env_block = env_export_block(args)
    reuse_server = "1" if args.reuse_server else "0"
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}/habit-bench-lumia-full-official
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" ./scripts/lumia/check_lumia_readiness.py
source ./scripts/lumia/lumia_env_example.sh
export HABITBENCH_SKIP_MODEL_DOWNLOAD=1
export HABITBENCH_REUSE_SERVER={reuse_server}
{env_block}
bash ./scripts/lumia/run_lumia_full_official_e2e.sh
"""


def remote_preflight_script(args: argparse.Namespace, bundle_name: str) -> str:
    env_block = env_export_block(args)
    user_env_block = remote_user_env_export_block(args)
    model_preflight = ""
    if not args.skip_model_download:
        model_preflight = """
"$PYTHON_BIN" ./scripts/lumia/preflight_open_models.py \
  --out ./runs/lumia_manifests/model_preflight_manifest_remote.json
"""
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
{user_env_block}
{remote_python_bootstrap()}
sha256sum -c {shlex.quote(bundle_name)}.sha256
rm -rf habit-bench-lumia-full-official
tar -xzf {shlex.quote(bundle_name)}
cd habit-bench-lumia-full-official
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r ./requirements-official.txt
"$PYTHON_BIN" ./scripts/lumia/check_lumia_readiness.py \
  --out ./runs/lumia_manifests/lumia_readiness_remote.json
source ./scripts/lumia/lumia_env_example.sh
{env_block}
"$PYTHON_BIN" ./scripts/lumia/preflight_lumia_run.py \
  --out ./runs/lumia_manifests/lumia_run_preflight_manifest_remote.json
{model_preflight}
echo "Remote Lumia preflight completed. No full official suite was started."
"""


def remote_download_models_script(args: argparse.Namespace, bundle_name: str) -> str:
    env_block = env_export_block(args)
    user_env_block = remote_user_env_export_block(args)
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
{user_env_block}
{remote_python_bootstrap()}
sha256sum -c {shlex.quote(bundle_name)}.sha256
rm -rf habit-bench-lumia-full-official
tar -xzf {shlex.quote(bundle_name)}
cd habit-bench-lumia-full-official
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r ./requirements-official.txt
source ./scripts/lumia/lumia_env_example.sh
{env_block}
bash ./scripts/lumia/download_open_models.sh
echo "Remote Lumia model download completed. No vLLM server or method suite was started."
"""


def remote_preflight_download_script(args: argparse.Namespace, bundle_name: str) -> str:
    env_block = env_export_block(args)
    user_env_block = remote_user_env_export_block(args)
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
{user_env_block}
{remote_python_bootstrap()}
sha256sum -c {shlex.quote(bundle_name)}.sha256
rm -rf habit-bench-lumia-full-official
tar -xzf {shlex.quote(bundle_name)}
cd habit-bench-lumia-full-official
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r ./requirements-official.txt
"$PYTHON_BIN" ./scripts/lumia/check_lumia_readiness.py \
  --out ./runs/lumia_manifests/lumia_readiness_remote.json
source ./scripts/lumia/lumia_env_example.sh
{env_block}
"$PYTHON_BIN" ./scripts/lumia/preflight_lumia_run.py \
  --out ./runs/lumia_manifests/lumia_run_preflight_manifest_remote.json
"$PYTHON_BIN" ./scripts/lumia/preflight_open_models.py \
  --out ./runs/lumia_manifests/model_preflight_manifest_remote.json
bash ./scripts/lumia/download_open_models.sh
echo "Remote Lumia preflight and model download completed. No vLLM server or method suite was started."
"""


def detached_script(args: argparse.Namespace, script: str) -> str:
    wrapped = (
        f"rm -f {shlex.quote(args.remote_exit_code)}; "
        f"({script}\n); code=$?; echo $code > {shlex.quote(args.remote_exit_code)}; exit $code"
    )
    quoted_wrapped = shlex.quote(wrapped)
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
nohup bash -lc {quoted_wrapped} > {shlex.quote(args.remote_log)} 2>&1 < /dev/null &
echo $! > {shlex.quote(args.remote_pid)}
echo started_detached_pid=$(cat {shlex.quote(args.remote_pid)})
echo remote_log={shlex.quote(args.remote_log)}
echo remote_exit_code={shlex.quote(args.remote_exit_code)}
"""


def slurm_detached_script(args: argparse.Namespace, script: str) -> str:
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.slurm_job_name}",
        f"#SBATCH --output={args.remote_log}",
        f"#SBATCH --error={args.remote_log}",
        f"#SBATCH --chdir={args.remote_dir}",
    ]
    if args.slurm_partition:
        directives.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_gres:
        directives.append(f"#SBATCH --gres={args.slurm_gres}")
    if args.slurm_time:
        directives.append(f"#SBATCH --time={args.slurm_time}")
    for extra in args.slurm_extra:
        directives.append(f"#SBATCH {extra}")
    batch = "\n".join(
        [
            *directives,
            "set -u",
            f"rm -f {shlex.quote(args.remote_exit_code)}",
            "(",
            "set -euo pipefail",
            script,
            ")",
            "code=$?",
            f"echo \"$code\" > {shlex.quote(args.remote_exit_code)}",
            "exit \"$code\"",
            "",
        ]
    )
    return f"""
set -euo pipefail
cd {shlex.quote(args.remote_dir)}
cat > {shlex.quote(args.remote_batch_script)} <<'SBATCH_EOF'
{batch}
SBATCH_EOF
rm -f {shlex.quote(args.remote_exit_code)}
jobid=$(sbatch --parsable {shlex.quote(args.remote_batch_script)} | cut -d';' -f1)
echo "$jobid" > {shlex.quote(args.remote_job_id)}
echo slurm_job_id="$jobid"
echo remote_log={shlex.quote(args.remote_log)}
echo remote_exit_code={shlex.quote(args.remote_exit_code)}
echo remote_batch_script={shlex.quote(args.remote_batch_script)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH target, e.g. user@lumia")
    parser.add_argument("--remote-dir", required=True, help="Remote directory where the bundle will be placed.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--handoff",
        type=Path,
        default=None,
        help="Optional handoff sidecar to upload. Defaults to the bundle stem plus .handoff.md when present.",
    )
    parser.add_argument("--returned-root", type=Path, default=Path("runs/lumia_returned/habit-bench-lumia-full-official"))
    parser.add_argument("--ssh", default="ssh")
    parser.add_argument("--scp", default="scp")
    parser.add_argument("--execute", action="store_true", help="Actually execute upload/ssh/fetch commands. Default is dry-run.")
    parser.add_argument("--skip-remote-run", action="store_true", help="Only upload/fetch command plan; do not run remote e2e.")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--detached", action="store_true", help="Start the remote e2e run with nohup and return without fetching results.")
    parser.add_argument("--preflight-only", action="store_true", help="Upload the bundle and run remote readiness/preflight checks only; do not start the full e2e suite.")
    parser.add_argument("--download-models-only", action="store_true", help="Upload the bundle, download/cache open models remotely, fetch manifests, and audit the model download.")
    parser.add_argument(
        "--preflight-then-detached",
        action="store_true",
        help="Run remote preflight, fetch/audit it locally, then start the detached full run only if the preflight audit passes.",
    )
    parser.add_argument(
        "--preflight-download-then-detached",
        action="store_true",
        help=(
            "Run remote preflight and open-model download, fetch/audit both locally, "
            "then start the detached full run only if both audits pass."
        ),
    )
    parser.add_argument("--fetch-after-detached", action="store_true", help="Fetch immediately even when --detached is used.")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch/import existing remote results; do not upload or run.")
    parser.add_argument("--status-only", action="store_true", help="Only check remote detached-run status and tail the remote log.")
    parser.add_argument(
        "--wait-and-fetch",
        action="store_true",
        help="Poll detached-run status; fetch/import only after the remote exit-code file contains 0.",
    )
    parser.add_argument("--wait-timeout-sec", type=int, default=0, help="Maximum wait for --wait-and-fetch. 0 checks once.")
    parser.add_argument("--wait-poll-sec", type=int, default=60, help="Polling interval for --wait-and-fetch.")
    parser.add_argument("--remote-log", default="habitbench_remote_e2e.log")
    parser.add_argument("--remote-pid", default="habitbench_remote_e2e.pid")
    parser.add_argument("--remote-job-id", default="habitbench_remote_e2e.jobid")
    parser.add_argument("--remote-exit-code", default="habitbench_remote_e2e.exitcode")
    parser.add_argument("--remote-batch-script", default="habitbench_remote_e2e.sbatch")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--reuse-remote-workspace", action="store_true", help="Start a full run from an existing unpacked remote workspace without upload/reinstall.")
    parser.add_argument("--slurm-detached", action="store_true", help="Submit detached full runs with sbatch instead of nohup.")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-gres", default="")
    parser.add_argument("--slurm-time", default="")
    parser.add_argument("--slurm-job-name", default="habitbench")
    parser.add_argument("--slurm-extra", action="append", default=[], help="Extra SBATCH directive text, e.g. '--cpus-per-task=8'.")
    parser.add_argument(
        "--require-preflight-pass",
        action="store_true",
        help="Refuse to start a full remote run unless the local preflight import audit JSON has status=pass.",
    )
    parser.add_argument("--preflight-audit", type=Path, default=DEFAULT_PREFLIGHT_AUDIT)
    parser.add_argument("--remote-env", action="append", default=[], help="Environment override passed to remote run, KEY=VALUE.")
    parser.add_argument(
        "--remote-run-prefix",
        default="",
        help=(
            "Optional shell prefix for remote execution scripts, e.g. "
            "'srun --partition=RTX4090 --gres=gpu:1 --time=02:00:00 bash -lc'. "
            "Upload, fetch, and status probes still run on the login host."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("./runs/lumia_remote_launch_plan.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ssh_cmd = command_prefix(args.ssh, "ssh")
    scp_cmd = command_prefix(args.scp, "scp")
    exclusive_modes = [args.status_only, args.fetch_only, args.wait_and_fetch]
    if sum(1 for value in exclusive_modes if value) > 1:
        raise ValueError("--status-only, --fetch-only, and --wait-and-fetch are mutually exclusive")
    if args.wait_and_fetch and args.skip_fetch:
        raise ValueError("--wait-and-fetch cannot be combined with --skip-fetch")
    if args.wait_timeout_sec < 0:
        raise ValueError("--wait-timeout-sec must be >= 0")
    if args.wait_poll_sec <= 0:
        raise ValueError("--wait-poll-sec must be > 0")
    if args.detached and args.skip_remote_run:
        raise ValueError("--detached cannot be combined with --skip-remote-run")
    if args.reuse_remote_workspace and (
        args.preflight_only
        or args.download_models_only
        or args.preflight_then_detached
        or args.preflight_download_then_detached
        or args.fetch_only
        or args.status_only
        or args.wait_and_fetch
        or args.skip_remote_run
    ):
        raise ValueError("--reuse-remote-workspace can only be used for a direct full run")
    if args.slurm_detached and not (
        args.detached
        or args.preflight_then_detached
        or args.preflight_download_then_detached
        or args.reuse_remote_workspace
        or args.fetch_only
        or args.status_only
        or args.wait_and_fetch
    ):
        raise ValueError("--slurm-detached requires --detached, --reuse-remote-workspace, or a guarded detached mode")
    if args.preflight_only and (args.detached or args.fetch_only or args.status_only or args.wait_and_fetch or args.skip_remote_run):
        raise ValueError("--preflight-only cannot be combined with --detached, --fetch-only, --status-only, --wait-and-fetch, or --skip-remote-run")
    if args.download_models_only and (args.detached or args.fetch_only or args.status_only or args.wait_and_fetch or args.skip_remote_run):
        raise ValueError("--download-models-only cannot be combined with --detached, --fetch-only, --status-only, --wait-and-fetch, or --skip-remote-run")
    if args.preflight_then_detached and (
        args.preflight_only
        or args.download_models_only
        or args.preflight_download_then_detached
        or args.detached
        or args.fetch_only
        or args.status_only
        or args.wait_and_fetch
        or args.skip_remote_run
        or args.skip_fetch
    ):
        raise ValueError(
            "--preflight-then-detached cannot be combined with --preflight-only, --detached, "
            "--download-models-only, --fetch-only, --status-only, --wait-and-fetch, --skip-remote-run, or --skip-fetch"
        )
    if args.preflight_download_then_detached and (
        args.preflight_only
        or args.download_models_only
        or args.preflight_then_detached
        or args.detached
        or args.fetch_only
        or args.status_only
        or args.wait_and_fetch
        or args.skip_remote_run
        or args.skip_fetch
        or args.skip_model_download
    ):
        raise ValueError(
            "--preflight-download-then-detached cannot be combined with --preflight-only, "
            "--download-models-only, --preflight-then-detached, --detached, --fetch-only, "
            "--status-only, --wait-and-fetch, --skip-remote-run, --skip-fetch, or --skip-model-download"
        )
    preflight_gate = None
    if args.require_preflight_pass and not (
        args.preflight_only
        or args.download_models_only
        or args.preflight_then_detached
        or args.preflight_download_then_detached
        or args.fetch_only
        or args.status_only
        or args.wait_and_fetch
        or args.skip_remote_run
    ):
        try:
            preflight_gate = require_preflight_pass(args.preflight_audit)
        except Exception as exc:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "status": "fail",
                        "mode": "execute" if args.execute else "dry_run",
                        "host": args.host,
                        "remote_dir": args.remote_dir,
                        "detached": args.detached,
                        "preflight_only": args.preflight_only,
                        "require_preflight_pass": args.require_preflight_pass,
                        "preflight_audit": str(args.preflight_audit),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            raise

    bundle = args.bundle
    bundle_name = bundle.name
    handoff = args.handoff or bundle.with_name(bundle.name.removesuffix(".tar.gz") + ".handoff.md")
    files = [
        bundle,
        bundle.with_suffix(bundle.suffix + ".sha256"),
        bundle.with_suffix(bundle.suffix + ".manifest.json"),
    ]
    if handoff.exists():
        files.append(handoff)
    missing = [str(path) for path in files if not path.exists()]
    if missing and not (args.fetch_only or args.status_only or args.wait_and_fetch):
        raise FileNotFoundError(f"Missing bundle files: {missing}")

    rows = []
    wait_status = None
    if args.wait_and_fetch:
        deadline = time.monotonic() + args.wait_timeout_sec
        while True:
            row = run([*ssh_cmd, args.host, remote_bash_command(remote_status_json_script(args))], args.execute)
            rows.append(row)
            wait_status = parse_remote_status(row)
            if not args.execute:
                break
            if wait_status.get("exit_code"):
                break
            if wait_status.get("running") in {"no_pid_file", "false"} and not wait_status.get("exit_code"):
                break
            if args.wait_timeout_sec == 0 or time.monotonic() >= deadline:
                break
            time.sleep(args.wait_poll_sec)
    elif args.status_only:
        rows.append(run([*ssh_cmd, args.host, remote_bash_command(remote_status_script(args))], args.execute))
    elif not args.fetch_only:
        rows.append(run([*ssh_cmd, args.host, "mkdir", "-p", args.remote_dir], args.execute))
        if not args.reuse_remote_workspace:
            for path in files:
                rows.append(run([*scp_cmd, shell_path(path), f"{args.host}:{args.remote_dir}/"], args.execute))

        if not args.skip_remote_run:
            if args.download_models_only:
                script = remote_download_models_script(args, bundle_name)
            elif args.preflight_download_then_detached:
                script = remote_preflight_download_script(args, bundle_name)
            elif args.preflight_only or args.preflight_then_detached:
                script = remote_preflight_script(args, bundle_name)
            elif args.reuse_remote_workspace:
                script = remote_reuse_workspace_script(args)
            else:
                script = remote_script(args, bundle_name)
            if args.detached:
                rows.append(run([*ssh_cmd, args.host, detached_runner_command(args, script)], args.execute))
            else:
                rows.append(run([*ssh_cmd, args.host, remote_runner_command(args, script)], args.execute))

    should_fetch = not args.skip_fetch and not args.status_only and (
        args.preflight_only
        or args.download_models_only
        or args.preflight_then_detached
        or args.preflight_download_then_detached
        or args.fetch_only
        or (args.wait_and_fetch and wait_status_allows_fetch(wait_status, args.execute))
        or not args.detached
        or args.fetch_after_detached
    )
    if should_fetch:
        local_root = args.returned_root
        local_root.mkdir(parents=True, exist_ok=True)
        fetch_remote_subdir(rows, scp_cmd, args, local_root, REMOTE_MANIFESTS_SUBDIR)
        fetch_results = not (
            args.preflight_only
            or args.download_models_only
            or args.preflight_then_detached
            or args.preflight_download_then_detached
        )
        if args.fetch_only and not remote_path_exists(rows, ssh_cmd, args, REMOTE_RESULTS_SUBDIR):
            fetch_results = False
        if not (
            args.preflight_only
            or args.download_models_only
            or args.preflight_then_detached
            or args.preflight_download_then_detached
        ) and fetch_results:
            fetch_remote_subdir(rows, scp_cmd, args, local_root, REMOTE_RESULTS_SUBDIR)

        if args.preflight_download_then_detached:
            import_cmds = [
                [
                    sys.executable,
                    "./scripts/lumia/audit_lumia_preflight.py",
                    "--returned-root",
                    str(local_root),
                ],
                [
                    sys.executable,
                    "./scripts/lumia/audit_model_download.py",
                    "--returned-root",
                    str(local_root),
                ],
            ]
        elif args.preflight_only or args.preflight_then_detached:
            import_cmds = [[
                sys.executable,
                "./scripts/lumia/audit_lumia_preflight.py",
                "--returned-root",
                str(local_root),
            ]]
        elif args.download_models_only:
            import_cmds = [[
                sys.executable,
                "./scripts/lumia/audit_model_download.py",
                "--returned-root",
                str(local_root),
            ]]
        else:
            import_cmds = [[
                sys.executable,
                "./scripts/lumia/import_lumia_results.py",
                "--returned-root",
                str(local_root),
            ]]
        for import_cmd in import_cmds:
            rows.append(run(import_cmd, args.execute))

    if args.preflight_then_detached or args.preflight_download_then_detached:
        prior_steps_ok = all(
            row.get("returncode") in (0, None) or row.get("optional_probe")
            for row in rows
        )
        if prior_steps_ok:
            full_script = (
                remote_reuse_workspace_script(args)
                if args.preflight_download_then_detached
                else remote_script(args, bundle_name)
            )
            rows.append(
                run(
                    [
                        *ssh_cmd,
                        args.host,
                        detached_runner_command(args, full_script),
                    ],
                    args.execute,
                )
            )
        else:
            rows.append(
                {
                    "command": [],
                    "printable": "detached full run skipped because a required local audit failed",
                    "returncode": None,
                    "skipped": True,
                    "skip_reason": "required_local_audit_failed",
                }
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "host": args.host,
        "remote_dir": args.remote_dir,
        "detached": args.detached,
        "preflight_only": args.preflight_only,
        "download_models_only": args.download_models_only,
        "preflight_then_detached": args.preflight_then_detached,
        "preflight_download_then_detached": args.preflight_download_then_detached,
        "require_preflight_pass": args.require_preflight_pass,
        "preflight_gate": preflight_gate,
        "fetch_only": args.fetch_only,
        "status_only": args.status_only,
        "wait_and_fetch": args.wait_and_fetch,
        "wait_timeout_sec": args.wait_timeout_sec,
        "wait_poll_sec": args.wait_poll_sec,
        "remote_log": args.remote_log,
        "remote_pid": args.remote_pid,
        "remote_job_id": args.remote_job_id,
        "remote_exit_code": args.remote_exit_code,
        "remote_batch_script": args.remote_batch_script,
        "skip_model_download": args.skip_model_download,
        "reuse_server": args.reuse_server,
        "reuse_remote_workspace": args.reuse_remote_workspace,
        "slurm_detached": args.slurm_detached,
        "slurm_partition": args.slurm_partition,
        "slurm_gres": args.slurm_gres,
        "slurm_time": args.slurm_time,
        "slurm_job_name": args.slurm_job_name,
        "slurm_extra": args.slurm_extra,
        "remote_run_prefix": args.remote_run_prefix,
        "wait_status": wait_status,
        "bundle": str(bundle),
        "handoff": str(handoff) if handoff.exists() else None,
        "returned_root": str(args.returned_root),
        "commands": rows,
        "note": (
            "Dry-run mode only prints the commands that would be executed. "
            "Preflight-only mode fetches returned readiness/preflight manifests "
            "and audits them unless --skip-fetch is supplied. "
            "Download-models-only mode fetches model manifests and audits the "
            "download without starting vLLM or methods. "
            "Preflight-then-detached mode starts the detached full run only "
            "after the local preflight audit command succeeds. "
            "Preflight-download-then-detached mode first requires both the "
            "local preflight audit and local model-download audit to pass, "
            "then reuses the same remote workspace with model download skipped "
            "so the audited model manifest is preserved. "
            "Use --require-preflight-pass before full runs to enforce the local "
            "preflight audit gate. "
            "Detached mode starts the remote job and skips fetch unless "
            "--fetch-after-detached is supplied. "
            "Wait-and-fetch mode polls the detached exit-code file and only "
            "fetches/imports after exit_code=0. Fetching copies only the "
            "returned manifest/results subdirectories, not the remote virtualenv. "
            "--remote-run-prefix can route remote execution through schedulers "
            "such as Slurm srun while keeping upload/fetch/status on the login host."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    failed = None
    for row in rows:
        if row.get("optional_probe") and row.get("returncode") not in (0, None):
            continue
        if row.get("returncode") not in (0, None):
            failed = row
            break
    if failed:
        summary["status"] = "fail"
        summary["failed_command"] = failed
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        require_ok(failed)

    if args.wait_and_fetch and args.execute and (wait_status or {}).get("exit_code") != "0":
        summary["status"] = "incomplete"
        summary["reason"] = wait_status_completion_reason(wait_status)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit("Remote detached run is not complete with exit_code=0; fetch/import skipped")

    summary["status"] = "pass"
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"mode": summary["mode"], "out": str(args.out), "commands": len(rows)}, indent=2))
    if not args.execute:
        print("\nCommands:")
        for row in rows:
            print(row["printable"])


if __name__ == "__main__":
    main()
