#!/usr/bin/env python3
"""Submit the Qwen3-8B LoCoMo comparison to H via RJob."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path("/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/H集群architecture分区RJob任务提交规范.md")
ENTRY = "http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451"
DEFAULT_METHODS = "long_context,bm25_rag,embedding_rag,mem0,amem,memos,memrl,lightmem,letta,mirix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creator-ad", default=os.environ.get("HABITBENCH_CREATOR_AD", ""))
    parser.add_argument("--creator-type", default=os.environ.get("HABITBENCH_CREATOR_TYPE", "group"), choices=["group"])
    parser.add_argument("--job-type", default="reserved", choices=["reserved"])
    parser.add_argument("--job-name", default="zjm-locomo-q8b-v1")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/habit-h200-locomo-qwen3-8b-v1")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "scripts/cluster/env.h.example.sh")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--image", default="registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20251117")
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--task-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)


def authenticated_creator() -> str:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source /etc/profile.d/ssh-init.sh; printf "%s\\n" "${BRAIN_USERNAME:-}"; hostname -f',
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    username = result[0].strip() if result else ""
    hostname = result[1].strip() if len(result) > 1 else ""
    if not username or username == "brainpp":
        parts = hostname.split(".")
        username = parts[1] if len(parts) > 1 else ""
    return username


def main() -> int:
    args = parse_args()
    if not args.creator_ad:
        raise SystemExit("--creator-ad is required and must be the actual Group-AD")
    actual_creator = authenticated_creator()
    if actual_creator != args.creator_ad:
        raise SystemExit(
            f"--creator-ad={args.creator_ad} does not match authenticated creator {actual_creator}"
        )
    if len(args.job_name) > 32 or "_" in args.job_name or not args.job_name.islower():
        raise SystemExit("--job-name must be <=32 lowercase letters/digits/hyphens")
    if args.port_base <= 0 or args.port_base + 8 > 65535:
        raise SystemExit("--port-base leaves no room for eight local servers")
    if args.task_attempts < 1:
        raise SystemExit("--task-attempts must be positive")

    env_file = args.env_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan = (args.plan or output_root / "locomo_plan.tsv").expanduser().resolve()
    dataset_file = (
        args.dataset_file.expanduser().resolve()
        if args.dataset_file
        else Path(
            os.environ.get(
                "HABITBENCH_LOCOMO_DATA_FILE",
                "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/datasets/locomo/locomo10.json",
            )
        ).expanduser().resolve()
    )
    for path in (SPEC_PATH, env_file, dataset_file, PROJECT_ROOT / "scripts/cluster/run_h_locomo.sh"):
        if not path.is_file():
            raise SystemExit(f"required path is missing: {path}")
    if not str(output_root).startswith("/mnt/shared-storage-"):
        raise SystemExit(f"--output-root must be persistent H storage: {output_root}")

    # Resolve the H paths exactly as the worker will see them.
    loaded = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(env_file)],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
    ).stdout
    base_env = dict(os.environ)
    for item in loaded.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            base_env[key.decode()] = value.decode()
    base_env["HABITBENCH_PROJECT_ROOT"] = str(PROJECT_ROOT)
    base_env["HABITBENCH_LOCOMO_DATA_FILE"] = str(dataset_file)
    base_env["HABITBENCH_LOCOMO_TASK_ATTEMPTS"] = str(args.task_attempts)
    python_bin = Path(base_env.get("PYTHON_BIN", ""))
    vllm_python = Path(base_env.get("HABITBENCH_VLLM_PYTHON", ""))
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise SystemExit(f"H method Python is not executable: {python_bin}")
    if not vllm_python.is_file() or not os.access(vllm_python, os.X_OK):
        raise SystemExit(f"H vLLM Python is not executable: {vllm_python}")
    if not dataset_file.is_file() or dataset_file.stat().st_size == 0:
        raise SystemExit(f"LoCoMo dataset is missing/empty: {dataset_file}")

    # Match the worker wrapper's runtime library order on the submit host so
    # the vLLM sqlite/diskcache import failure is caught before requesting
    # GPUs.  The H base image's system libstdc++ is too old for this frozen
    # vLLM environment's ICU dependency.
    method_env_lib = python_bin.parent.parent / "lib"
    vllm_env_lib = vllm_python.parent.parent / "lib"
    for env_lib in (method_env_lib, vllm_env_lib):
        if not env_lib.is_dir():
            raise SystemExit(f"Persistent Conda runtime directory is missing: {env_lib}")
    existing_ld = base_env.get("LD_LIBRARY_PATH", "")
    library_parts = [str(vllm_env_lib), str(method_env_lib)]
    if existing_ld:
        library_parts.append(existing_ld)
    base_env["LD_LIBRARY_PATH"] = ":".join(library_parts)
    sqlite_check = subprocess.run(
        [str(vllm_python), "-c", "import sqlite3"],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=base_env,
        timeout=60,
    )
    if sqlite_check.returncode != 0:
        detail = (sqlite_check.stderr or sqlite_check.stdout or "unknown error").strip()
        raise SystemExit(f"H vLLM sqlite runtime preflight failed: {detail[-4000:]}")

    plan_cmd = [
        str(python_bin),
        str(PROJECT_ROOT / "scripts/create_locomo_plan.py"),
        "--dataset-file", str(dataset_file),
        "--output-root", str(output_root),
        "--plan", str(plan),
        "--methods", args.methods,
        "--model-name", base_env.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
        "--model-path", base_env["HABITBENCH_LLM_MODEL"],
        "--embedding-path", base_env["HABITBENCH_EMBED_MODEL"],
    ]
    run(plan_cmd, env=base_env)

    # Run the same dependency/model/MIRIX profile checks that the H worker will
    # use before requesting any GPUs. This catches drift in the persistent
    # method environment without starting a server or changing RJob state.
    preflight_code = (
        "import json, os; "
        "from scripts.run_multigpu_plan import _preflight_runtime; "
        "methods=set(os.environ['HABITBENCH_LOCOMO_METHODS'].split(',')); "
        "result=_preflight_runtime(os.environ.copy(), methods, inference_backend='local-vllm'); "
        "print(json.dumps({'status': result['status'], 'methods': sorted(methods), 'method_imports': result['method_imports']}, sort_keys=True))"
    )
    preflight_env = dict(base_env)
    preflight_env["PYTHONPATH"] = str(PROJECT_ROOT) + (":" + preflight_env["PYTHONPATH"] if preflight_env.get("PYTHONPATH") else "")
    preflight_env["HABITBENCH_LOCOMO_METHODS"] = args.methods
    try:
        preflight = subprocess.run(
            [str(python_bin), "-c", preflight_code],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=preflight_env,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "LoCoMo runtime preflight failed:\n" + (exc.stderr or exc.stdout or "unknown error")[-12000:]
        ) from exc
    print("LoCoMo runtime preflight passed: " + preflight.stdout.strip().splitlines()[-1], flush=True)

    # All required paths are below this user's persistent GPFS2 root. One
    # explicit mount therefore covers source, model, caches, data and outputs.
    mount = "gpfs://gpfs2/plm-gpfs/jmzhang:/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang"
    worker = [
        "bash", str(PROJECT_ROOT / "scripts/cluster/run_h_locomo.sh"),
        "--plan", str(plan),
        "--env-file", str(env_file),
        "--port-base", str(args.port_base),
    ]
    rjob = [
        "rjob", "submit",
        "--name", args.job_name,
        "--namespace", "ailab-llmarchitecture",
        "--task-type", "normal",
        "--priority", "5",
        "--charged-group", "llmarchitecture_gpu",
        "--private-machine", "group",
        "--gpu", "8",
        "--cpu", "64",
        "--memory", "524288",
        "-P", "2",
        "--image", args.image,
        "--image-pull-policy", "IfNotPresent",
        # rjob treats --host-network as a flag-only option and rewrites the
        # bare flag to ``--host-network=true`` internally.  Passing a separate
        # ``true`` value leaves an unconsumed positional argument and makes
        # submission fail before the RJob is created.
        "--host-network=true",
        "--share-host-shm", "True",
        "-e", "DISTRIBUTED_JOB=true",
        "--mount", mount,
        "--",
        *worker,
    ]
    print("creator=" + args.creator_ad + "/" + args.creator_type + " job_type=" + args.job_type + " resources=8 GPUs x 2 Replicas = 16 H200s", flush=True)
    print("plan=" + str(plan) + " output_root=" + str(output_root), flush=True)
    if args.dry_run:
        print("dry-run: " + " ".join(rjob))
        return 0

    # Immediately before the only state-changing operation, reread the latest
    # specification and reinitialize SSH and the llmarchitecture scheduler.
    SPEC_PATH.read_text(encoding="utf-8")
    state_env = dict(base_env)
    state_env["KUBEBRAIN_CLUSTER_ENTRY"] = ENTRY
    run(
        [
            "bash",
            "-c",
            'source /etc/profile.d/ssh-init.sh; export KUBEBRAIN_CLUSTER_ENTRY="$1"; shift; exec "$@"',
            "bash",
            ENTRY,
            *rjob,
        ],
        env=state_env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
