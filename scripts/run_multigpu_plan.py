#!/usr/bin/env python
"""Run a HABIT shard plan on one multi-GPU node with persistent vLLM workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.context_windows import resolve_context_window


O200K_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"
EMBEDDING_METHODS = {
    "mem0",
    "amem",
    "memos",
    "memrl",
    "lightmem",
    "letta",
    "mirix",
    "graphiti",
    "secom",
    "omem",
}
MEDMEMORY_METHODS = {
    "mem0",
    "amem",
    "memos",
    "memrl",
    "lightmem",
    "letta",
    "mirix",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_env_file(path: Path, base_env: dict[str, str]) -> dict[str, str]:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -a; source "$1"; env -0',
            "bash",
            str(path.resolve()),
        ],
        check=True,
        stdout=subprocess.PIPE,
        env=base_env,
    )
    loaded = dict(base_env)
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        loaded[key.decode()] = value.decode()
    return loaded


def _safe_config(env: dict[str, str]) -> dict[str, Any]:
    names = (
        "HABITBENCH_LLM_MODEL",
        "HABITBENCH_SERVED_MODEL",
        "HABITBENCH_CHAT_TEMPLATE",
        "HABITBENCH_EMBED_MODEL",
        "HABITBENCH_EMBED_DIMS",
        "HABITBENCH_EMBED_DEVICE",
        "HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES",
        "HABITBENCH_ADAPTER_CPU_THREADS",
        "HABITBENCH_MED_USER_WORKERS",
        "HABITBENCH_MEM0_USER_WORKERS",
        "HABITBENCH_AMEM_USER_WORKERS",
        "HABITBENCH_MEMOS_USER_WORKERS",
        "HABITBENCH_MEMRL_USER_WORKERS",
        "HABITBENCH_LIGHTMEM_USER_WORKERS",
        "HABITBENCH_LETTA_USER_WORKERS",
        "HABITBENCH_MIRIX_USER_WORKERS",
        "HABITBENCH_LIGHTMEM_MODEL",
        "HABITBENCH_SECOM_COMPRESSOR",
        "HABITBENCH_SECOM_REPO",
        "HABITBENCH_OMEM_REPO",
        "HABITBENCH_CONTEXT_WINDOW_TIER",
        "HABITBENCH_MAX_INPUT_TOKENS",
        "HABITBENCH_FULL_MEMORY_RESERVED_TOKENS",
        "HABITBENCH_FULL_MEMORY_MAX_TOKENS",
        "HABITBENCH_GPU_MEMORY_UTIL",
        "HABITBENCH_MAX_MODEL_LEN",
        "HABITBENCH_ENABLE_PREFIX_CACHING",
        "HABITBENCH_VLLM_PYTHON",
        "HABITBENCH_VLLM_EXTRA_ARGS",
        "HABITBENCH_VLLM_MIN_TOKENS_PER_SEC",
        "HABITBENCH_VLLM_BENCHMARK_TOKENS",
        "HABITBENCH_VLLM_BENCHMARK_TIMEOUT_SEC",
        "HABITBENCH_VLLM_BENCHMARK_CONCURRENCY",
        "HABITBENCH_MEMORY_LLM_MAX_TOKENS",
        "HABITBENCH_PROGRESS_EVERY",
        "HABITBENCH_OFFICIAL_TIMEOUT_SEC",
        "HABITBENCH_STRUCTURED_OUTPUT_MODE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TIKTOKEN_CACHE_DIR",
        "TRITON_PTXAS_PATH",
        "VLLM_BATCH_INVARIANT",
    )
    return {name: env.get(name) for name in names}


def _inspect_vllm_runtime(python_bin: Path, env: dict[str, str]) -> dict[str, str]:
    package_names = (
        "vllm",
        "torch",
        "triton",
        "transformers",
        "flashinfer-python",
        "xgrammar",
    )
    code = (
        "import importlib.metadata as m, json, platform, torch;"
        f"names={package_names!r};"
        "result={name:m.version(name) for name in names};"
        "result['python']=platform.python_version();"
        "result['torch_cuda']=torch.version.cuda;"
        "print(json.dumps(result, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    versions: dict[str, str] = json.loads(completed.stdout)
    expected = {
        "python": "3.10.20",
        "vllm": "0.17.1",
        "torch": "2.10.0",
        "torch_cuda": "12.8",
        "triton": "3.6.0",
        "transformers": "4.57.6",
        "flashinfer-python": "0.6.4",
        "xgrammar": "0.1.29",
    }
    mismatches = {
        name: {"expected": value, "actual": versions.get(name)}
        for name, value in expected.items()
        if versions.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Dedicated vLLM runtime mismatch: {mismatches}")
    return versions


def _preflight_runtime(
    env: dict[str, str], methods: set[str] | None = None
) -> dict[str, Any]:
    python_bin = Path(env.get("PYTHON_BIN") or sys.executable).expanduser().resolve()
    vllm_python = Path(
        env.get("HABITBENCH_VLLM_PYTHON") or python_bin
    ).expanduser().resolve()
    llm_path = Path(
        env.get(
            "HABITBENCH_LLM_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/Qwen3-8B",
        )
    ).expanduser().resolve()
    embed_path = Path(
        env.get(
            "HABITBENCH_EMBED_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/bge-m3",
        )
    ).expanduser().resolve()
    tiktoken_root = Path(
        env.get(
            "TIKTOKEN_CACHE_DIR",
            "/plm-shared/zhangjunming/.cache/tiktoken",
        )
    ).expanduser().resolve()
    required: dict[str, Path] = {
        "method_python": python_bin,
        "vllm_python": vllm_python,
        "llm_config": llm_path / "config.json",
        "tiktoken_o200k": tiktoken_root / O200K_CACHE_KEY,
    }
    packages: dict[str, str] = {}
    full_memory_window = None
    needs_embedding = methods is None or bool(EMBEDDING_METHODS & methods)
    if needs_embedding:
        required.update(
            {
                "embedding_config": embed_path / "config.json",
                "embedding_weights": embed_path / "pytorch_model.bin",
                "embedding_identity": embed_path / "HABIT_MODEL_INFO.json",
            }
        )
    chat_template = env.get("HABITBENCH_CHAT_TEMPLATE")
    if chat_template:
        required["chat_template"] = Path(chat_template).expanduser().resolve()
    triton_ptxas = env.get("TRITON_PTXAS_PATH")
    if triton_ptxas:
        required["triton_ptxas"] = Path(triton_ptxas).expanduser().resolve()
    if methods and "lightmem" in methods:
        lightmem_path = Path(
            env.get(
                "HABITBENCH_LIGHTMEM_MODEL",
                "/plm-shared/zhangjunming/Workspace/models/"
                "llmlingua-2-xlm-roberta-large-meetingbank",
            )
        ).expanduser().resolve()
        required["lightmem_config"] = lightmem_path / "config.json"
        required["lightmem_weights"] = lightmem_path / "model.safetensors"
    if methods and "secom" in methods:
        compressor_path = Path(
            env.get(
                "HABITBENCH_SECOM_COMPRESSOR",
                "/plm-shared/zhangjunming/Workspace/models/"
                "llmlingua-2-xlm-roberta-large-meetingbank",
            )
        ).expanduser().resolve()
        required["secom_compressor_config"] = compressor_path / "config.json"
        required["secom_compressor_weights"] = (
            compressor_path / "model.safetensors"
        )
    if methods and "secom" in methods:
        secom_path = Path(
            env.get(
                "HABITBENCH_SECOM_REPO",
                str(
                    PROJECT_ROOT
                    / "third_party/official-baselines/vendor/SeCom"
                ),
            )
        ).expanduser().resolve()
        required["secom_source"] = secom_path / "secom/secom.py"
        required["secom_license"] = secom_path / "LICENSE"
    if methods and "omem" in methods:
        omem_path = Path(
            env.get(
                "HABITBENCH_OMEM_REPO",
                str(
                    PROJECT_ROOT
                    / "third_party/official-baselines/vendor/O-Mem"
                ),
            )
        ).expanduser().resolve()
        required["omem_source"] = omem_path / "example_usage.py"
        required["omem_license"] = omem_path / "LICENSE"
    if methods and "graphiti" in methods:
        if importlib.util.find_spec("graphiti_core") is None:
            raise ModuleNotFoundError(
                "graphiti-core is required when graphiti is selected"
            )
        packages["graphiti-core"] = package_version("graphiti-core")
        packages["kuzu"] = package_version("kuzu")
        if packages["graphiti-core"] != "0.29.2":
            raise ValueError(
                "Graphiti runtime revision mismatch: expected graphiti-core "
                f"0.29.2, got {packages['graphiti-core']}"
            )
    if methods and ({"full_memory", "full_history"} & methods):
        full_memory_window = resolve_context_window(
            env.get("HABITBENCH_CONTEXT_WINDOW_TIER", "auto"),
            int(env.get("HABITBENCH_MAX_MODEL_LEN", "40960")),
            custom_max_input_tokens=(
                int(env["HABITBENCH_MAX_INPUT_TOKENS"])
                if env.get("HABITBENCH_MAX_INPUT_TOKENS")
                else None
            ),
            reserved_prompt_tokens=(
                int(env["HABITBENCH_FULL_MEMORY_RESERVED_TOKENS"])
                if env.get("HABITBENCH_FULL_MEMORY_RESERVED_TOKENS")
                else None
            ),
            max_history_tokens=(
                int(env["HABITBENCH_FULL_MEMORY_MAX_TOKENS"])
                if env.get("HABITBENCH_FULL_MEMORY_MAX_TOKENS")
                else None
            ),
        ).public_dict()
    missing = [
        f"{name}={path}"
        for name, path in required.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Cluster runtime preflight found missing/empty files: "
            + ", ".join(missing)
        )
    for name in ("method_python", "vllm_python", "triton_ptxas"):
        path = required.get(name)
        if path is not None and not os.access(path, os.X_OK):
            raise PermissionError(f"Cluster runtime preflight is not executable: {path}")

    identity = None
    if needs_embedding:
        identity = json.loads(
            required["embedding_identity"].read_text(encoding="utf-8")
        )
        expected_identity = {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "dense_embedding_dimension": 1024,
        }
        mismatches = {
            key: {"expected": value, "actual": identity.get(key)}
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        if env.get("HABITBENCH_EMBED_DIMS", "1024") != "1024":
            mismatches["HABITBENCH_EMBED_DIMS"] = {
                "expected": "1024",
                "actual": env.get("HABITBENCH_EMBED_DIMS"),
            }
        if mismatches:
            raise ValueError(f"BGE-M3 runtime identity mismatch: {mismatches}")
    vllm_versions = _inspect_vllm_runtime(vllm_python, env)
    return {
        "status": "pass",
        "files": {
            name: {"path": str(path), "size_bytes": path.stat().st_size}
            for name, path in required.items()
        },
        "embedding_identity": identity,
        "packages": packages,
        "vllm_runtime": vllm_versions,
        "full_memory_window": full_memory_window,
    }


def _load_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Shard plan is empty: {path}")
    return rows


def _group_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    group_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["dataset_name"],
            row["method_output_root"],
            row["shard_count"],
        )
        if key not in group_by_key:
            group_by_key[key] = []
            groups.append(group_by_key[key])
        group_by_key[key].append(row)
    for group in groups:
        expected = int(group[0]["shard_count"])
        indices = sorted(int(row["shard_index"]) for row in group)
        if len(group) != expected or indices != list(range(expected)):
            raise ValueError(
                f"Incomplete plan group {group[0]['method']}/{group[0]['dataset_name']}: "
                f"expected shards 0..{expected - 1}, got {indices}"
            )
    return groups


def _server_command(env: dict[str, str], port: int) -> list[str]:
    python_bin = (
        env.get("HABITBENCH_VLLM_PYTHON")
        or env.get("PYTHON_BIN")
        or sys.executable
    )
    command = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        env.get(
            "HABITBENCH_LLM_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/Qwen3-8B",
        ),
        "--served-model-name",
        env.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        env.get("HABITBENCH_GPU_MEMORY_UTIL", "0.82"),
        "--max-model-len",
        env.get("HABITBENCH_MAX_MODEL_LEN", "40960"),
    ]
    chat_template = env.get("HABITBENCH_CHAT_TEMPLATE")
    if chat_template:
        command.extend(["--chat-template", chat_template])
    if env.get("HABITBENCH_ENABLE_PREFIX_CACHING", "1") == "1":
        command.append("--enable-prefix-caching")
    command.extend(shlex.split(env.get("HABITBENCH_VLLM_EXTRA_ARGS", "")))
    return command


def _wait_for_server(
    process: subprocess.Popen[str],
    base_url: str,
    attempts: int,
    sleep_sec: float,
    log_path: Path,
) -> None:
    for _ in range(attempts):
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"vLLM exited with code {process.returncode}; tail of {log_path}:\n{tail}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2):
                return
        except Exception:
            time.sleep(sleep_sec)
    raise TimeoutError(f"vLLM did not become ready at {base_url}; see {log_path}")


def _request_benchmark_completion(
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_sec: float,
) -> tuple[int, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate a long plain-text hardware throughput sample. "
                    "Write only the word benchmark separated by spaces."
                ),
            }
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    completion_tokens = int(result.get("usage", {}).get("completion_tokens") or 0)
    if completion_tokens <= 0:
        raise RuntimeError(
            f"vLLM benchmark returned no completion-token usage: {result}"
        )
    return completion_tokens, elapsed


def _benchmark_server(worker: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    timeout_sec = float(
        env.get("HABITBENCH_VLLM_BENCHMARK_TIMEOUT_SEC", "120")
    )
    benchmark_tokens = int(
        env.get("HABITBENCH_VLLM_BENCHMARK_TOKENS", "128")
    )
    minimum_rate = float(
        env.get("HABITBENCH_VLLM_MIN_TOKENS_PER_SEC", "60")
    )
    concurrency = int(
        env.get(
            "HABITBENCH_VLLM_BENCHMARK_CONCURRENCY",
            env.get("HABITBENCH_MED_USER_WORKERS", "1"),
        )
    )
    if concurrency <= 0:
        raise ValueError("HABITBENCH_MED_USER_WORKERS must be positive")
    if benchmark_tokens < 64:
        raise ValueError(
            "HABITBENCH_VLLM_BENCHMARK_TOKENS must be at least 64"
        )
    common = {
        "base_url": worker["base_url"],
        "model": env.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
        "timeout_sec": timeout_sec,
    }
    warmup_tokens, warmup_elapsed = _request_benchmark_completion(
        max_tokens=16,
        **common,
    )
    single_tokens, single_elapsed = _request_benchmark_completion(
        max_tokens=64,
        **common,
    )
    measured_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _request_benchmark_completion,
                max_tokens=benchmark_tokens,
                **common,
            )
            for _ in range(concurrency)
        ]
        measurements = [future.result() for future in futures]
    measured_elapsed = time.perf_counter() - measured_started
    measured_tokens = sum(tokens for tokens, _ in measurements)
    rate = measured_tokens / measured_elapsed
    record = {
        "status": "pass" if rate >= minimum_rate else "fail",
        "concurrent_requests": concurrency,
        "minimum_completion_tokens_per_sec": minimum_rate,
        "measured_completion_tokens": measured_tokens,
        "measured_elapsed_sec": round(measured_elapsed, 3),
        "measured_aggregate_completion_tokens_per_sec": round(rate, 3),
        "request_elapsed_sec": [
            round(elapsed, 3) for _, elapsed in measurements
        ],
        "single_completion_tokens": single_tokens,
        "single_elapsed_sec": round(single_elapsed, 3),
        "single_completion_tokens_per_sec": round(
            single_tokens / single_elapsed,
            3,
        ),
        "warmup_completion_tokens": warmup_tokens,
        "warmup_elapsed_sec": round(warmup_elapsed, 3),
    }
    return record


def _start_server(
    worker_index: int,
    gpu: str,
    port: int,
    env: dict[str, str],
    log_root: Path,
) -> dict[str, Any]:
    worker_env = dict(env)
    worker_env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_root / f"vllm_worker_{worker_index:02d}_gpu_{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = _server_command(worker_env, port)
    started_at = _utc_now()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=worker_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        _wait_for_server(
            process,
            base_url,
            int(env.get("HABITBENCH_SERVER_READY_ATTEMPTS", "180")),
            float(env.get("HABITBENCH_SERVER_READY_SLEEP_SEC", "2")),
            log_path,
        )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        log_handle.close()
        raise
    return {
        "worker_index": worker_index,
        "gpu": gpu,
        "port": port,
        "base_url": base_url,
        "process": process,
        "log_handle": log_handle,
        "log": str(log_path),
        "command": command,
        "started_at": started_at,
        "ready_at": _utc_now(),
        "startup_wall_clock_sec": round(time.perf_counter() - started, 3),
    }


def _stop_server(worker: dict[str, Any]) -> None:
    process: subprocess.Popen[str] = worker["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    worker["log_handle"].close()


def _public_worker_record(worker: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in worker.items()
        if key not in {"process", "log_handle"}
    }


def _run_task(
    row: dict[str, str],
    worker_pool: queue.Queue[dict[str, Any]],
    base_env: dict[str, str],
) -> dict[str, Any]:
    worker = worker_pool.get()
    shard_index = int(row["shard_index"])
    shard_count = int(row["shard_count"])
    shard_name = f"shard_{shard_index:03d}_of_{shard_count:03d}"
    output_dir = Path(row["method_output_root"]) / shard_name
    output_dir.mkdir(parents=True, exist_ok=True)
    was_complete = (
        base_env.get("HABITBENCH_FORCE_RERUN", "0") != "1"
        and (output_dir / "metrics.json").is_file()
    )

    task_env = dict(base_env)
    if "HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES" in base_env:
        adapter_cuda = base_env["HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES"]
    elif base_env.get("HABITBENCH_EMBED_DEVICE", "cpu").lower() == "cpu":
        adapter_cuda = ""
    else:
        adapter_cuda = worker["gpu"]
    task_env["CUDA_VISIBLE_DEVICES"] = adapter_cuda
    method = row["method"]
    method_workers: int | None = None
    if method in MEDMEMORY_METHODS:
        worker_variable = f"HABITBENCH_{method.upper()}_USER_WORKERS"
        method_workers = int(
            base_env.get(
                worker_variable,
                base_env.get("HABITBENCH_MED_USER_WORKERS", "1"),
            )
        )
        if method_workers <= 0:
            raise ValueError(f"{worker_variable} must be positive")
        task_env["HABITBENCH_MED_USER_WORKERS"] = str(method_workers)
    cpu_threads = base_env.get("HABITBENCH_ADAPTER_CPU_THREADS", "2")
    if method_workers is not None and method_workers >= 5:
        cpu_threads = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        task_env[name] = cpu_threads
    task_env["OPENAI_BASE_URL"] = worker["base_url"]
    command = [
        "bash",
        str(PROJECT_ROOT / "scripts/run_eval.sh"),
        row["method"],
        row["dataset_dir"],
        str(output_dir),
        "--user-shard-index",
        str(shard_index),
        "--user-shard-count",
        str(shard_count),
        "--base-url",
        worker["base_url"],
        "--progress-every",
        base_env.get("HABITBENCH_PROGRESS_EVERY", "25"),
    ]
    stdout_path = output_dir / "task.stdout.log"
    stderr_path = output_dir / "task.stderr.log"
    started_at = _utc_now()
    started = time.perf_counter()
    returncode: int | None = None
    try:
        try:
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=task_env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            returncode = completed.returncode
            error_type = None
            error = None
        except BaseException as exc:
            error_type = type(exc).__name__
            error = str(exc)
        record = {
            "contract_version": "habitbench.shard_worker.v1",
            "task_id": int(row["task_id"]),
            "method": row["method"],
            "dataset_name": row["dataset_name"],
            "dataset_dir": row["dataset_dir"],
            "output_dir": str(output_dir),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "worker_index": worker["worker_index"],
            "gpu": worker["gpu"],
            "adapter_cuda_visible_devices": adapter_cuda,
            "adapter_cpu_threads": int(cpu_threads),
            "med_user_workers": method_workers,
            "vllm_port": worker["port"],
            "vllm_log": worker["log"],
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_clock_sec": round(time.perf_counter() - started, 3),
            "status": (
                "skipped_completed"
                if was_complete and returncode == 0
                else "succeeded"
                if returncode == 0
                else "failed"
            ),
            "returncode": returncode,
            "error_type": error_type,
            "error": error,
            "command": command,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        _write_json(output_dir / "worker_runtime.json", record)
        if error is not None:
            raise RuntimeError(
                f"Task {row['task_id']} could not run: {error}"
            )
        if returncode != 0:
            raise RuntimeError(
                f"Task {row['task_id']} failed with code {returncode}; see {stderr_path}"
            )
        return record
    finally:
        worker_pool.put(worker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--plan-manifest",
        type=Path,
        help="Plan metadata JSON; defaults to PLAN with .manifest.json suffix.",
    )
    parser.add_argument("--gpus", required=True, help="CUDA device ids, for example 0,1,2,3")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--runtime-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    gpus = _split_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    rows = _load_plan(plan_path)
    groups = _group_rows(rows)
    plan_sha256 = _sha256(plan_path)
    plan_manifest_path = (
        args.plan_manifest.expanduser().resolve()
        if args.plan_manifest
        else plan_path.with_suffix(".manifest.json")
    )
    plan_manifest = (
        json.loads(plan_manifest_path.read_text(encoding="utf-8"))
        if plan_manifest_path.is_file()
        else None
    )
    if plan_manifest:
        if plan_manifest.get("plan_sha256") != plan_sha256:
            raise ValueError(
                f"Plan hash no longer matches {plan_manifest_path}: {plan_sha256}"
            )
        if plan_manifest.get("task_count") != len(rows):
            raise ValueError(
                f"Plan task count no longer matches {plan_manifest_path}: {len(rows)}"
            )
    base_env = os.environ.copy()
    if args.env_file:
        base_env = _source_env_file(args.env_file, base_env)

    runtime_path = (
        args.runtime_output.expanduser().resolve()
        if args.runtime_output
        else plan_path.parent / "suite_runtime.json"
    )
    log_root = plan_path.parent / "vllm_logs"
    suite_started = time.perf_counter()
    manifest: dict[str, Any] = {
        "contract_version": "habitbench.multigpu_suite.v1",
        "status": "preflight",
        "started_at": _utc_now(),
        "finished_at": None,
        "wall_clock_sec": None,
        "host": socket.gethostname(),
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "plan_manifest": (
            str(plan_manifest_path) if plan_manifest_path.is_file() else None
        ),
        "launcher": (plan_manifest or {}).get("launcher"),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "task_count": len(rows),
        "group_count": len(groups),
        "config": _safe_config(base_env),
        "preflight": None,
        "servers": [],
        "groups": [],
    }
    _write_json(runtime_path, manifest)

    workers: list[dict[str, Any]] = []
    try:
        manifest["preflight"] = _preflight_runtime(
            base_env, {row["method"] for row in rows}
        )
        manifest["status"] = "starting_servers"
        _write_json(runtime_path, manifest)
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(
                    _start_server,
                    index,
                    gpu,
                    args.port_base + index,
                    base_env,
                    log_root,
                )
                for index, gpu in enumerate(gpus)
            ]
            startup_errors: list[BaseException] = []
            for future in as_completed(futures):
                try:
                    workers.append(future.result())
                except BaseException as exc:
                    startup_errors.append(exc)
        if startup_errors:
            raise startup_errors[0]
        workers.sort(key=lambda row: row["worker_index"])
        manifest["servers"] = [_public_worker_record(worker) for worker in workers]
        manifest["status"] = "benchmarking_servers"
        _write_json(runtime_path, manifest)
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            benchmark_futures = {
                executor.submit(_benchmark_server, worker, base_env): worker
                for worker in workers
            }
            for future in as_completed(benchmark_futures):
                worker = benchmark_futures[future]
                worker["throughput_gate"] = future.result()
        manifest["servers"] = [_public_worker_record(worker) for worker in workers]
        failed_gates = [
            worker
            for worker in workers
            if worker["throughput_gate"]["status"] != "pass"
        ]
        _write_json(runtime_path, manifest)
        if failed_gates:
            details = ", ".join(
                "worker="
                f"{worker['worker_index']} gpu={worker['gpu']} "
                "aggregate_rate="
                f"{worker['throughput_gate']['measured_aggregate_completion_tokens_per_sec']}"
                for worker in failed_gates
            )
            raise RuntimeError(
                "vLLM aggregate throughput gate failed: "
                f"minimum={base_env.get('HABITBENCH_VLLM_MIN_TOKENS_PER_SEC', '60')} "
                f"tokens/s; {details}"
            )
        manifest["status"] = "running"
        _write_json(runtime_path, manifest)

        worker_pool: queue.Queue[dict[str, Any]] = queue.Queue()
        for worker in workers:
            worker_pool.put(worker)

        for group in groups:
            first = group[0]
            group_started = time.perf_counter()
            group_record: dict[str, Any] = {
                "method": first["method"],
                "dataset_name": first["dataset_name"],
                "dataset_dir": first["dataset_dir"],
                "method_output_root": first["method_output_root"],
                "shard_count": int(first["shard_count"]),
                "started_at": _utc_now(),
                "finished_at": None,
                "wall_clock_sec": None,
                "status": "running",
                "tasks": [],
            }
            manifest["groups"].append(group_record)
            _write_json(runtime_path, manifest)
            try:
                with ThreadPoolExecutor(max_workers=len(workers)) as executor:
                    futures = [
                        executor.submit(_run_task, row, worker_pool, base_env)
                        for row in group
                    ]
                    task_errors: list[BaseException] = []
                    for future in as_completed(futures):
                        try:
                            group_record["tasks"].append(future.result())
                        except BaseException as exc:
                            task_errors.append(exc)
                group_record["tasks"].sort(key=lambda row: row["shard_index"])
                if task_errors:
                    group_record["errors"] = [
                        f"{type(exc).__name__}: {exc}" for exc in task_errors
                    ]
                    raise task_errors[0]
            except BaseException:
                group_record["status"] = "failed"
                raise
            else:
                group_record["status"] = "succeeded"
            finally:
                group_record["finished_at"] = _utc_now()
                group_record["wall_clock_sec"] = round(
                    time.perf_counter() - group_started, 3
                )
                _write_json(runtime_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        raise
    else:
        manifest["status"] = "succeeded"
    finally:
        manifest["servers"] = [_public_worker_record(worker) for worker in workers]
        for worker in workers:
            _stop_server(worker)
        manifest["finished_at"] = _utc_now()
        manifest["wall_clock_sec"] = round(time.perf_counter() - suite_started, 3)
        _write_json(runtime_path, manifest)


if __name__ == "__main__":
    main()
