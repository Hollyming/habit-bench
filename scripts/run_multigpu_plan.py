#!/usr/bin/env python
"""Run a HABIT shard plan on one multi-GPU node with persistent vLLM workers."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.context_windows import resolve_context_window


O200K_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"
CL100K_CACHE_KEY = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
GPT2_CACHE_KEYS = {
    "gpt2_vocab_bpe": "6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
    "gpt2_encoder_json": "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
}
EMBEDDING_METHODS = {
    "mem0",
    "amem",
    "memos",
    "memrl",
    "lightmem",
    "letta",
    "mirix",
    "secom",
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
# These adapters load native local encoders/compressors on CUDA. They must see
# only the GPU paired with their vLLM worker; an empty global adapter setting
# is the normal CPU-isolation policy for the other methods.
CUDA_REQUIRED_ADAPTER_METHODS = {"mirix", "secom"}
ORACLE_METHODS = {"oracle_evidence", "oracle_habit_state"}
SHARD_DIR_PATTERN = re.compile(r"^shard_\d{3}_of_\d{3}$")
COMPLETION_FILES = (
    "worker_runtime.json",
    "run_manifest.json",
    "memory_contexts.jsonl",
    "predictions.jsonl",
    "scored_predictions.jsonl",
    "metrics.json",
)
NO_EXPECTED_METHOD_CONFIG = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **fields: Any) -> None:
    rendered = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"{_utc_now()} {event}{(' ' + rendered) if rendered else ''}", flush=True)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_mirix_vllm_profile(env: dict[str, str]) -> dict[str, Any]:
    """Require the compact xgrammar profile used by the local MIRIX bridge."""
    arguments = shlex.split(env.get("HABITBENCH_VLLM_EXTRA_ARGS", ""))
    if any(
        argument == "--reasoning-parser"
        or argument.startswith("--reasoning-parser=")
        for argument in arguments
    ):
        raise ValueError(
            "MIRIX must not enable --reasoning-parser when thinking is "
            "disabled: the JSON bridge requires response_format output in "
            "message.content"
        )
    option = "--structured-outputs-config"
    if option not in arguments:
        raise ValueError(
            "MIRIX requires --structured-outputs-config with compact xgrammar; "
            "source scripts/cluster/env.example.sh or restore the matching "
            "HABITBENCH_VLLM_EXTRA_ARGS value"
        )
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise ValueError(f"{option} is missing its JSON value")
    try:
        profile = json.loads(arguments[index + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} is not valid JSON: {exc}") from exc
    if (
        profile.get("backend") != "xgrammar"
        or profile.get("disable_any_whitespace") is not True
        or profile.get("disable_fallback") is not True
    ):
        raise ValueError(
            "MIRIX requires structured output backend=xgrammar and "
            "disable_any_whitespace=true, disable_fallback=true to keep "
            "bounded tool JSON finite and fail closed"
        )
    if profile.get("reasoning_parser"):
        raise ValueError(
            "MIRIX must not configure a structured-output reasoning_parser "
            "when thinking is disabled"
        )
    return profile


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
        "HABITBENCH_METHOD_IMPORT_TIMEOUT_SEC",
        "HABITBENCH_VLLM_INTERNAL_PORT_BASE",
        "HABITBENCH_VLLM_INTERNAL_PORT_STRIDE",
        "HABITBENCH_LIGHTMEM_MODEL",
        "HABITBENCH_SECOM_COMPRESSOR",
        "HABITBENCH_SECOM_REPO",
        "HABITBENCH_OMEM_REPO",
        "HABITBENCH_CONTEXT_WINDOW_TIER",
        "HABITBENCH_MAX_INPUT_TOKENS",
        "HABITBENCH_FULL_MEMORY_RESERVED_TOKENS",
        "HABITBENCH_FULL_MEMORY_MAX_TOKENS",
        "HABITBENCH_COMPACT_SUMMARY_MAX_TOKENS",
        "HABITBENCH_COMPACT_RECENT_TOKENS",
        "HABITBENCH_COMPACTOR_INPUT_TOKENS",
        "HABITBENCH_COMPACTOR_TIMEOUT_SEC",
        "HABITBENCH_COMPACTOR_MAX_RETRIES",
        "HABITBENCH_COMPACTOR_SEED",
        "HABITBENCH_GPU_MEMORY_UTIL",
        "HABITBENCH_MAX_MODEL_LEN",
        "HABITBENCH_ENABLE_PREFIX_CACHING",
        "HABITBENCH_VLLM_PYTHON",
        "HABITBENCH_VLLM_EXTRA_ARGS",
        "HABITBENCH_VLLM_MIN_TOKENS_PER_SEC",
        "HABITBENCH_VLLM_BENCHMARK_TOKENS",
        "HABITBENCH_VLLM_BENCHMARK_TIMEOUT_SEC",
        "HABITBENCH_VLLM_BENCHMARK_CONCURRENCY",
        "HABITBENCH_TASK_LOCK_POLL_SEC",
        "HABITBENCH_TASK_LOCK_LOG_EVERY_SEC",
        "HABITBENCH_MEMORY_LLM_MAX_TOKENS",
        "HABITBENCH_MEMORY_LLM_TEMPERATURE",
        "HABITBENCH_MEMORY_LLM_SEED",
        "MIRIX_JSON_TOOL_BRIDGE_SEED",
        "MIRIX_JSON_TOOL_BRIDGE_RETRY_TEMPERATURE",
        "HABITBENCH_GRAPHITI_LLM_MAX_TOKENS",
        "HABITBENCH_GRAPHITI_SCHEMA_MAX_ITEMS",
        "HABITBENCH_GRAPHITI_SCHEMA_MAX_STRING_CHARS",
        "HABITBENCH_GRAPHITI_REQUEST_TIMEOUT_SEC",
        "HABITBENCH_GRAPHITI_REQUEST_MAX_RETRIES",
        "HABITBENCH_GRAPHITI_USER_WORKERS",
        "HABITBENCH_OMEM_LLM_MAX_TOKENS",
        "HABITBENCH_OMEM_TOPIC_MERGE_MAX_TOKENS",
        "HABITBENCH_OMEM_REQUEST_TIMEOUT_SEC",
        "HABITBENCH_PROGRESS_EVERY",
        "HABITBENCH_OFFICIAL_TIMEOUT_SEC",
        "HABITBENCH_STRUCTURED_OUTPUT_MODE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
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
    mismatches = {}
    for name, value in expected.items():
        actual = versions.get(name)
        # Official CUDA wheels use the PEP 440 local tag (for example
        # 2.10.0+cu128), while PyPI may expose the same public Torch release as
        # 2.10.0. CUDA compatibility is checked independently below through
        # torch.version.cuda, so accept either spelling of the same public
        # release without weakening any other package pin.
        matches = (
            isinstance(actual, str)
            and actual.split("+", 1)[0] == value
            if name == "torch"
            else actual == value
        )
        if not matches:
            mismatches[name] = {"expected": value, "actual": actual}
    if mismatches:
        raise ValueError(f"Dedicated vLLM runtime mismatch: {mismatches}")
    return versions


def _require_package_set(
    installed_versions: dict[str, str],
    requirements: dict[str, tuple[str, str]],
    *,
    runtime_name: str,
) -> None:
    """Fail before GPU startup when a method's import-time dependencies drift."""
    missing: list[str] = []
    mismatches: dict[str, dict[str, str]] = {}
    for distribution, (module, expected_version) in requirements.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"{distribution}=={expected_version}")
            continue
        actual_version = package_version(distribution)
        installed_versions[distribution] = actual_version
        if actual_version != expected_version:
            mismatches[distribution] = {
                "expected": expected_version,
                "actual": actual_version,
            }
    if missing:
        raise ModuleNotFoundError(
            f"{runtime_name} runtime dependencies are missing: {', '.join(missing)}"
        )
    if mismatches:
        raise ValueError(
            f"{runtime_name} runtime dependency mismatch: {mismatches}"
        )


def _preflight_method_imports(
    python_bin: Path,
    env: dict[str, str],
    methods: set[str],
) -> dict[str, Any]:
    """Import every selected adapter in a fresh method-runtime process.

    MemOS and MemRL load their vendored implementations lazily, so their smoke
    imports deliberately traverse the same deep modules used by constructors.
    Keeping methods isolated also catches namespace-package contamination.
    """

    med_root = Path(
        env.get(
            "HABITBENCH_MEDMEMORYBENCH_ROOT",
            str(PROJECT_ROOT / "third_party/medmemorybench"),
        )
    ).expanduser().resolve()
    script = r"""
import importlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
method = sys.argv[2]
sys.path.insert(0, str(root))
importlib.import_module(f"methods.{method}_agent")
loaded = [f"methods.{method}_agent"]

if method in {"memos", "memrl"}:
    methods_root = root / "methods"
    memos_src = methods_root / "memOS" / "MemOS" / "src"
    sys.path.insert(0, str(memos_src))
    importlib.import_module("prometheus_client")
    importlib.import_module("memos.utils")
    importlib.import_module("memos.memories.factory")
    importlib.import_module("memos.configs.memory")
    loaded.extend([
        "prometheus_client",
        "memos.utils",
        "memos.memories.factory",
        "memos.configs.memory",
    ])

if method == "mirix":
    importlib.import_module("json_repair")
    loaded.append("json_repair")

if method == "memrl":
    memrl_root = root / "methods" / "MemRL"
    sys.path.insert(0, str(memrl_root))
    for module in (
        "memos.mem_os.main",
        "memrl.providers.embedding",
        "memrl.service.memory_service",
        "memrl.service.strategies",
        "memrl.service.value_driven",
    ):
        importlib.import_module(module)
        loaded.append(module)

print(json.dumps({"status": "pass", "modules": loaded}, sort_keys=True))
"""
    timeout_sec = float(env.get("HABITBENCH_METHOD_IMPORT_TIMEOUT_SEC", "300"))
    if timeout_sec <= 0:
        raise ValueError("HABITBENCH_METHOD_IMPORT_TIMEOUT_SEC must be positive")
    results: dict[str, Any] = {}
    for method in sorted(methods & MEDMEMORY_METHODS):
        try:
            completed = subprocess.run(
                [str(python_bin), "-c", script, str(med_root), method],
                check=False,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{method} adapter import preflight exceeded "
                f"HABITBENCH_METHOD_IMPORT_TIMEOUT_SEC={timeout_sec:g}"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"{method} adapter import preflight failed with code "
                f"{completed.returncode}: {completed.stderr[-8000:]}"
            )
        try:
            results[method] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{method} adapter import preflight returned invalid output: "
                f"{completed.stdout[-2000:]}"
            ) from exc
    return results


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
    if methods and "memrl" in methods:
        if importlib.util.find_spec("chonkie") is None:
            raise ModuleNotFoundError(
                "chonkie==1.2.1 is required when memrl is selected"
            )
        packages["chonkie"] = package_version("chonkie")
        if packages["chonkie"] != "1.2.1":
            raise ValueError(
                "MemRL runtime dependency mismatch: expected chonkie 1.2.1, "
                f"got {packages['chonkie']}"
            )
        required.update(
            {
                name: tiktoken_root / cache_key
                for name, cache_key in GPT2_CACHE_KEYS.items()
            }
        )
    if methods and ({"memos", "memrl"} & methods):
        _require_package_set(
            packages,
            {"prometheus-client": ("prometheus_client", "0.23.1")},
            runtime_name="MemOS/MemRL",
        )
    if methods and "letta" in methods:
        required["tiktoken_cl100k"] = tiktoken_root / CL100K_CACHE_KEY
        _require_package_set(
            packages,
            {
                "anthropic": ("anthropic", "0.49.0"),
                "APScheduler": ("apscheduler", "3.11.0"),
                "composio_core": ("composio", "0.7.15"),
                "datamodel-code-generator": (
                    "datamodel_code_generator",
                    "0.25.9",
                ),
                "demjson3": ("demjson3", "3.0.6"),
                "docstring-parser": ("docstring_parser", "0.16"),
                "mcp": ("mcp", "1.6.0"),
                "llama-index-core": ("llama_index.core", "0.12.30"),
                "opentelemetry-instrumentation-requests": (
                    "opentelemetry.instrumentation.requests",
                    "0.65b0",
                ),
                "pathvalidate": ("pathvalidate", "3.2.3"),
                "pyhumps": ("humps", "3.8.0"),
                "SQLAlchemy-Utils": ("sqlalchemy_utils", "0.41.2"),
                "sqlalchemy-json": ("sqlalchemy_json", "0.7.0"),
                "sqlmodel": ("sqlmodel", "0.0.16"),
            },
            runtime_name="Letta",
        )
    mirix_vllm_profile = None
    if methods and "mirix" in methods:
        mirix_vllm_profile = _validate_mirix_vllm_profile(env)
        _require_package_set(
            packages,
            {
                "aiosqlite": ("aiosqlite", "0.22.1"),
                "json-repair": ("json_repair", "0.53.0"),
                "langfuse": ("langfuse", "3.15.0"),
                "pydub": ("pydub", "0.25.1"),
                "RapidFuzz": ("rapidfuzz", "3.14.5"),
                "SpeechRecognition": (
                    "speech_recognition",
                    "3.17.0",
                ),
            },
            runtime_name="MIRIX",
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
    method_imports = _preflight_method_imports(
        python_bin,
        env,
        methods or set(),
    )
    return {
        "status": "pass",
        "files": {
            name: {"path": str(path), "size_bytes": path.stat().st_size}
            for name, path in required.items()
        },
        "embedding_identity": identity,
        "packages": packages,
        "vllm_runtime": vllm_versions,
        "mirix_vllm_profile": mirix_vllm_profile,
        "full_memory_window": full_memory_window,
        "method_imports": method_imports,
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


class DistributedTaskCoordinator:
    """A GPFS-backed queue using atomic per-task directory creation.

    Mutable JSON cursors are intentionally avoided: their read/replace cycle
    was observed to lose updates across H-cluster nodes even while guarded by
    ``flock``. POSIX directory creation is a single metadata operation, so a
    task claim has exactly one winner on the shared filesystem.
    """

    CONTRACT_VERSION = "habitbench.distributed_queue.v2"

    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        *,
        coordinator_id: str,
        plan_sha256: str,
        replica_count: int,
    ) -> None:
        if replica_count < 1:
            raise ValueError("replica_count must be positive")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", coordinator_id):
            raise ValueError(f"Unsafe coordinator id: {coordinator_id!r}")
        task_ids = [int(row["task_id"]) for row in rows]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Plan contains duplicate task_id values")
        self.rows = rows
        self.coordinator_id = coordinator_id
        self.plan_sha256 = plan_sha256
        self.replica_count = replica_count
        self.root = root.expanduser().resolve() / coordinator_id
        self.contract_root = self.root / "contract"
        self.contract_path = self.contract_root / "contract.json"
        self.claim_root = self.root / "claims"
        self.result_root = self.root / "results"
        self._scan_lock = threading.Lock()
        self._next_hint = 0
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.contract_root.mkdir()
        except FileExistsError:
            # The other Replica may still be publishing its immutable
            # contract. _write_json makes the final pathname visible only
            # after the complete payload has been written.
            deadline = time.monotonic() + 30
            while not self.contract_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Distributed queue contract was not published: {self.contract_path}"
                    )
                time.sleep(0.1)
        else:
            _write_json(
                self.contract_path,
                {
                    "contract_version": self.CONTRACT_VERSION,
                    "claim_protocol": "atomic-mkdir-per-task",
                    "coordinator_id": coordinator_id,
                    "plan_sha256": plan_sha256,
                    "replica_count": replica_count,
                    "task_count": len(rows),
                    "created_at": _utc_now(),
                },
            )
        self._validate_contract(
            json.loads(self.contract_path.read_text(encoding="utf-8"))
        )
        self.claim_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)

    def _validate_contract(self, contract: dict[str, Any]) -> None:
        expected = {
            "contract_version": self.CONTRACT_VERSION,
            "claim_protocol": "atomic-mkdir-per-task",
            "coordinator_id": self.coordinator_id,
            "plan_sha256": self.plan_sha256,
            "replica_count": self.replica_count,
            "task_count": len(self.rows),
        }
        mismatches = {
            key: {"expected": value, "actual": contract.get(key)}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Distributed queue contract mismatch at {self.root}: {mismatches}"
            )

    def claim_next(
        self,
        *,
        replica_index: int,
        worker_index: int,
        host: str,
    ) -> tuple[dict[str, str], dict[str, Any]] | None:
        # Serialize only this process's worker threads. Cross-process and
        # cross-node exclusion comes solely from atomic mkdir below.
        with self._scan_lock:
            for ordinal in range(self._next_hint, len(self.rows)):
                claim_dir = self.claim_root / f"task-{ordinal:06d}"
                try:
                    claim_dir.mkdir()
                except FileExistsError:
                    self._next_hint = ordinal + 1
                    continue
                self._next_hint = ordinal + 1
                row = self.rows[ordinal]
                claim = {
                    "ordinal": ordinal,
                    "task_id": int(row["task_id"]),
                    "method": row["method"],
                    "dataset_name": row["dataset_name"],
                    "shard_index": int(row["shard_index"]),
                    "shard_count": int(row["shard_count"]),
                    "replica_index": replica_index,
                    "worker_index": worker_index,
                    "host": host,
                    "claim_protocol": "atomic-mkdir-per-task",
                    "claimed_at": _utc_now(),
                }
                _write_json(claim_dir / "claim.json", claim)
                return row, claim
        return None

    def aggregate_state(self) -> dict[str, Any]:
        claimed = sum(
            1
            for path in self.claim_root.glob("task-*")
            if path.is_dir()
        )
        result_paths = list(self.result_root.glob("task-*.status-*.json"))
        statuses = [
            path.name.split(".status-", 1)[1].rsplit(".json", 1)[0]
            for path in result_paths
        ]
        skipped = statuses.count("skipped_completed")
        completed = statuses.count("succeeded") + skipped
        failed = statuses.count("failed")
        return {
            "contract_version": self.CONTRACT_VERSION,
            "claim_protocol": "atomic-mkdir-per-task",
            "coordinator_id": self.coordinator_id,
            "plan_sha256": self.plan_sha256,
            "replica_count": self.replica_count,
            "task_count": len(self.rows),
            "claimed": claimed,
            "finished": len(result_paths),
            "succeeded": completed,
            "failed": failed,
            "skipped_completed": skipped,
            "active": claimed - len(result_paths),
            "unclaimed": len(self.rows) - claimed,
            "observed_at": _utc_now(),
        }

    def finish(self, claim: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        ordinal = int(claim["ordinal"])
        status = str(record.get("status"))
        if status not in {"succeeded", "skipped_completed", "failed"}:
            status = "failed"
        existing = list(
            self.result_root.glob(f"task-{ordinal:06d}.status-*.json")
        )
        if existing:
            raise RuntimeError(
                f"Distributed task already finished: {existing[0]}"
            )
        result_path = (
            self.result_root / f"task-{ordinal:06d}.status-{status}.json"
        )
        _write_json(
            result_path,
            {
                "claim": claim,
                "status": status,
                "returncode": record.get("returncode"),
                "finished_at": _utc_now(),
            },
        )
        return self.aggregate_state()


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
    internal_port_base = int(
        env.get("HABITBENCH_VLLM_INTERNAL_PORT_BASE", "20000")
    )
    internal_port_stride = int(
        env.get("HABITBENCH_VLLM_INTERNAL_PORT_STRIDE", "64")
    )
    if internal_port_base <= 0 or internal_port_stride <= 0:
        raise ValueError(
            "HABITBENCH_VLLM_INTERNAL_PORT_BASE and "
            "HABITBENCH_VLLM_INTERNAL_PORT_STRIDE must be positive"
        )
    internal_port = internal_port_base + worker_index * internal_port_stride
    if internal_port + internal_port_stride - 1 > 65535:
        raise ValueError(
            "vLLM internal worker port range exceeds TCP port 65535: "
            f"worker={worker_index} base={internal_port_base} "
            f"stride={internal_port_stride}"
        )
    # vLLM otherwise probes an ephemeral port in every child process. Eight
    # servers launched concurrently can observe the same free port and then
    # race while binding it. Give every worker a disjoint deterministic range;
    # vLLM starts at VLLM_PORT and increments only within that worker's range
    # under ordinary transient conflicts.
    worker_env["VLLM_PORT"] = str(internal_port)
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
        "internal_port": internal_port,
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


def _apply_config_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return the raw method config with plan-declared runtime overrides."""

    normalized = json.loads(json.dumps(config))
    for dotted_name, value in overrides.items():
        names = dotted_name.split(".")
        target = normalized
        for name in names[:-1]:
            child = target.get(name)
            if not isinstance(child, dict):
                child = {}
                target[name] = child
            target = child
        target[names[-1]] = value
    return normalized


def _method_config_matches_plan(observed: Any, expected: Any) -> bool:
    """Validate checkpoint behavior against the plan's effective config.

    The plan stores the effective config after model/path overrides, whereas a
    shard manifest snapshots the raw YAML.  Reapply the recorded overrides and
    compare parsed values.  The YAML byte hash is intentionally not compared:
    comment-only edits must not invalidate a semantically identical result.
    """

    if expected is NO_EXPECTED_METHOD_CONFIG:
        return True
    if expected is None:
        return observed is None
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return False
    if observed.get("name") != expected.get("name"):
        return False
    observed_config = observed.get("config")
    expected_config = expected.get("config")
    overrides = expected.get("path_overrides") or {}
    if (
        not isinstance(observed_config, dict)
        or not isinstance(expected_config, dict)
        or not isinstance(overrides, dict)
    ):
        return False
    return _apply_config_overrides(observed_config, overrides) == expected_config


def _expected_method_config_for_row(
    row: dict[str, str], plan_manifest: dict[str, Any] | None
) -> Any:
    methods = (plan_manifest or {}).get("methods")
    if not isinstance(methods, dict) or row["method"] not in methods:
        return NO_EXPECTED_METHOD_CONFIG
    return methods[row["method"]]


def _completed_task_record(
    output_dir: Path,
    *,
    expected_method_config: Any = NO_EXPECTED_METHOD_CONFIG,
) -> dict[str, Any] | None:
    """Return the atomic shard checkpoint only when every final artifact exists."""

    runtime_path = output_dir / "worker_runtime.json"
    if not runtime_path.is_file():
        return None
    try:
        record = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != "succeeded":
        return None
    if any(not (output_dir / name).is_file() for name in COMPLETION_FILES):
        return None
    try:
        run_manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if (run_manifest.get("execution") or {}).get("status") != "succeeded":
        return None
    if not _method_config_matches_plan(
        run_manifest.get("method_config"), expected_method_config
    ):
        return None
    return record


def _prepare_task_output(
    output_dir: Path,
    *,
    force_rerun: bool,
    expected_method_config: Any = NO_EXPECTED_METHOD_CONFIG,
) -> bool:
    """Keep a verified checkpoint; remove failed/interrupted state before rerun."""

    if not SHARD_DIR_PATTERN.fullmatch(output_dir.name):
        raise ValueError(f"Refusing non-shard output directory: {output_dir}")
    was_complete = (
        not force_rerun
        and _completed_task_record(
            output_dir, expected_method_config=expected_method_config
        )
        is not None
    )
    if output_dir.exists() and not was_complete:
        shutil.rmtree(output_dir)
    if not was_complete:
        output_dir.mkdir(parents=True, exist_ok=True)
    return was_complete


def _task_output_lock_path(output_dir: Path) -> Path:
    """Return a persistent lock file shared by every run of one shard."""

    if not SHARD_DIR_PATTERN.fullmatch(output_dir.name):
        raise ValueError(f"Refusing non-shard output directory: {output_dir}")
    return (
        output_dir.parent
        / ".habitbench-shard-locks"
        / f"{output_dir.name}.lock"
    )


@contextmanager
def _exclusive_task_output_lock(
    output_dir: Path,
    *,
    label: str,
    poll_sec: float = 5.0,
    log_every_sec: float = 60.0,
):
    """Serialize all writers to one persistent shard output directory.

    Distributed queue claims are intentionally scoped to one coordinator. A
    second RJob can therefore have a different queue while targeting the same
    output shard.  Holding a POSIX lock across checkpoint validation, partial
    output cleanup, execution, and final marker publication prevents those
    independent jobs from deleting or mutating each other's state.  The kernel
    releases the lock automatically if a worker or RJob is terminated.
    """

    if poll_sec <= 0:
        raise ValueError(f"poll_sec must be positive: {poll_sec}")
    if log_every_sec <= 0:
        raise ValueError(f"log_every_sec must be positive: {log_every_sec}")

    lock_path = _task_output_lock_path(output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    wait_started = time.monotonic()
    next_log_at = wait_started
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now >= next_log_at:
                    handle.seek(0)
                    owner = handle.read().strip().replace("\n", " ") or "unknown"
                    _log(
                        "task_output_lock_wait",
                        task=label,
                        waited_sec=round(now - wait_started, 1),
                        lock=lock_path,
                        owner=owner,
                    )
                    next_log_at = now + log_every_sec
                time.sleep(poll_sec)

        acquired_at = _utc_now()
        waited_sec = round(time.monotonic() - wait_started, 3)
        owner = {
            "acquired_at": acquired_at,
            "host": socket.gethostname(),
            "job_id": os.environ.get("JOB_ID"),
            "pid": os.getpid(),
            "task": label,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        _log(
            "task_output_lock_acquired",
            task=label,
            waited_sec=waited_sec,
            lock=lock_path,
        )
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        _log("task_output_lock_released", task=label, lock=lock_path)


def _run_streamed_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    label: str,
) -> tuple[int, list[str]]:
    """Persist child logs while relaying prefixed lines to RJob stdout/stderr."""

    tail: deque[str] = deque(maxlen=40)
    tail_lock = threading.Lock()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def relay(
        source: Any,
        path: Path,
        stream_name: str,
        console: Any,
    ) -> None:
        with path.open("a", encoding="utf-8") as sink:
            for line in iter(source.readline, ""):
                sink.write(line)
                sink.flush()
                clean = line.rstrip("\n")
                with tail_lock:
                    tail.append(f"{stream_name}: {clean}")
                print(f"[{label} {stream_name}] {clean}", file=console, flush=True)
        source.close()

    threads = [
        threading.Thread(
            target=relay,
            args=(process.stdout, stdout_path, "stdout", sys.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=relay,
            args=(process.stderr, stderr_path, "stderr", sys.stderr),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return returncode, list(tail)


def _run_task(
    row: dict[str, str],
    worker: dict[str, Any],
    base_env: dict[str, str],
    expected_method_config: Any = NO_EXPECTED_METHOD_CONFIG,
) -> dict[str, Any]:
    shard_index = int(row["shard_index"])
    shard_count = int(row["shard_count"])
    shard_name = f"shard_{shard_index:03d}_of_{shard_count:03d}"
    output_dir = Path(row["method_output_root"]) / shard_name
    label = f"{row['method']}/{row['dataset_name']}/{shard_name}"
    poll_sec = float(base_env.get("HABITBENCH_TASK_LOCK_POLL_SEC", "5"))
    log_every_sec = float(
        base_env.get("HABITBENCH_TASK_LOCK_LOG_EVERY_SEC", "60")
    )
    with _exclusive_task_output_lock(
        output_dir,
        label=label,
        poll_sec=poll_sec,
        log_every_sec=log_every_sec,
    ):
        return _run_task_with_output_lock(
            row,
            worker,
            base_env,
            expected_method_config=expected_method_config,
        )


def _run_task_with_output_lock(
    row: dict[str, str],
    worker: dict[str, Any],
    base_env: dict[str, str],
    *,
    expected_method_config: Any = NO_EXPECTED_METHOD_CONFIG,
) -> dict[str, Any]:
    shard_index = int(row["shard_index"])
    shard_count = int(row["shard_count"])
    shard_name = f"shard_{shard_index:03d}_of_{shard_count:03d}"
    output_dir = Path(row["method_output_root"]) / shard_name
    was_complete = _prepare_task_output(
        output_dir,
        force_rerun=base_env.get("HABITBENCH_FORCE_RERUN", "0") == "1",
        expected_method_config=expected_method_config,
    )
    label = f"{row['method']}/{row['dataset_name']}/{shard_name}"
    if was_complete:
        checkpoint = _completed_task_record(
            output_dir, expected_method_config=expected_method_config
        )
        if checkpoint is None:
            raise RuntimeError(f"Checkpoint disappeared while resuming: {output_dir}")
        _log("task_skip_checkpoint", task=label)
        resumed_at = _utc_now()
        return {
            **checkpoint,
            "status": "skipped_completed",
            "checkpoint_status": "succeeded",
            "checkpoint_finished_at": checkpoint.get("finished_at"),
            "resumed_at": resumed_at,
            "finished_at": resumed_at,
        }

    task_env = dict(base_env)
    task_env["PYTHONUNBUFFERED"] = "1"
    method = row["method"]
    if method in CUDA_REQUIRED_ADAPTER_METHODS:
        adapter_cuda = worker["gpu"]
    elif "HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES" in base_env:
        adapter_cuda = base_env["HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES"]
    elif base_env.get("HABITBENCH_EMBED_DEVICE", "cpu").lower() == "cpu":
        adapter_cuda = ""
    else:
        adapter_cuda = worker["gpu"]
    task_env["CUDA_VISIBLE_DEVICES"] = adapter_cuda
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
    if method in ORACLE_METHODS:
        command = [
            sys.executable,
            "-m",
            "eval.supplementary.oracle_controls",
            "--dataset-dir",
            row["dataset_dir"],
            "--output-dir",
            str(output_dir),
            "--mode",
            method,
            "--user-shard-index",
            str(shard_index),
            "--user-shard-count",
            str(shard_count),
            "--base-model",
            base_env.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
            "--base-model-path",
            base_env.get(
                "HABITBENCH_LLM_MODEL",
                "/plm-shared/zhangjunming/Workspace/models/Qwen3-8B",
            ),
            "--base-url",
            worker["base_url"],
            "--progress-every",
            base_env.get("HABITBENCH_PROGRESS_EVERY", "25"),
        ]
    else:
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
    if row.get("domain_filter"):
        command.extend(["--domain-filter", row["domain_filter"]])
    if row.get("max_users"):
        command.extend(["--max-users", row["max_users"]])
    if row.get("max_probes"):
        command.extend(["--max-probes", row["max_probes"]])
    stdout_path = output_dir / "task.stdout.log"
    stderr_path = output_dir / "task.stderr.log"
    started_at = _utc_now()
    started = time.perf_counter()
    returncode: int | None = None
    log_tail: list[str] = []
    _log(
        "task_start",
        task=label,
        worker=worker["worker_index"],
        gpu=worker["gpu"],
    )
    try:
        returncode, log_tail = _run_streamed_command(
            command,
            cwd=PROJECT_ROOT,
            env=task_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label=label,
        )
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
            "domain_filter": row.get("domain_filter") or None,
            "max_users": int(row["max_users"]) if row.get("max_users") else None,
            "max_probes": int(row["max_probes"]) if row.get("max_probes") else None,
            "output_dir": str(output_dir),
            "incomplete_output_policy": "remove_before_retry",
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
            "status": "succeeded" if returncode == 0 and error is None else "failed",
            "returncode": returncode,
            "error_type": error_type,
            "error": error,
            "log_tail": log_tail,
            "command": command,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
    }
    if record["status"] == "succeeded":
        _write_json(output_dir / "worker_runtime.json", record)
        if (
            _completed_task_record(
                output_dir, expected_method_config=expected_method_config
            )
            is None
        ):
            record.update(
                {
                    "status": "failed",
                    "error_type": "IncompleteSuccessArtifacts",
                    "error": "process exited zero but final checkpoint validation failed",
                }
            )
    if record["status"] == "failed":
        _log(
            "task_failed",
            task=label,
            returncode=returncode,
            error=error or record.get("error"),
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        record["failed_output_removed"] = True
    else:
        _log(
            "task_succeeded",
            task=label,
            elapsed_sec=record["wall_clock_sec"],
        )
    return record


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
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument(
        "--coordination-root",
        type=Path,
        help="Persistent shared root for cross-Replica dynamic task claims.",
    )
    parser.add_argument(
        "--coordinator-id",
        help="Unique launch id below --coordination-root; defaults to JOB_ID.",
    )
    parser.add_argument(
        "--continue-on-group-error",
        action="store_true",
        help=(
            "Compatibility flag. Dynamic scheduling always records task failures "
            "and lets other GPUs continue so the final merge can report coverage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    gpus = _split_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    global_rows = _load_plan(plan_path)
    global_groups = _group_rows(global_rows)
    if (
        args.replica_count < 1
        or args.replica_index < 0
        or args.replica_index >= args.replica_count
    ):
        raise ValueError(
            f"Invalid replica index/count: {args.replica_index}/{args.replica_count}"
        )
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
        if plan_manifest.get("task_count") != len(global_rows):
            raise ValueError(
                "Plan task count no longer matches "
                f"{plan_manifest_path}: {len(global_rows)}"
            )
    base_env = os.environ.copy()
    if args.env_file:
        base_env = _source_env_file(args.env_file, base_env)

    coordinator_id = args.coordinator_id or base_env.get("JOB_ID")
    if not coordinator_id:
        if args.replica_count > 1:
            raise ValueError(
                "Multi-Replica scheduling requires --coordinator-id or RJob JOB_ID"
            )
        coordinator_id = f"manual-{plan_sha256[:12]}-{os.getpid()}"
    coordination_root = (
        args.coordination_root.expanduser().resolve()
        if args.coordination_root
        else plan_path.parent / "distributed_queue"
    )
    coordinator = DistributedTaskCoordinator(
        coordination_root,
        global_rows,
        coordinator_id=coordinator_id,
        plan_sha256=plan_sha256,
        replica_count=args.replica_count,
    )

    runtime_path = (
        args.runtime_output.expanduser().resolve()
        if args.runtime_output
        else plan_path.parent / "suite_runtime.json"
    )
    log_root = (
        args.log_root.expanduser().resolve()
        if args.log_root
        else plan_path.parent / "vllm_logs"
    )
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
        "launcher": {
            **((plan_manifest or {}).get("launcher") or {}),
            "runtime_job_id": base_env.get("JOB_ID"),
            "replicas": str(args.replica_count),
            "gpus_per_replica": str(len(gpus)),
            "total_gpus": str(len(gpus) * args.replica_count),
        },
        "gpu_count": len(gpus),
        "gpus": gpus,
        "replica_index": args.replica_index,
        "replica_count": args.replica_count,
        "task_count": 0,
        "global_task_count": len(global_rows),
        "group_count": 0,
        "distributed_queue": {
            "contract_version": coordinator.CONTRACT_VERSION,
            "coordinator_id": coordinator_id,
            "root": str(coordinator.root),
            "policy": "global-dynamic-shard-claims",
        },
        "config": _safe_config(base_env),
        "preflight": None,
        "servers": [],
        "groups": [],
    }
    _write_json(runtime_path, manifest)
    checkpoint_count = sum(
        _completed_task_record(
            Path(row["method_output_root"])
            / f"shard_{int(row['shard_index']):03d}_of_{int(row['shard_count']):03d}",
            expected_method_config=_expected_method_config_for_row(
                row, plan_manifest
            ),
        )
        is not None
        for row in global_rows
    )
    _log(
        "suite_start",
        replica=f"{args.replica_index}/{args.replica_count}",
        local_workers=len(gpus),
        global_tasks=len(global_rows),
        reusable_checkpoints=checkpoint_count,
        pending_tasks=len(global_rows) - checkpoint_count,
        coordinator=coordinator.root,
        runtime=runtime_path,
    )

    workers: list[dict[str, Any]] = []
    task_failure_count = 0
    try:
        manifest["preflight"] = _preflight_runtime(
            base_env, {row["method"] for row in global_rows}
        )
        _log("runtime_preflight_passed", replica=args.replica_index)
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
                    worker = future.result()
                    workers.append(worker)
                    _log(
                        "vllm_server_ready",
                        replica=args.replica_index,
                        worker=worker["worker_index"],
                        gpu=worker["gpu"],
                        port=worker["port"],
                        startup_sec=worker["startup_wall_clock_sec"],
                    )
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
                _log(
                    "vllm_throughput_pass",
                    replica=args.replica_index,
                    worker=worker["worker_index"],
                    tokens_per_sec=worker["throughput_gate"][
                        "measured_aggregate_completion_tokens_per_sec"
                    ],
                )
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

        manifest_lock = threading.Lock()
        group_records: dict[tuple[str, str, str], dict[str, Any]] = {}

        def group_key(row: dict[str, str]) -> tuple[str, str, str]:
            return (
                row["method"],
                row["dataset_name"],
                str(Path(row["method_output_root"]).resolve()),
            )

        def register_claim(
            row: dict[str, str], claim: dict[str, Any]
        ) -> dict[str, Any]:
            key = group_key(row)
            with manifest_lock:
                record = group_records.get(key)
                if record is None:
                    record = {
                        "method": row["method"],
                        "dataset_name": row["dataset_name"],
                        "dataset_dir": row["dataset_dir"],
                        "domain_filter": row.get("domain_filter") or None,
                        "max_users": (
                            int(row["max_users"]) if row.get("max_users") else None
                        ),
                        "max_probes": (
                            int(row["max_probes"]) if row.get("max_probes") else None
                        ),
                        "method_output_root": row["method_output_root"],
                        "shard_count": int(row["shard_count"]),
                        "started_at": claim["claimed_at"],
                        "finished_at": None,
                        "wall_clock_sec": None,
                        "status": "running",
                        "tasks": [],
                    }
                    group_records[key] = record
                    manifest["groups"].append(record)
                return record

        def record_completion(
            row: dict[str, str],
            task_record: dict[str, Any],
            queue_state: dict[str, Any],
        ) -> None:
            key = group_key(row)
            with manifest_lock:
                group_record = group_records[key]
                group_record["tasks"].append(task_record)
                group_record["tasks"].sort(
                    key=lambda item: int(item.get("shard_index", -1))
                )
                group_record["finished_at"] = task_record.get("finished_at") or _utc_now()
                group_record["wall_clock_sec"] = round(
                    (
                        datetime.fromisoformat(group_record["finished_at"])
                        - datetime.fromisoformat(group_record["started_at"])
                    ).total_seconds(),
                    3,
                )
                if task_record.get("status") == "failed":
                    group_record["status"] = "failed"
                manifest["task_count"] = sum(
                    len(item["tasks"]) for item in manifest["groups"]
                )
                manifest["group_count"] = len(manifest["groups"])
                manifest["queue_progress"] = queue_state
                _write_json(runtime_path, manifest)

        def run_worker(worker: dict[str, Any]) -> int:
            local_failures = 0
            while True:
                claimed = coordinator.claim_next(
                    replica_index=args.replica_index,
                    worker_index=int(worker["worker_index"]),
                    host=socket.gethostname(),
                )
                if claimed is None:
                    return local_failures
                row, claim = claimed
                register_claim(row, claim)
                _log(
                    "queue_claim",
                    replica=args.replica_index,
                    worker=worker["worker_index"],
                    ordinal=claim["ordinal"],
                    task=claim["task_id"],
                    method=row["method"],
                    dataset=row["dataset_name"],
                    shard=row["shard_index"],
                )
                try:
                    task_record = _run_task(
                        row,
                        worker,
                        base_env,
                        expected_method_config=_expected_method_config_for_row(
                            row, plan_manifest
                        ),
                    )
                except BaseException as exc:
                    task_record = {
                        "contract_version": "habitbench.shard_worker.v1",
                        "task_id": int(row["task_id"]),
                        "method": row["method"],
                        "dataset_name": row["dataset_name"],
                        "output_dir": row["method_output_root"],
                        "shard_index": int(row["shard_index"]),
                        "shard_count": int(row["shard_count"]),
                        "worker_index": worker["worker_index"],
                        "gpu": worker["gpu"],
                        "started_at": claim["claimed_at"],
                        "finished_at": _utc_now(),
                        "status": "failed",
                        "returncode": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    _log(
                        "task_failed_unexpected",
                        task=claim["task_id"],
                        error_type=type(exc).__name__,
                        error=exc,
                    )
                task_record["distributed_claim"] = claim
                if task_record.get("status") == "failed":
                    local_failures += 1
                queue_state = coordinator.finish(claim, task_record)
                record_completion(row, task_record, queue_state)
                _log(
                    "queue_progress",
                    replica=args.replica_index,
                    worker=worker["worker_index"],
                    finished=queue_state["finished"],
                    total=queue_state["task_count"],
                    succeeded=queue_state["succeeded"],
                    failed=queue_state["failed"],
                    status=task_record.get("status"),
                )

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [executor.submit(run_worker, worker) for worker in workers]
            task_failure_count = sum(future.result() for future in as_completed(futures))

        with manifest_lock:
            for group_record in manifest["groups"]:
                if group_record["status"] == "running":
                    group_record["status"] = "succeeded"
            manifest["groups"].sort(
                key=lambda item: (
                    item["method"],
                    item["dataset_name"],
                    item["method_output_root"],
                )
            )
            manifest["task_failure_count"] = task_failure_count
            _write_json(runtime_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        raise
    else:
        manifest["status"] = (
            "completed_with_task_failures"
            if task_failure_count
            else "succeeded"
        )
    finally:
        manifest["servers"] = [_public_worker_record(worker) for worker in workers]
        for worker in workers:
            _stop_server(worker)
        manifest["finished_at"] = _utc_now()
        manifest["wall_clock_sec"] = round(time.perf_counter() - suite_started, 3)
        _write_json(runtime_path, manifest)
        _log(
            "suite_finished",
            replica=f"{args.replica_index}/{args.replica_count}",
            status=manifest["status"],
            elapsed_sec=manifest["wall_clock_sec"],
        )


if __name__ == "__main__":
    main()
