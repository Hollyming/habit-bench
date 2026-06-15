#!/usr/bin/env python
"""Preflight a Lumia HABIT-Bench full official run.

This check is intentionally local to the Lumia workspace: it verifies the
dataset contract, dependency imports, disk headroom, and GPU availability before
the expensive model download / vLLM / method runs begin.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DATASET = Path(
    os.getenv(
        "HABITBENCH_DATASET",
        "./runs/habit_bench_balanced_v0_3_official_subset_90",
    )
)
DEFAULT_RESULTS = Path(os.getenv("HABITBENCH_OFFICIAL_OUT", str(DEFAULT_DATASET / "full_official_results")))
DEFAULT_OUT = Path(
    os.getenv(
        "HABITBENCH_LUMIA_PREFLIGHT_MANIFEST",
        "./runs/lumia_manifests/lumia_run_preflight_manifest.json",
    )
)

REQUIRED_MODULES = [
    "huggingface_hub",
    "openai",
    "mem0",
    "graphiti_core",
    "kuzu",
    "sentence_transformers",
    "vllm",
]

ENV_KEYS = [
    "HABITBENCH_LLM_MODEL",
    "HABITBENCH_SERVED_MODEL",
    "HABITBENCH_EMBED_MODEL",
    "HABITBENCH_STRUCTURED_OUTPUT_MODE",
    "HABITBENCH_DATASET",
    "HABITBENCH_OFFICIAL_OUT",
    "HABITBENCH_SKIP_MODEL_DOWNLOAD",
    "HABITBENCH_REUSE_SERVER",
    "HABITBENCH_MIN_FREE_GB",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "CUDA_VISIBLE_DEVICES",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        return {"command": command, "available": False, "returncode": None, "error": str(exc)}
    except Exception as exc:
        return {
            "command": command,
            "available": True,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def existing_anchor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def disk_row(label: str, path: Path) -> Dict[str, Any]:
    anchor = existing_anchor(path)
    usage = shutil.disk_usage(anchor)
    return {
        "label": label,
        "path": str(path),
        "anchor": str(anchor),
        "total_gb": round(usage.total / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
    }


def selected_env() -> Dict[str, str | None]:
    values: Dict[str, str | None] = {}
    for key in ENV_KEYS:
        value = os.getenv(key)
        if "KEY" in key or "TOKEN" in key:
            value = "<redacted>" if value else None
        values[key] = value
    return values


def check_imports(errors: List[str]) -> Dict[str, bool]:
    imports = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    for name, ok in imports.items():
        if not ok:
            errors.append(f"missing_python_module:{name}")
    return imports


def check_dataset(dataset_dir: Path, errors: List[str]) -> Dict[str, Any]:
    paths = {
        "public_probes": dataset_dir / "public" / "probes.jsonl",
        "public_sessions": dataset_dir / "public" / "lifelines.jsonl",
        "private_key": dataset_dir / "private" / "probe_key.jsonl",
        "manifest": dataset_dir / "reports" / "official_subset_manifest.json",
        "provenance": dataset_dir / "reports" / "domain_provenance_summary.json",
    }
    for label, path in paths.items():
        if not path.exists():
            errors.append(f"missing_dataset_file:{label}:{path}")

    manifest = read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    provenance = read_json(paths["provenance"]) if paths["provenance"].exists() else {}
    counts = manifest.get("counts", {})
    actual_counts = {
        "public_probes": line_count(paths["public_probes"]) if paths["public_probes"].exists() else None,
        "public_sessions": line_count(paths["public_sessions"]) if paths["public_sessions"].exists() else None,
        "private_keys": line_count(paths["private_key"]) if paths["private_key"].exists() else None,
    }
    expected_counts = {
        "public_probes": counts.get("probes", 90),
        "public_sessions": counts.get("sessions"),
        "private_keys": counts.get("keys", 90),
    }
    for key, expected in expected_counts.items():
        if expected is not None and actual_counts.get(key) != expected:
            errors.append(f"dataset_count_mismatch:{key}:expected={expected}:actual={actual_counts.get(key)}")
    if counts.get("probes") != 90:
        errors.append(f"subset_manifest_probe_count_not_90:{counts.get('probes')}")
    if provenance.get("status") != "pass":
        errors.append(f"domain_provenance_not_pass:{provenance.get('status')}")

    source_contract = manifest.get("source_contract", {})
    if source_contract.get("seed_prompts") != "allenai/WildChat":
        errors.append(f"source_contract_seed_prompts_unexpected:{source_contract.get('seed_prompts')}")
    if source_contract.get("family_domain_contract") != "nine_unique_representative_domains":
        errors.append(
            "source_contract_family_domain_contract_unexpected:"
            f"{source_contract.get('family_domain_contract')}"
        )

    return {
        "dataset_dir": str(dataset_dir),
        "paths": {key: str(path) for key, path in paths.items()},
        "counts": counts,
        "actual_counts": actual_counts,
        "source_contract": source_contract,
        "provenance_status": provenance.get("status"),
        "sha256": {
            "public_probes": sha256_file(paths["public_probes"]) if paths["public_probes"].exists() else None,
            "public_sessions": sha256_file(paths["public_sessions"]) if paths["public_sessions"].exists() else None,
            "private_key": sha256_file(paths["private_key"]) if paths["private_key"].exists() else None,
        },
    }


def check_gpu(allow_no_gpu: bool, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    gpu_rows = []
    if query.get("returncode") == 0:
        for line in query.get("stdout", "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            gpu_rows.append(
                {
                    "name": parts[0] if len(parts) > 0 else None,
                    "memory_total": parts[1] if len(parts) > 1 else None,
                    "memory_free": parts[2] if len(parts) > 2 else None,
                    "driver_version": parts[3] if len(parts) > 3 else None,
                }
            )
    elif allow_no_gpu:
        warnings.append("gpu_check_unavailable_but_allow_no_gpu")
    else:
        errors.append("gpu_check_failed_or_no_nvidia_smi")

    if not gpu_rows and not allow_no_gpu:
        errors.append("no_gpu_rows_detected")
    return {"query": query, "gpus": gpu_rows, "allow_no_gpu": allow_no_gpu}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        default=os.getenv("HABITBENCH_REUSE_SERVER") == "1",
        help="Allow no local GPU, useful when reusing an already-running endpoint.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.getenv("HABITBENCH_MIN_FREE_GB", "20")),
        help="Minimum free disk space required on dataset/results/cache anchors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: List[str] = []
    warnings: List[str] = []

    dataset = check_dataset(args.dataset_dir, errors)
    imports = {} if args.skip_imports else check_imports(errors)
    gpu = check_gpu(args.allow_no_gpu, errors, warnings)

    cache_path = Path(os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface")
    disk = [
        disk_row("dataset", args.dataset_dir),
        disk_row("results", args.results_dir),
        disk_row("hf_cache", cache_path),
    ]
    for row in disk:
        if row["free_gb"] < args.min_free_gb:
            errors.append(f"low_disk_free_gb:{row['label']}:expected>={args.min_free_gb}:actual={row['free_gb']}")

    commands = {
        "python": command_output([sys.executable, "--version"], timeout=10),
        "pip": command_output([sys.executable, "-m", "pip", "--version"], timeout=20),
    }
    if not args.allow_no_gpu:
        commands["nvidia_smi"] = command_output(["nvidia-smi", "--version"], timeout=20)

    manifest = {
        "created_at": now(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "thresholds": {
            "min_free_gb": args.min_free_gb,
            "allow_no_gpu": args.allow_no_gpu,
            "skip_imports": args.skip_imports,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "env": selected_env(),
        "dataset": dataset,
        "results_dir": str(args.results_dir),
        "imports": imports,
        "gpu": gpu,
        "disk": disk,
        "commands": commands,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "errors": len(errors), "warnings": len(warnings), "out": str(args.out)}, indent=2))
    if errors:
        raise SystemExit("Lumia run preflight failed")


if __name__ == "__main__":
    main()
