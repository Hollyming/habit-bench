#!/usr/bin/env python
"""Write reproducibility manifests for Lumia HABIT-Bench runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_ENV_KEYS = [
    "HABITBENCH_LLM_MODEL",
    "HABITBENCH_SERVED_MODEL",
    "HABITBENCH_EMBED_MODEL",
    "HABITBENCH_STRUCTURED_OUTPUT_MODE",
    "HABITBENCH_DATASET",
    "HABITBENCH_OFFICIAL_OUT",
    "OPENAI_BASE_URL",
    "HABITBENCH_VLLM_HOST",
    "HABITBENCH_VLLM_PORT",
    "HABITBENCH_TENSOR_PARALLEL",
    "HABITBENCH_GPU_MEMORY_UTIL",
    "HABITBENCH_BUNDLE_SHA256",
    "HABITBENCH_BUNDLE_FILE_COUNT",
    "HABITBENCH_BUNDLE_BYTES",
    "HABITBENCH_BUNDLE_MANIFEST",
    "HABITBENCH_BUNDLE_VERIFY_REPORT",
    "HABITBENCH_MODEL_PREFLIGHT_MANIFEST",
    "HABITBENCH_MODEL_DOWNLOAD_MANIFEST",
    "HABITBENCH_BUNDLE_VERIFY_STATUS",
    "HABITBENCH_SOURCE_DOMAIN_AUDIT_STATUS",
    "HABITBENCH_MODEL_PREFLIGHT_STATUS",
    "CUDA_VISIBLE_DEVICES",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def maybe_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists() and path.is_file():
        row["sha256"] = sha256_file(path)
    return row


def command_output(command: List[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"
    text = (completed.stdout or completed.stderr or "").strip()
    return text[:4000]


def dataset_fingerprint(dataset_dir: Path | None) -> Dict[str, Any] | None:
    if dataset_dir is None:
        return None
    public_probes = dataset_dir / "public" / "probes.jsonl"
    public_sessions = dataset_dir / "public" / "lifelines.jsonl"
    private_key = dataset_dir / "private" / "probe_key.jsonl"
    subset_manifest = dataset_dir / "reports" / "official_subset_manifest.json"
    v03_manifest = dataset_dir / "reports" / "balanced_v03_manifest.json"
    provenance = dataset_dir / "reports" / "domain_provenance_summary.json"
    source_domain_audit = dataset_dir / "reports" / "source_domain_contract_audit.json"
    source_domain_payload = maybe_json(source_domain_audit)
    return {
        "dataset_dir": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "counts": {
            "public_probes": line_count(public_probes),
            "public_sessions": line_count(public_sessions),
            "private_keys": line_count(private_key),
        },
        "sha256": {
            "public_probes": sha256_file(public_probes) if public_probes.exists() else None,
            "public_sessions": sha256_file(public_sessions) if public_sessions.exists() else None,
            "private_key": sha256_file(private_key) if private_key.exists() else None,
        },
        "manifest": maybe_json(subset_manifest) or maybe_json(v03_manifest),
        "domain_provenance": maybe_json(provenance),
        "source_domain_contract_audit": {
            "artifact": artifact(source_domain_audit),
            "status": source_domain_payload.get("status") if source_domain_payload else None,
            "expected_source": source_domain_payload.get("expected_source") if source_domain_payload else None,
            "families": source_domain_payload.get("counts", {}).get("families") if source_domain_payload else None,
        },
    }


def run_artifacts() -> Dict[str, Any]:
    bundle_manifest_path = Path(
        os.getenv(
            "HABITBENCH_BUNDLE_MANIFEST",
            "./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz.manifest.json",
        )
    )
    bundle_verify_path = Path(
        os.getenv(
            "HABITBENCH_BUNDLE_VERIFY_REPORT",
            "./runs/lumia_bundle_verify/lumia_bundle_verify.json",
        )
    )
    model_preflight_path = Path(
        os.getenv(
            "HABITBENCH_MODEL_PREFLIGHT_MANIFEST",
            "./runs/lumia_manifests/model_preflight_manifest_local.json",
        )
    )
    model_download_path = Path(
        os.getenv(
            "HABITBENCH_MODEL_DOWNLOAD_MANIFEST",
            "./runs/lumia_manifests/model_download_manifest.json",
        )
    )
    bundle_manifest = maybe_json(bundle_manifest_path)
    bundle_verify = maybe_json(bundle_verify_path)
    model_preflight = maybe_json(model_preflight_path)
    model_download = maybe_json(model_download_path)
    return {
        "bundle_manifest": {
            "artifact": artifact(bundle_manifest_path),
            "tarball": bundle_manifest.get("tarball") if bundle_manifest else None,
            "file_count": bundle_manifest.get("file_count") if bundle_manifest else None,
        },
        "bundle_verify": {
            "artifact": artifact(bundle_verify_path),
            "status": bundle_verify.get("status") if bundle_verify else None,
            "command_count": len(bundle_verify.get("commands", [])) if bundle_verify else 0,
        },
        "model_preflight": {
            "artifact": artifact(model_preflight_path),
            "status": model_preflight.get("status") if model_preflight else None,
            "models": [row.get("repo_id") for row in model_preflight.get("models", [])] if model_preflight else [],
        },
        "model_download": {
            "artifact": artifact(model_download_path),
            "status": model_download.get("status") if model_download else None,
            "dry_run": model_download.get("dry_run") if model_download else None,
            "model_count": len(model_download.get("models", [])) if model_download else 0,
        },
    }


def selected_env(keys: Iterable[str]) -> Dict[str, str | None]:
    redacted = {}
    for key in keys:
        value = os.getenv(key)
        if "KEY" in key or "TOKEN" in key:
            value = "<redacted>" if value else None
        redacted[key] = value
    return redacted


def parse_key_value(values: List[str]) -> Dict[str, str]:
    parsed = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--command", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--extra", action="append", default=[], help="Additional KEY=VALUE entries.")
    parser.add_argument("--env-key", action="append", default=[], help="Additional env var names to capture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_keys = list(dict.fromkeys(DEFAULT_ENV_KEYS + args.env_key))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "command": args.command,
        "note": args.note,
        "extra": parse_key_value(args.extra),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "versions": {
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
            "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        },
        "env": selected_env(env_keys),
        "dataset": dataset_fingerprint(args.dataset_dir),
        "results_dir": str(args.results_dir) if args.results_dir else None,
        "artifacts": run_artifacts(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out), "stage": args.stage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
