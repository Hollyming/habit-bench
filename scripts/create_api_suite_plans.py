#!/usr/bin/env python
"""Create or validate persistent per-model plans for an external API suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def model_slug(model: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", model.lower().replace(".", "-"))
    return slug.strip("-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(plan: Path, model: str, expected_tasks: int) -> dict[str, Any]:
    manifest_path = plan.with_suffix(".manifest.json")
    if not plan.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Plan/manifest pair is incomplete: {plan}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    llm = ((manifest.get("models") or {}).get("llm") or {})
    if llm.get("served_model_name") != model:
        raise ValueError(
            f"Plan model mismatch at {plan}: {llm.get('served_model_name')} != {model}"
        )
    if llm.get("provider") != "external-openai-compatible":
        raise ValueError(f"Plan is not an external API plan: {plan}")
    if manifest.get("task_count") != expected_tasks:
        raise ValueError(
            f"Plan task count mismatch at {plan}: "
            f"{manifest.get('task_count')} != {expected_tasks}"
        )
    if manifest.get("plan_sha256") != sha256_file(plan):
        raise ValueError(f"Plan hash mismatch: {plan}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--gpu-allocations", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-model-path", type=Path, required=True)
    parser.add_argument("--lightmem-model-path", type=Path, required=True)
    parser.add_argument("--secom-compressor-path", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--api-origin", required=True)
    parser.add_argument("--rpm", type=int, required=True)
    parser.add_argument("--tpm", type=int, required=True)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = split_csv(args.models)
    allocations = [int(value) for value in split_csv(args.gpu_allocations)]
    methods = split_csv(args.methods)
    datasets = split_csv(args.datasets)
    if not models or len(models) != len(allocations):
        raise ValueError("models and gpu allocations must have equal nonzero lengths")
    if sum(allocations) != 8 or min(allocations) < 1:
        raise ValueError("API suite GPU allocations must be positive and sum to 8")
    if args.shards < 1:
        raise ValueError("shards must be positive")
    expected_tasks = len(methods) * len(datasets) * args.shards
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for model, allocation in zip(models, allocations, strict=True):
        slug = model_slug(model)
        model_root = (args.output_root / slug).resolve()
        plan = model_root / "shard_plan.tsv"
        manifest_path = plan.with_suffix(".manifest.json")
        if args.force or not plan.exists():
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/create_shard_plan.py"),
                "--methods",
                args.methods,
                "--datasets",
                args.datasets,
                "--shards",
                str(args.shards),
                "--embedding-model-path",
                str(args.embedding_model_path),
                "--llm-model-path",
                str(args.tokenizer_model_path),
                "--served-model-name",
                model,
                "--llm-provider",
                "external-openai-compatible",
                "--lightmem-model-path",
                str(args.lightmem_model_path),
                "--secom-compressor-path",
                str(args.secom_compressor_path),
                "--output-root",
                str(model_root),
                "--plan",
                str(plan),
                "--metadata",
                "api_model=" + model,
                "--metadata",
                "model_worker_gpus=" + str(allocation),
                "--metadata",
                "api_origin=" + args.api_origin,
                "--metadata",
                "api_rpm=" + str(args.rpm),
                "--metadata",
                "api_tpm=" + str(args.tpm),
                "--metadata",
                "api_key=redacted-external-file",
            ]
            for metadata in args.metadata:
                command.extend(["--metadata", metadata])
            if args.force and plan.exists():
                command.append("--force")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        manifest = validate_plan(plan, model, expected_tasks)
        records.append(
            {
                "model": model,
                "slug": slug,
                "worker_gpus": allocation,
                "plan": str(plan),
                "plan_sha256": manifest["plan_sha256"],
                "task_count": manifest["task_count"],
            }
        )

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    suite_manifest = {
        "contract_version": "habitbench.h_api_suite_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_name": args.job_name,
        "job_type": args.job_type,
        "models": records,
        "methods": methods,
        "datasets": datasets,
        "shards_per_method_domain": args.shards,
        "resources": {"gpus_per_replica": 8, "replicas": 1, "total_gpus": 8},
        "api": {
            "origin": args.api_origin,
            "rpm": args.rpm,
            "tpm": args.tpm,
            "credential": "external-mode-600-file",
        },
        "shared_context_tokenizer": str(args.tokenizer_model_path.resolve()),
        "git": {"revision": revision, "dirty": dirty},
    }
    output = args.output_root / "api_suite_manifest.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "suite_manifest": str(output),
                "models": len(records),
                "tasks": sum(record["task_count"] for record in records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
