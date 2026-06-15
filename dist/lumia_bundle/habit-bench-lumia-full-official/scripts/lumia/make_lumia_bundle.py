#!/usr/bin/env python
"""Create a minimal Lumia bundle for HABIT-Bench full official runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


INCLUDE_DIRS = [
    "docs",
    "eval",
    "schema",
    "scripts",
]

INCLUDE_FILES = [
    "README.md",
    "requirements-official.txt",
]

DATASET_INCLUDE = [
    "DATASET_CARD.md",
    "public/lifelines.jsonl",
    "public/probes.jsonl",
    "private/probe_key.jsonl",
    "reports/official_subset_manifest.json",
    "reports/domain_provenance_summary.json",
    "reports/domain_provenance_summary.md",
    "reports/source_domain_contract_audit.json",
    "reports/source_domain_contract_audit.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def copy_tree(src: Path, dst: Path, excludes: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    exclude_parts = set(excludes)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if any(part in exclude_parts for part in rel.parts):
            continue
        if rel.suffix in {".pyc", ".pyo"} or "__pycache__" in rel.parts:
            continue
        if rel.as_posix() == "lumia_handoff.md":
            continue
        rows.append(copy_file(path, dst / rel))
    return rows


def make_tarball(src_dir: Path, tar_path: Path) -> str:
    if tar_path.exists():
        tar_path.unlink()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return sha256_file(tar_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./runs/habit_bench_balanced_v0_3_official_subset_90"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./dist/lumia_bundle"))
    parser.add_argument("--name", default="habit-bench-lumia-full-official")
    parser.add_argument("--no-tar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = args.out_dir / args.name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_project = bundle_root
    copied: List[Dict[str, Any]] = []

    for rel in INCLUDE_FILES:
        copied.append(copy_file(args.root / rel, bundle_project / rel))
    for rel in INCLUDE_DIRS:
        copied.extend(copy_tree(args.root / rel, bundle_project / rel, excludes=[]))

    dataset_dst = bundle_project / "runs" / args.dataset_dir.name
    for rel in DATASET_INCLUDE:
        copied.append(copy_file(args.dataset_dir / rel, dataset_dst / rel))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_name": args.name,
        "source_root": str(args.root),
        "source_dataset": str(args.dataset_dir),
        "bundle_root": str(bundle_root),
        "dataset_relpath": str(Path("runs") / args.dataset_dir.name),
        "file_count": len(copied),
        "total_bytes": sum(row["bytes"] for row in copied),
        "files": copied,
        "usage": [
            "tar -xzf habit-bench-lumia-full-official.tar.gz",
            "cd habit-bench-lumia-full-official",
            "python -m venv .venv && source .venv/bin/activate",
            "python -m pip install -U pip",
            "python -m pip install -r ./requirements-official.txt",
            "python ./scripts/lumia/check_lumia_readiness.py",
            "source ./scripts/lumia/lumia_env_example.sh",
            "bash ./scripts/lumia/download_open_models.sh",
            "bash ./scripts/lumia/start_vllm_openai_server.sh",
            "bash ./scripts/run_full_official_subset_suite.sh",
        ],
    }
    manifest_path = bundle_project / "runs" / "lumia_bundle_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    tar_info = None
    if not args.no_tar:
        tar_path = args.out_dir / f"{args.name}.tar.gz"
        tar_info = {
            "path": str(tar_path),
            "sha256": make_tarball(bundle_root, tar_path),
            "bytes": tar_path.stat().st_size,
        }
        tar_manifest = {
            **manifest,
            "tarball": tar_info,
            "note": "This sidecar manifest is outside the tarball and includes the tarball digest. The in-bundle manifest intentionally cannot include its containing tarball digest.",
        }
        sidecar_path = tar_path.with_suffix(tar_path.suffix + ".manifest.json")
        sidecar_path.write_text(
            json.dumps(tar_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with tar_path.with_suffix(tar_path.suffix + ".sha256").open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"{tar_info['sha256']}  {tar_path.name}\n")

    print(
        json.dumps(
            {
                "bundle_root": str(bundle_root),
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "tarball": tar_info,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
