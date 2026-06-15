#!/usr/bin/env python
"""Verify a HABIT-Bench Lumia bundle with lightweight release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BUNDLE = Path("./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz")
DEFAULT_OUT = Path("./runs/lumia_bundle_verify")
REQUIRED_BUNDLE_PATHS = [
    "README.md",
    "requirements-official.txt",
    "docs/9_family_taxonomy.md",
    "docs/9_family_unified_table.md",
    "schema/probe.schema.json",
    "schema/session.schema.json",
    "eval/run_external_baseline.py",
    "eval/audit_full_official_results.py",
    "eval/official_adapters/official_mem0_full_llm_adapter.py",
    "eval/official_adapters/official_graphiti_full_llm_adapter.py",
    "scripts/run_full_official_subset_suite.sh",
    "scripts/lumia/check_lumia_readiness.py",
    "scripts/lumia/run_lumia_full_official_e2e.sh",
    "runs/habit_bench_balanced_v0_3_official_subset_90/public/probes.jsonl",
    "runs/habit_bench_balanced_v0_3_official_subset_90/public/lifelines.jsonl",
    "runs/habit_bench_balanced_v0_3_official_subset_90/private/probe_key.jsonl",
    "runs/habit_bench_balanced_v0_3_official_subset_90/reports/official_subset_manifest.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run(command: List[str], cwd: Path) -> Dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def sidecar_status(bundle: Path) -> Dict[str, Any]:
    sha_path = bundle.with_suffix(bundle.suffix + ".sha256")
    manifest_path = bundle.with_suffix(bundle.suffix + ".manifest.json")
    actual_sha = sha256_file(bundle)
    status = {
        "bundle": str(bundle),
        "bytes": bundle.stat().st_size,
        "sha256": actual_sha,
        "sha256_sidecar": str(sha_path),
        "manifest_sidecar": str(manifest_path),
        "checks": [],
        "errors": [],
    }
    if not sha_path.exists():
        status["errors"].append(f"missing_sha256_sidecar:{sha_path}")
    else:
        text = sha_path.read_text(encoding="utf-8").strip()
        expected_sha = text.split()[0] if text else ""
        status["expected_sha256"] = expected_sha
        if expected_sha != actual_sha:
            status["errors"].append(f"sha256_mismatch:expected={expected_sha}:actual={actual_sha}")
        else:
            status["checks"].append("sha256_sidecar_matches")
    if not manifest_path.exists():
        status["errors"].append(f"missing_manifest_sidecar:{manifest_path}")
    else:
        manifest = read_json(manifest_path)
        tarball = manifest.get("tarball", {})
        status["manifest_file_count"] = manifest.get("file_count")
        status["manifest_tarball"] = tarball
        if tarball.get("sha256") != actual_sha:
            status["errors"].append(
                f"manifest_sha256_mismatch:expected={tarball.get('sha256')}:actual={actual_sha}"
            )
        else:
            status["checks"].append("manifest_sha256_matches")
    return status


def unpack_bundle(bundle: Path, temp_root: Path) -> Path:
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(temp_root)
    candidates = [path for path in temp_root.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one top-level bundle directory, found {len(candidates)}")
    return candidates[0]


def check_required_paths(root: Path) -> List[str]:
    return [rel for rel in REQUIRED_BUNDLE_PATHS if not (root / rel).exists()]


def write_markdown(summary: Dict[str, Any], out: Path) -> None:
    lines = [
        "# Lumia Bundle Verify",
        "",
        f"- Status: `{summary['status']}`",
        f"- Created: {summary['created_at']}",
        f"- Bundle: `{summary['sidecar']['bundle']}`",
        f"- SHA256: `{summary['sidecar']['sha256']}`",
        f"- Bundle files: {summary['sidecar'].get('manifest_file_count', '-')}",
        "",
        "## Commands",
        "",
        "| command | returncode |",
        "| --- | ---: |",
    ]
    for row in summary["commands"]:
        command = " ".join(row["command"])
        lines.append(f"| `{command}` | {row['returncode']} |")
    lines.extend(["", "## Errors", ""])
    if summary["errors"]:
        lines.extend(f"- {error}" for error in summary["errors"])
    else:
        lines.append("- none")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: List[str] = []
    commands: List[Dict[str, Any]] = []
    sidecar = sidecar_status(args.bundle)
    errors.extend(sidecar["errors"])
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="habitbench_lumia_bundle_")
    temp_root = Path(temp_dir_obj.name)
    bundle_root = None
    try:
        bundle_root = unpack_bundle(args.bundle, temp_root)
        missing = check_required_paths(bundle_root)
        errors.extend(f"missing_bundle_file:{rel}" for rel in missing)
        readiness_command = [
            "scripts/lumia/check_lumia_readiness.py",
            "--root",
            ".",
            "--dataset-dir",
            "runs/habit_bench_balanced_v0_3_official_subset_90",
            "--out",
            "runs/lumia_manifests/lumia_readiness_unpacked.json",
        ]
        row = run([sys.executable, *readiness_command], bundle_root)
        commands.append(row)
        if row["returncode"] != 0:
            errors.append(f"command_failed:{' '.join(readiness_command)}")
    finally:
        if args.keep_temp:
            kept_path = args.out_dir / "kept_temp_path.txt"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            kept_path.write_text(str(temp_root), encoding="utf-8")
            temp_dir_obj.cleanup = lambda: None  # type: ignore[method-assign]
        else:
            temp_dir_obj.cleanup()

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "sidecar": sidecar,
        "bundle_root_name": bundle_root.name if bundle_root else None,
        "commands": commands,
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "lumia_bundle_verify.json"
    md_path = args.out_dir / "lumia_bundle_verify.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"status": summary["status"], "out": str(json_path), "errors": len(errors)}, indent=2))
    if errors:
        raise SystemExit("Lumia bundle verification failed")


if __name__ == "__main__":
    main()
