#!/usr/bin/env python
"""Import returned Lumia full-official results and run the local audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def copy_tree(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        return {"source": str(src), "dest": str(dst), "copied": False, "reason": "missing_source"}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = [path for path in dst.rglob("*") if path.is_file()]
    return {
        "source": str(src),
        "dest": str(dst),
        "copied": True,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def copy_tree_merge(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        return {"source": str(src), "dest": str(dst), "copied": False, "reason": "missing_source"}
    copied = []
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return {
        "source": str(src),
        "dest": str(dst),
        "copied": True,
        "mode": "merge",
        "files": len(copied),
        "bytes": sum(path.stat().st_size for path in copied),
    }


def run_command(command: List[str]) -> Dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned-root", type=Path, required=True, help="Local directory containing returned Lumia work/ tree or result dirs.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./runs/habit_bench_balanced_v0_3_official_subset_90"),
    )
    parser.add_argument("--results-subdir", default="./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results")
    parser.add_argument("--manifests-subdir", default="./runs/lumia_manifests")
    parser.add_argument("--local-manifests-dir", type=Path, default=Path("./runs/lumia_manifests"))
    parser.add_argument("--out", type=Path, default=Path("./runs/lumia_import_summary.json"))
    parser.add_argument("--no-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returned_root = args.returned_root
    local_results = args.dataset_dir / "full_official_results"
    local_manifests = args.local_manifests_dir

    copies = []
    if not args.no_copy:
        copies.append(copy_tree(returned_root / args.results_subdir, local_results))
        copies.append(copy_tree_merge(returned_root / args.manifests_subdir, local_manifests))

    audit_cmd = [
        sys.executable,
        "./eval/audit_full_official_results.py",
        "--dataset-dir",
        str(args.dataset_dir),
        "--results-dir",
        str(local_results),
        "--model-manifest",
        str(local_manifests / "model_download_manifest.json"),
    ]
    audit = run_command(audit_cmd)
    status = "pass" if audit["returncode"] == 0 else "fail"
    audit_json = local_results / "audit" / "full_official_audit.json"
    audit_payload = json.loads(audit_json.read_text(encoding="utf-8")) if audit_json.exists() else None
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "returned_root": str(returned_root),
        "dataset_dir": str(args.dataset_dir),
        "local_results": str(local_results),
        "local_manifests": str(local_manifests),
        "copies": copies,
        "audit_command": audit,
        "audit": audit_payload,
        "run_log_hint": {
            "metric": "lumia_full_official_import_audit",
            "value": f"audit_status={status}",
            "notes": "Only claim completion if status=pass and audit errors are empty.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "out": str(args.out)}, indent=2))
    if status != "pass":
        raise SystemExit("Imported Lumia results did not pass audit")


if __name__ == "__main__":
    main()
