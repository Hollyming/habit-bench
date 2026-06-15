#!/usr/bin/env python
"""Audit returned open-model download manifests from Lumia."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def copy_file(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        return {"source": str(src), "dest": str(dst), "copied": False, "reason": "missing_source"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "dest": str(dst), "copied": True, "bytes": dst.stat().st_size}


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Lumia Model Download Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Created: {summary['created_at']}",
        f"- Manifest: `{summary['manifest']}`",
        f"- Preflight manifest: `{summary['preflight_manifest']}`",
        "",
        "## Models",
        "",
    ]
    for row in summary.get("models", []):
        lines.append(f"- `{row.get('repo_id')}`: `{row.get('status')}` cache=`{row.get('cache_path')}`")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in summary["errors"]) if summary["errors"] else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"]) if summary["warnings"] else lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned-root", type=Path, default=Path("runs/lumia_returned/habit-bench-lumia-full-official"))
    parser.add_argument("--remote-manifests-subdir", default="./runs/lumia_manifests")
    parser.add_argument("--local-manifests-dir", type=Path, default=Path("./runs/lumia_manifests"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./runs/lumia_model_download_audit.json"),
    )
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--no-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_dir = args.returned_root / args.remote_manifests_subdir
    manifest_src = remote_dir / "model_download_manifest.json"
    preflight_src = remote_dir / "model_preflight_manifest.json"
    manifest_dst = args.local_manifests_dir / "model_download_manifest.json"
    preflight_dst = args.local_manifests_dir / "model_preflight_manifest_remote_download.json"

    copies = []
    if not args.no_copy:
        copies.append(copy_file(manifest_src, manifest_dst))
        copies.append(copy_file(preflight_src, preflight_dst))
    manifest_path = manifest_dst if not args.no_copy else manifest_src
    preflight_path = preflight_dst if not args.no_copy else preflight_src

    errors: List[str] = []
    warnings: List[str] = []
    manifest: Dict[str, Any] = {}
    preflight: Dict[str, Any] = {}

    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        errors.append(f"missing_or_empty_model_download_manifest:{manifest_path}")
    else:
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            errors.append(f"model_download_manifest_unreadable:{type(exc).__name__}:{exc}")

    if preflight_path.exists() and preflight_path.stat().st_size > 0:
        try:
            preflight = read_json(preflight_path)
        except Exception as exc:
            warnings.append(f"model_preflight_manifest_unreadable:{type(exc).__name__}:{exc}")
    else:
        warnings.append(f"missing_model_preflight_manifest:{preflight_path}")

    if manifest:
        if manifest.get("dry_run"):
            errors.append("model_download_manifest_is_dry_run")
        if manifest.get("status") != "pass":
            errors.append(f"model_download_status_not_pass:{manifest.get('status')}")
        models = manifest.get("models", [])
        if len(models) < 2:
            errors.append(f"model_download_expected_at_least_two_models:actual={len(models)}")
        for row in models:
            if row.get("status") != "pass":
                errors.append(f"model_download_row_not_pass:{row.get('repo_id')}:{row.get('status')}")
            if not row.get("cache_path"):
                errors.append(f"model_download_missing_cache_path:{row.get('repo_id')}")

    if preflight and preflight.get("status") != "pass":
        warnings.append(f"model_preflight_status_not_pass:{preflight.get('status')}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "returned_root": str(args.returned_root),
        "remote_manifest_dir": str(remote_dir),
        "local_manifests_dir": str(args.local_manifests_dir),
        "manifest": str(manifest_path),
        "preflight_manifest": str(preflight_path),
        "copies": copies,
        "models": manifest.get("models", []),
        "errors": errors,
        "warnings": warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out = args.md_out or args.out.with_suffix(".md")
    write_markdown(md_out, summary)
    print(json.dumps({"status": summary["status"], "errors": len(errors), "out": str(args.out), "md": str(md_out)}, indent=2))
    if errors:
        raise SystemExit("Lumia model download audit failed")


if __name__ == "__main__":
    main()
