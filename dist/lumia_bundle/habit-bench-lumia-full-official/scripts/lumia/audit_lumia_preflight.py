#!/usr/bin/env python
"""Import and audit returned Lumia preflight-only artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_FILES = {
    "readiness": "lumia_readiness_remote.json",
    "run_preflight": "lumia_run_preflight_manifest_remote.json",
    "model_preflight": "model_preflight_manifest_remote.json",
}


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
        "# Lumia Preflight Import Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Created: {summary['created_at']}",
        f"- Returned root: `{summary['returned_root']}`",
        f"- Remote manifests: `{summary['remote_manifest_dir']}`",
        f"- Local manifests: `{summary['local_manifests_dir']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Files", ""])
    for key, value in summary["files"].items():
        lines.append(f"- {key}: `{value}`")
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
        default=Path("./runs/lumia_preflight_import_audit.json"),
    )
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--no-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_manifest_dir = args.returned_root / args.remote_manifests_subdir
    errors: List[str] = []
    warnings: List[str] = []
    copies = []
    payloads: Dict[str, Any] = {}

    for label, filename in EXPECTED_FILES.items():
        src = remote_manifest_dir / filename
        dst = args.local_manifests_dir / filename
        if not args.no_copy:
            copies.append(copy_file(src, dst))
        local_path = dst if not args.no_copy else src
        if not local_path.exists() or local_path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{label}:{local_path}")
            continue
        try:
            payloads[label] = read_json(local_path)
        except Exception as exc:
            errors.append(f"json_unreadable:{label}:{local_path}:{type(exc).__name__}:{exc}")

    readiness = payloads.get("readiness", {})
    run_preflight = payloads.get("run_preflight", {})
    model_preflight = payloads.get("model_preflight", {})

    if readiness and readiness.get("status") != "pass":
        errors.append(f"readiness_not_pass:{readiness.get('status')}")
    if run_preflight and run_preflight.get("status") != "pass":
        errors.append(f"run_preflight_not_pass:{run_preflight.get('status')}")
    if model_preflight and model_preflight.get("status") != "pass":
        errors.append(f"model_preflight_not_pass:{model_preflight.get('status')}")

    source_contract = run_preflight.get("dataset", {}).get("source_contract", {})
    if source_contract and source_contract.get("family_domain_contract") != "nine_unique_representative_domains":
        errors.append(
            "run_preflight_source_contract_unexpected:"
            f"{source_contract.get('family_domain_contract')}"
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "returned_root": str(args.returned_root),
        "remote_manifest_dir": str(remote_manifest_dir),
        "local_manifests_dir": str(args.local_manifests_dir),
        "copies": copies,
        "files": {
            label: str((args.local_manifests_dir if not args.no_copy else remote_manifest_dir) / filename)
            for label, filename in EXPECTED_FILES.items()
        },
        "checks": {
            "readiness_status": readiness.get("status"),
            "run_preflight_status": run_preflight.get("status"),
            "model_preflight_status": model_preflight.get("status"),
            "family_domain_contract": source_contract.get("family_domain_contract") if source_contract else None,
        },
        "errors": errors,
        "warnings": warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out = args.md_out or args.out.with_suffix(".md")
    write_markdown(md_out, summary)
    print(json.dumps({"status": summary["status"], "errors": len(errors), "out": str(args.out), "md": str(md_out)}, indent=2))
    if errors:
        raise SystemExit("Lumia preflight import/audit failed")


if __name__ == "__main__":
    main()
