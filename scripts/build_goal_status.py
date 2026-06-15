#!/usr/bin/env python
"""Build an objective-level status report for the HABIT-Bench Lumia goal."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(".")
RESEARCH_LOG = Path("research/research-runs/2026-06-11-agent-memory-long-term-user-preference-benchmark/05_run_log.csv")


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def status_item(name: str, status: str, evidence: List[str], missing: List[str] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "missing": missing or [],
    }


def main() -> None:
    v03_manifest_path = ROOT / "runs/habit_bench_balanced_v0_3/reports/balanced_v03_manifest.json"
    v03_prov_path = ROOT / "runs/habit_bench_balanced_v0_3/reports/domain_provenance_summary.json"
    v03_source_domain_audit_path = ROOT / "runs/habit_bench_balanced_v0_3/reports/source_domain_contract_audit.json"
    subset_manifest_path = ROOT / "runs/habit_bench_balanced_v0_3_official_subset_90/reports/official_subset_manifest.json"
    subset_prov_path = ROOT / "runs/habit_bench_balanced_v0_3_official_subset_90/reports/domain_provenance_summary.json"
    subset_source_domain_audit_path = ROOT / "runs/habit_bench_balanced_v0_3_official_subset_90/reports/source_domain_contract_audit.json"
    full_audit_path = ROOT / "runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results/audit/full_official_audit.json"
    bundle_manifest_path = ROOT / "dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz.manifest.json"
    bundle_handoff_path = ROOT / "dist/lumia_bundle/habit-bench-lumia-full-official.handoff.md"
    preflight_path = ROOT / "runs/lumia_manifests/model_preflight_manifest_local.json"
    remote_preflight_audit_path = ROOT / "runs/lumia_preflight_import_audit.json"
    remote_preflight_audit_md_path = ROOT / "runs/lumia_preflight_import_audit.md"
    model_download_audit_path = ROOT / "runs/lumia_model_download_audit.json"
    model_download_audit_md_path = ROOT / "runs/lumia_model_download_audit.md"

    v03_manifest = read_json(v03_manifest_path) or {}
    v03_prov = read_json(v03_prov_path) or {}
    v03_source_domain_audit = read_json(v03_source_domain_audit_path) or {}
    subset_manifest = read_json(subset_manifest_path) or {}
    subset_prov = read_json(subset_prov_path) or {}
    subset_source_domain_audit = read_json(subset_source_domain_audit_path) or {}
    full_audit = read_json(full_audit_path) or {}
    bundle_manifest = read_json(bundle_manifest_path) or {}
    preflight = read_json(preflight_path) or {}
    remote_preflight_audit = read_json(remote_preflight_audit_path) or {}
    model_download_audit = read_json(model_download_audit_path) or {}
    source_contract = subset_manifest.get("source_contract", {})
    checked_preflight = full_audit.get("run_manifests", {}).get("checked_preflight", [])
    successful_full_run_preflights = [
        str(item.get("path"))
        for item in checked_preflight
        if item.get("status") == "pass" and item.get("path")
    ]
    remote_preflight_complete = remote_preflight_audit.get("status") == "pass" or bool(successful_full_run_preflights)

    items = [
        status_item(
            "9-family unified table",
            "complete" if (ROOT / "docs/9_family_unified_table.md").exists() else "missing",
            [str(ROOT / "docs/9_family_unified_table.md")],
        ),
        status_item(
            "expanded balanced v0.3 candidate",
            "complete"
            if v03_manifest.get("validation", {}).get("status") == "pass"
            and v03_manifest.get("counts", {}).get("probes") == 2010
            else "incomplete",
            [str(v03_manifest_path)],
            [] if v03_manifest.get("counts", {}).get("probes") == 2010 else ["expected 2010 probes"],
        ),
        status_item(
            "9-family domain provenance",
            "complete" if v03_prov.get("status") == "pass" and v03_prov.get("counts", {}).get("families") == 9 else "incomplete",
            [str(v03_prov_path), str(subset_prov_path)],
        ),
        status_item(
            "official 90-probe subset",
            "complete" if subset_manifest.get("counts", {}).get("probes") == 90 and subset_prov.get("status") == "pass" else "incomplete",
            [str(subset_manifest_path), str(subset_prov_path)],
        ),
        status_item(
            "source/domain contract",
            "complete"
            if source_contract.get("seed_prompts") == "allenai/WildChat"
            and source_contract.get("family_domain_contract") == "nine_unique_representative_domains"
            and v03_prov.get("status") == "pass"
            and subset_prov.get("status") == "pass"
            and v03_source_domain_audit.get("status") == "pass"
            and subset_source_domain_audit.get("status") == "pass"
            else "incomplete",
            [
                str(ROOT / "docs/9_family_unified_table.md"),
                str(ROOT / "docs/9_family_taxonomy.md"),
                str(v03_manifest_path),
                str(subset_manifest_path),
                str(v03_prov_path),
                str(subset_prov_path),
                str(v03_source_domain_audit_path),
                str(subset_source_domain_audit_path),
            ],
            []
            if source_contract.get("seed_prompts") == "allenai/WildChat"
            and source_contract.get("family_domain_contract") == "nine_unique_representative_domains"
            and v03_source_domain_audit.get("status") == "pass"
            and subset_source_domain_audit.get("status") == "pass"
            else ["expected WildChat seed source and nine_unique_representative_domains contract"],
        ),
        status_item(
            "open model preflight",
            "complete" if preflight.get("status") == "pass" else "incomplete",
            [str(preflight_path)],
            [] if preflight.get("status") == "pass" else ["run preflight_open_models.py on Lumia or locally"],
        ),
        status_item(
            "Lumia bundle and remote launcher",
            "complete" if bundle_manifest.get("tarball", {}).get("sha256") else "incomplete",
            [
                str(bundle_manifest_path),
                str(bundle_handoff_path),
                str(ROOT / "dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz"),
                str(ROOT / "scripts/lumia/launch_lumia_remote.py"),
            ],
        ),
        status_item(
            "remote preflight import audit",
            "complete" if remote_preflight_complete else "external_pending",
            [
                str(remote_preflight_audit_path),
                str(remote_preflight_audit_md_path),
                str(ROOT / "scripts/lumia/audit_lumia_preflight.py"),
                str(full_audit_path),
            ]
            + successful_full_run_preflights,
            []
            if remote_preflight_complete
            else ["run launch_lumia_remote.py --preflight-only --execute on Lumia and require local preflight audit status=pass"],
        ),
        status_item(
            "remote open-model download audit",
            "complete" if model_download_audit.get("status") == "pass" else "external_pending",
            [
                str(model_download_audit_path),
                str(model_download_audit_md_path),
                str(ROOT / "scripts/lumia/audit_model_download.py"),
            ],
            []
            if model_download_audit.get("status") == "pass"
            else ["run launch_lumia_remote.py --download-models-only --execute on Lumia and require local model download audit status=pass"],
        ),
        status_item(
            "full official subset run on Lumia",
            "complete" if full_audit.get("status") == "pass" else "blocked_external_pending",
            [str(full_audit_path)],
            full_audit.get("errors", ["missing full official audit pass"]),
        ),
    ]

    overall = "complete" if all(item["status"] == "complete" for item in items) else "incomplete"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "items": items,
        "current_bundle": bundle_manifest.get("tarball"),
        "next_decisive_action": (
            "No required action remains; all objective-level checks are complete."
            if overall == "complete"
            else (
                "Run launch_lumia_remote.py with --execute against a real Lumia host, or run "
                "run_lumia_full_official_e2e.sh on Lumia, then import_lumia_results.py and "
                "require full_official_audit.status == pass."
            )
        ),
        "files": {
            "research_log": file_info(RESEARCH_LOG),
            "full_audit": file_info(full_audit_path),
            "bundle_manifest": file_info(bundle_manifest_path),
            "remote_preflight_audit": file_info(remote_preflight_audit_path),
            "remote_preflight_audit_md": file_info(remote_preflight_audit_md_path),
            "model_download_audit": file_info(model_download_audit_path),
            "model_download_audit_md": file_info(model_download_audit_md_path),
        },
    }

    out_json = ROOT / "runs/goal_status.json"
    out_md = ROOT / "runs/goal_status.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# HABIT-Bench Goal Status",
        "",
        f"- Overall status: `{overall}`",
        f"- Created: {summary['created_at']}",
        "",
        "## Items",
        "",
        "| item | status | evidence | missing / blocker |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        evidence = "<br>".join(f"`{path}`" for path in item["evidence"])
        missing = "<br>".join(item["missing"]) if item["missing"] else "-"
        lines.append(f"| {item['name']} | `{item['status']}` | {evidence} | {missing} |")
    lines.extend(
        [
            "",
            "## Next Decisive Action",
            "",
            summary["next_decisive_action"],
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": overall, "json": str(out_json), "md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
