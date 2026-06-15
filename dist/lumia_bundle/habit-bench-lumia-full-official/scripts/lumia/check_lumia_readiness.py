#!/usr/bin/env python
"""Check that a HABIT-Bench Lumia workspace is ready to run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FILES = [
    "requirements-official.txt",
    "README.md",
    "schema/probe.schema.json",
    "schema/session.schema.json",
    "eval/run_external_baseline.py",
    "eval/collect_official_results.py",
    "eval/audit_full_official_results.py",
    "scripts/audit_source_domain_contract.py",
    "eval/official_adapters/official_mem0_full_llm_adapter.py",
    "eval/official_adapters/official_graphiti_full_llm_adapter.py",
    "scripts/run_full_official_subset_suite.sh",
    "scripts/run_full_official_subset_mem0.sh",
    "scripts/run_full_official_subset_graphiti.sh",
    "scripts/lumia/download_open_models.sh",
    "scripts/lumia/download_open_models.py",
    "scripts/lumia/start_vllm_openai_server.sh",
    "scripts/lumia/check_openai_endpoint.py",
    "scripts/lumia/write_run_manifest.py",
    "scripts/lumia/check_lumia_readiness.py",
    "scripts/lumia/make_lumia_bundle.py",
    "scripts/lumia/verify_lumia_bundle.py",
    "scripts/lumia/run_lumia_full_official_e2e.sh",
    "scripts/lumia/preflight_lumia_run.py",
    "scripts/lumia/preflight_open_models.py",
    "scripts/lumia/audit_lumia_preflight.py",
    "scripts/lumia/audit_model_download.py",
    "scripts/lumia/import_lumia_results.py",
    "scripts/lumia/launch_lumia_remote.py",
    "scripts/lumia/run_lumia_guarded_full_cycle.py",
    "scripts/lumia/write_lumia_handoff.py",
]

COMPILE_FILES = [
    "eval/run_external_baseline.py",
    "eval/collect_official_results.py",
    "eval/audit_full_official_results.py",
    "scripts/audit_source_domain_contract.py",
    "eval/official_adapters/official_mem0_full_llm_adapter.py",
    "eval/official_adapters/official_graphiti_full_llm_adapter.py",
    "scripts/lumia/check_openai_endpoint.py",
    "scripts/lumia/write_run_manifest.py",
    "scripts/lumia/check_lumia_readiness.py",
    "scripts/lumia/make_lumia_bundle.py",
    "scripts/lumia/verify_lumia_bundle.py",
    "scripts/lumia/preflight_lumia_run.py",
    "scripts/lumia/preflight_open_models.py",
    "scripts/lumia/audit_lumia_preflight.py",
    "scripts/lumia/audit_model_download.py",
    "scripts/lumia/import_lumia_results.py",
    "scripts/lumia/launch_lumia_remote.py",
    "scripts/lumia/run_lumia_guarded_full_cycle.py",
    "scripts/lumia/write_lumia_handoff.py",
]

OPTIONAL_IMPORTS = [
    "mem0",
    "graphiti_core",
    "kuzu",
    "sentence_transformers",
    "vllm",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def check_dataset(dataset_dir: Path, errors: List[str]) -> Dict[str, Any]:
    required = {
        "public_probes": dataset_dir / "public" / "probes.jsonl",
        "public_sessions": dataset_dir / "public" / "lifelines.jsonl",
        "private_key": dataset_dir / "private" / "probe_key.jsonl",
        "subset_manifest": dataset_dir / "reports" / "official_subset_manifest.json",
        "domain_provenance": dataset_dir / "reports" / "domain_provenance_summary.json",
        "source_domain_contract_audit": dataset_dir / "reports" / "source_domain_contract_audit.json",
    }
    for label, path in required.items():
        if not path.exists():
            errors.append(f"missing_dataset_file:{label}:{path}")

    counts: Dict[str, Any] = {}
    if required["public_probes"].exists():
        counts["public_probes"] = line_count(required["public_probes"])
    if required["public_sessions"].exists():
        counts["public_sessions"] = line_count(required["public_sessions"])
    if required["private_key"].exists():
        counts["private_keys"] = line_count(required["private_key"])

    manifest = read_json(required["subset_manifest"]) if required["subset_manifest"].exists() else {}
    provenance = read_json(required["domain_provenance"]) if required["domain_provenance"].exists() else {}
    source_domain_audit = (
        read_json(required["source_domain_contract_audit"])
        if required["source_domain_contract_audit"].exists()
        else {}
    )
    manifest_counts = manifest.get("counts", {})
    expected_counts = {
        "public_probes": manifest_counts.get("probes", 90),
        "public_sessions": manifest_counts.get("sessions"),
        "private_keys": manifest_counts.get("keys", 90),
    }
    for key, expected in expected_counts.items():
        if expected is None:
            continue
        if counts.get(key) != expected:
            errors.append(f"dataset_count_mismatch:{key}:expected={expected}:actual={counts.get(key)}")
    if manifest.get("counts", {}).get("probes") != 90:
        errors.append("subset_manifest_probe_count_not_90")
    if provenance.get("status") != "pass":
        errors.append(f"domain_provenance_not_pass:{provenance.get('status')}")
    if source_domain_audit.get("status") != "pass":
        errors.append(f"source_domain_contract_audit_not_pass:{source_domain_audit.get('status')}")
    source_contract = manifest.get("source_contract", {})
    if source_contract.get("seed_prompts") != "allenai/WildChat":
        errors.append(f"source_contract_seed_prompts_unexpected:{source_contract.get('seed_prompts')}")
    if source_contract.get("family_domain_contract") != "nine_unique_representative_domains":
        errors.append(
            "source_contract_family_domain_contract_unexpected:"
            f"{source_contract.get('family_domain_contract')}"
        )
    if "different external dataset" not in source_contract.get("claim_to_avoid", ""):
        errors.append("source_contract_missing_external_dataset_boundary")
    provenance_contract = provenance.get("source_contract", {})
    if provenance_contract.get("family_domain_contract") != "nine_unique_representative_domains":
        errors.append(
            "provenance_family_domain_contract_unexpected:"
            f"{provenance_contract.get('family_domain_contract')}"
        )
    return {
        "dataset_dir": str(dataset_dir),
        "counts": counts,
        "manifest_status": manifest.get("status"),
        "provenance_status": provenance.get("status"),
        "source_domain_contract_audit_status": source_domain_audit.get("status"),
    }


def check_files(root: Path, errors: List[str]) -> List[str]:
    present = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if path.exists():
            present.append(rel)
        else:
            errors.append(f"missing_required_file:{rel}")
    return present


def compile_files(root: Path, errors: List[str]) -> List[str]:
    compiled = []
    for rel in COMPILE_FILES:
        path = root / rel
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            compiled.append(rel)
        except py_compile.PyCompileError as exc:
            errors.append(f"py_compile_failed:{rel}:{exc.msg}")
    return compiled


def check_imports() -> Dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_IMPORTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./runs/habit_bench_balanced_v0_3_official_subset_90"),
    )
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "root": str(args.root),
        "python": sys.version,
        "required_files_present": check_files(args.root, errors),
        "compiled": compile_files(args.root, errors),
        "dataset": check_dataset(args.dataset_dir, errors),
    }
    if args.check_imports:
        summary["optional_imports"] = check_imports()
    summary["status"] = "pass" if not errors else "fail"
    summary["errors"] = errors

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"status": summary["status"], "errors": len(errors), "root": str(args.root)}, indent=2))
    if errors:
        raise SystemExit("Lumia readiness check failed")


if __name__ == "__main__":
    main()
