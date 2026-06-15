#!/usr/bin/env python
"""Audit whether the Lumia full official subset run is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_METHODS = {
    "mem0_full_llm_openai": {
        "config_glob": "*config*.json",
        "runtime_glob": "*runtime*.json",
        "required_config_contains": ["openai"],
    },
    "graphiti_full_llm_episode_kuzu": {
        "config_glob": "*config*.json",
        "runtime_glob": "*runtime*.json",
        "required_config_contains": ["OpenAIGenericClient", "structured_output_mode"],
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def has_nonempty(path: Path | None) -> bool:
    return bool(path and path.exists() and path.stat().st_size > 0)


def check_dataset(dataset_dir: Path, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    manifest_path = dataset_dir / "reports" / "official_subset_manifest.json"
    provenance_path = dataset_dir / "reports" / "domain_provenance_summary.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    counts = manifest.get("counts", {})
    expected_counts = {
        "probes": 90,
        "sessions": counts.get("sessions"),
        "users": counts.get("users"),
        "keys": counts.get("keys", 90),
    }
    for key, expected in expected_counts.items():
        if expected is None:
            continue
        if counts.get(key) != expected:
            errors.append(f"dataset_count_mismatch:{key}:expected={expected}:actual={counts.get(key)}")
    if provenance.get("status") != "pass":
        errors.append(f"domain_provenance_not_pass:{provenance.get('status')}")
    return {"manifest": str(manifest_path), "provenance": str(provenance_path), "counts": counts}


def file_text(paths: List[Path]) -> str:
    chunks = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return "\n".join(chunks)


def check_method(results_dir: Path, method_dir_name: str, spec: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    path = results_dir / method_dir_name
    row: Dict[str, Any] = {"method_dir": method_dir_name, "exists": path.exists()}
    if not path.exists():
        errors.append(f"missing_method_dir:{method_dir_name}")
        return row

    raw_path = first(path, "*_raw_predictions.jsonl")
    scored_path = first(path, "*_scored_predictions.jsonl")
    metrics_path = first(path, "*_metrics_summary.csv")
    diagnostic_path = first(path, "*_diagnostic_summary.csv")
    report_path = first(path, "*_baseline_report.md")
    stderr_path = first(path, "*_stderr.txt")
    config_paths = sorted(path.glob(spec["config_glob"]))
    runtime_paths = sorted(path.glob(spec["runtime_glob"]))

    required_paths = {
        "raw_predictions": raw_path,
        "scored_predictions": scored_path,
        "metrics_summary": metrics_path,
        "diagnostic_summary": diagnostic_path,
        "baseline_report": report_path,
    }
    for label, required_path in required_paths.items():
        if not has_nonempty(required_path):
            errors.append(f"missing_or_empty:{method_dir_name}:{label}")

    if not config_paths:
        errors.append(f"missing_config:{method_dir_name}")
    if not runtime_paths:
        errors.append(f"missing_runtime:{method_dir_name}")

    combined_config = file_text(config_paths)
    combined_runtime = file_text(runtime_paths)
    combined = combined_config + "\n" + combined_runtime
    if "dry_run_config" in combined and "true" in combined.lower():
        errors.append(f"dry_run_config_present_in_full_results:{method_dir_name}")

    for needle in spec.get("required_config_contains", []):
        if needle not in combined_config:
            errors.append(f"config_missing_marker:{method_dir_name}:{needle}")

    if method_dir_name == "mem0_full_llm_openai":
        # Mem0's full-path evidence is in the adapter command/config note rather than
        # a literal config flag; require the baseline report to mention infer=True.
        report_text = report_path.read_text(encoding="utf-8", errors="replace") if report_path else ""
        if (
            "infer=True" not in report_text
            and "infer=True" not in combined
            and "memory_add_infer" not in combined
        ):
            errors.append("mem0_full_path_marker_missing:infer=True_or_memory_add_infer")

    if stderr_path and stderr_path.read_text(encoding="utf-8", errors="replace").strip():
        warnings.append(f"stderr_nonempty:{method_dir_name}:{stderr_path}")

    row.update(
        {
            "raw_predictions": str(raw_path) if raw_path else None,
            "scored_predictions": str(scored_path) if scored_path else None,
            "metrics_summary": str(metrics_path) if metrics_path else None,
            "diagnostic_summary": str(diagnostic_path) if diagnostic_path else None,
            "baseline_report": str(report_path) if report_path else None,
            "stderr": str(stderr_path) if stderr_path else None,
            "config_paths": [str(path) for path in config_paths],
            "runtime_paths": [str(path) for path in runtime_paths],
        }
    )
    return row


def manifest_exit_code(path: Path) -> str | None:
    try:
        data = read_json(path)
    except Exception:
        return None
    extra = data.get("extra") or {}
    value = extra.get("exit_code")
    return None if value is None else str(value)


def check_run_manifests(
    results_dir: Path,
    run_manifests: List[Path],
    errors: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    suite_end = [path for path in run_manifests if path.name == "suite_end_manifest.json"]
    e2e_end = [path for path in run_manifests if path.name == "e2e_end_manifest.json"]
    preflight = [path for path in run_manifests if path.name == "lumia_preflight_manifest.json"]
    checked = []

    if not suite_end:
        errors.append(f"missing_suite_end_manifest:{results_dir / 'run_manifests'}")
    if not preflight:
        errors.append(f"missing_lumia_preflight_manifest:{results_dir / 'run_manifests'}")

    successful_suite_end = []
    successful_e2e_end = []
    for path in suite_end + e2e_end:
        code = manifest_exit_code(path)
        checked.append({"path": str(path), "exit_code": code})
        if code is None:
            warnings.append(f"missing_exit_code:{path}")
        elif code != "0":
            warnings.append(f"historical_nonzero_exit_code:{path}:{code}")
        elif path in suite_end:
            successful_suite_end.append(path)
        else:
            successful_e2e_end.append(path)

    if suite_end and not successful_suite_end:
        errors.append(f"missing_successful_suite_end_manifest:{results_dir / 'run_manifests'}")

    if not e2e_end:
        warnings.append(f"missing_e2e_end_manifest:{results_dir / 'run_manifests'}")
    elif not successful_e2e_end:
        warnings.append(f"missing_successful_e2e_end_manifest:{results_dir / 'run_manifests'}")

    preflight_checked = []
    for path in preflight:
        try:
            data = read_json(path)
        except Exception as exc:
            errors.append(f"preflight_manifest_unreadable:{path}:{type(exc).__name__}")
            preflight_checked.append({"path": str(path), "status": None})
            continue
        status = data.get("status")
        preflight_checked.append({"path": str(path), "status": status})
        if status != "pass":
            errors.append(f"preflight_manifest_not_pass:{path}:{status}")

    return {
        "all": [str(path) for path in run_manifests],
        "suite_end": [str(path) for path in suite_end],
        "e2e_end": [str(path) for path in e2e_end],
        "lumia_preflight": [str(path) for path in preflight],
        "checked_exit_codes": checked,
        "checked_preflight": preflight_checked,
        "successful_suite_end": [str(path) for path in successful_suite_end],
        "successful_e2e_end": [str(path) for path in successful_e2e_end],
    }


def check_model_download_manifest(
    manifest_path: Path,
    errors: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"path": str(manifest_path), "exists": manifest_path.exists()}
    if not has_nonempty(manifest_path):
        errors.append(f"missing_model_download_manifest:{manifest_path}")
        return row

    try:
        data = read_json(manifest_path)
    except Exception as exc:
        errors.append(f"model_download_manifest_unreadable:{manifest_path}:{type(exc).__name__}")
        row["read_error"] = type(exc).__name__
        return row

    models = data.get("models") or []
    row.update(
        {
            "status": data.get("status"),
            "dry_run": data.get("dry_run"),
            "model_count": len(models),
            "models": [
                {
                    "repo_id": model.get("repo_id"),
                    "status": model.get("status"),
                    "cache_path": model.get("cache_path"),
                }
                for model in models
            ],
        }
    )

    if data.get("dry_run"):
        errors.append(f"model_download_manifest_is_dry_run:{manifest_path}")
    if data.get("status") != "pass":
        errors.append(f"model_download_status_not_pass:{manifest_path}:{data.get('status')}")
    if len(models) < 2:
        errors.append(f"model_download_expected_at_least_two_models:{manifest_path}:actual={len(models)}")

    for model in models:
        repo_id = model.get("repo_id") or "<missing_repo_id>"
        if model.get("status") != "pass":
            errors.append(f"model_download_row_not_pass:{manifest_path}:{repo_id}:{model.get('status')}")
        if not model.get("cache_path"):
            errors.append(f"model_download_missing_cache_path:{manifest_path}:{repo_id}")

    manifest_errors = data.get("errors") or []
    if manifest_errors:
        warnings.append(f"model_download_manifest_errors_present:{manifest_path}:count={len(manifest_errors)}")

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("./runs/lumia_manifests/model_download_manifest.json"),
    )
    parser.add_argument("--skip-manifest-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or args.dataset_dir / "full_official_results"
    out_dir = args.out_dir or results_dir / "audit"
    errors: List[str] = []
    warnings: List[str] = []
    dataset = check_dataset(args.dataset_dir, errors, warnings)
    methods = [
        check_method(results_dir, method_dir, spec, errors, warnings)
        for method_dir, spec in EXPECTED_METHODS.items()
    ]
    collected = results_dir / "collected" / "official_results_collected.csv"
    if not has_nonempty(collected):
        errors.append(f"missing_collected_summary:{collected}")
    run_manifests = sorted((results_dir / "run_manifests").glob("*/*.json"))
    run_manifest_summary = check_run_manifests(results_dir, run_manifests, errors, warnings) if run_manifests else {
        "all": [],
        "suite_end": [],
        "e2e_end": [],
        "lumia_preflight": [],
        "checked_exit_codes": [],
        "checked_preflight": [],
    }
    model_manifest_summary: Dict[str, Any] = {"path": str(args.model_manifest), "skipped": True}
    if not args.skip_manifest_checks:
        model_manifest_summary = check_model_download_manifest(args.model_manifest, errors, warnings)
        if not run_manifests:
            errors.append(f"missing_run_manifests:{results_dir / 'run_manifests'}")
            errors.append(f"missing_lumia_preflight_manifest:{results_dir / 'run_manifests'}")

    summary = {
        "status": "pass" if not errors else "fail",
        "dataset_dir": str(args.dataset_dir),
        "results_dir": str(results_dir),
        "dataset": dataset,
        "expected_methods": list(EXPECTED_METHODS),
        "methods": methods,
        "collected_summary": str(collected),
        "model_manifest": str(args.model_manifest),
        "model_manifest_summary": model_manifest_summary,
        "run_manifests": run_manifest_summary,
        "errors": errors,
        "warnings": warnings,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_official_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Full Official Results Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Results: `{results_dir}`",
        f"- Expected methods: {', '.join(EXPECTED_METHODS)}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    (out_dir / "full_official_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "errors": len(errors), "warnings": len(warnings), "out_dir": str(out_dir)}, indent=2))
    if errors:
        raise SystemExit("Full official results audit failed")


if __name__ == "__main__":
    main()
