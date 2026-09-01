#!/usr/bin/env python
"""Batch supplementary analyses over a completed HABIT-Bench result suite."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_csv, write_json, write_jsonl
from eval.supplementary.analyze import (
    build_supplementary_analysis,
    validate_scored_rows,
)
from eval.supplementary.compare import compare_runs


DEFAULT_DATASETS = {
    "food": (
        PROJECT_ROOT / "domain/food/food_habit_lifelines_final_check",
        None,
    ),
    "finance": (
        PROJECT_ROOT
        / "domain/finance-software"
        / "habit_bench_multidogo_finance_software_release_gated_v1_4",
        "finance",
    ),
    "software": (
        PROJECT_ROOT
        / "domain/finance-software"
        / "habit_bench_multidogo_finance_software_release_gated_v1_4",
        "software",
    ),
    "travel": (
        PROJECT_ROOT
        / "domain/travel"
        / "release_candidate_v16_postrepair_repaired_r4",
        None,
    ),
}


def _load_suite_datasets(suite_root: Path) -> dict[str, tuple[Path, str | None]]:
    """Resolve the exact dataset versions recorded by a main-run manifest.

    Suites without ``shard_plan.manifest.json`` use the current four-domain
    release mapping.  Manifested suites always use their recorded versions.
    """

    manifest_path = suite_root / "shard_plan.manifest.json"
    if not manifest_path.is_file():
        return DEFAULT_DATASETS

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetContractError(
            f"Invalid suite manifest JSON: {manifest_path}: {exc}"
        ) from exc
    raw_datasets = manifest.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise DatasetContractError(
            f"Suite manifest has no non-empty datasets mapping: {manifest_path}"
        )

    datasets: dict[str, tuple[Path, str | None]] = {}
    for domain, raw_config in raw_datasets.items():
        if not isinstance(domain, str) or not domain or not isinstance(raw_config, dict):
            raise DatasetContractError(
                f"Malformed dataset entry in suite manifest: {domain!r}"
            )
        raw_path = raw_config.get("dataset_dir")
        if not isinstance(raw_path, str) or not raw_path:
            raise DatasetContractError(
                f"Dataset {domain!r} has no dataset_dir in {manifest_path}"
            )
        dataset_dir = Path(raw_path).expanduser()
        if not dataset_dir.is_absolute():
            dataset_dir = (PROJECT_ROOT / dataset_dir).resolve()
        domain_filter = raw_config.get("domain_filter")
        if domain_filter is not None and not isinstance(domain_filter, str):
            raise DatasetContractError(
                f"Dataset {domain!r} has invalid domain_filter={domain_filter!r}"
            )
        datasets[domain] = (dataset_dir, domain_filter)
    return datasets


def _split(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    if not selected:
        raise DatasetContractError("Comma-separated selection cannot be empty")
    return selected


def _parse_domain_suite_roots(
    values: list[str] | None,
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values or []:
        domain, separator, raw_path = value.partition("=")
        domain = domain.strip()
        raw_path = raw_path.strip()
        if not separator or not domain or not raw_path:
            raise DatasetContractError(
                "--domain-suite-root must use DOMAIN=PATH"
            )
        if domain in overrides:
            raise DatasetContractError(
                f"Duplicate suite override for domain {domain!r}"
            )
        suite_root = Path(raw_path).expanduser().resolve()
        if not suite_root.is_dir():
            raise FileNotFoundError(
                f"Domain suite root not found for {domain}: {suite_root}"
            )
        overrides[domain] = suite_root
    return overrides


def _discover_runs(
    suite_root: Path, domain: str, selected_methods: set[str] | None
) -> dict[str, Path]:
    domain_root = suite_root / domain
    runs: dict[str, Path] = {}
    if domain_root.is_dir():
        for method_dir in sorted(domain_root.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if selected_methods is not None and method not in selected_methods:
                continue
            scored = method_dir / "merged" / "scored_predictions.jsonl"
            if scored.is_file():
                runs[method] = scored

    # Compatibility with the older method/domain/method/merged suite layout.
    for method_root in sorted(suite_root.iterdir()):
        if not method_root.is_dir():
            continue
        method = method_root.name
        if selected_methods is not None and method not in selected_methods:
            continue
        scored = (
            method_root
            / domain
            / method
            / "merged"
            / "scored_predictions.jsonl"
        )
        if scored.is_file():
            runs.setdefault(method, scored)
    if selected_methods is not None:
        missing = selected_methods - set(runs)
        if missing:
            raise DatasetContractError(
                f"Missing completed {domain} merged results for {sorted(missing)}"
            )
    return runs


def run(args: argparse.Namespace) -> None:
    suite_root = args.suite_root.expanduser().resolve()
    if not suite_root.is_dir():
        raise FileNotFoundError(f"Suite root not found: {suite_root}")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else suite_root / "supplementary"
    )
    datasets = _load_suite_datasets(suite_root)
    domain_suite_roots = {domain: suite_root for domain in datasets}
    for domain, domain_suite_root in _parse_domain_suite_roots(
        args.domain_suite_root
    ).items():
        override_datasets = _load_suite_datasets(domain_suite_root)
        if domain not in override_datasets:
            raise DatasetContractError(
                f"Suite override for {domain!r} does not contain that domain: "
                f"{domain_suite_root}"
            )
        datasets[domain] = override_datasets[domain]
        domain_suite_roots[domain] = domain_suite_root
    selected_domains = _split(args.domains) or set(datasets)
    unknown_domains = selected_domains - set(datasets)
    if unknown_domains:
        raise DatasetContractError(f"Unknown domains: {sorted(unknown_domains)}")
    selected_methods = _split(args.methods)
    records: list[dict[str, object]] = []

    for domain in sorted(selected_domains):
        dataset_dir, domain_filter = datasets[domain]
        bundle = load_dataset(dataset_dir, domain_filter=domain_filter)
        domain_suite_root = domain_suite_roots[domain]
        paths = _discover_runs(
            domain_suite_root, domain, selected_methods
        )
        if not paths:
            raise DatasetContractError(
                f"No completed merged runs were found for {domain}"
            )
        loaded: dict[str, list[dict]] = {}
        for method, scored_path in sorted(paths.items()):
            rows = read_jsonl(scored_path)
            validate_scored_rows(rows, bundle)
            loaded[method] = rows
            analysis, slices, diagnostics, per_user = build_supplementary_analysis(
                rows,
                bundle,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                artifact_root=scored_path.parent,
                utility_lambdas=args.utility_lambda,
            )
            method_output = output_root / domain / method
            write_json(method_output / "supplementary_metrics.json", analysis)
            write_csv(
                method_output / "supplementary_metrics_by_slice.csv", slices
            )
            write_jsonl(
                method_output / "supplementary_probe_diagnostics.jsonl",
                diagnostics,
            )
            write_csv(
                method_output / "supplementary_metrics_by_user.csv", per_user
            )
            records.append(
                {
                    "domain": domain,
                    "method": method,
                    "source_suite": str(domain_suite_root),
                    "scored_predictions": str(scored_path),
                    "probes": len(rows),
                    "users": analysis["accuracy"]["users"],
                    "micro_accuracy": analysis["accuracy"]["micro_accuracy"],
                    "user_macro_accuracy": analysis["accuracy"][
                        "user_macro_accuracy"
                    ],
                    "output_dir": str(method_output),
                }
            )

        if len(loaded) >= 2:
            comparison, methods, pairs = compare_runs(
                loaded,
                bundle,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                allow_partial=False,
            )
            comparison_output = output_root / domain / "comparison"
            write_json(
                comparison_output / "supplementary_comparison.json",
                comparison,
            )
            write_csv(
                comparison_output / "supplementary_comparison_methods.csv",
                methods,
            )
            write_csv(
                comparison_output / "supplementary_comparison_pairs.csv", pairs
            )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "supplementary_summary.csv", records)
    write_json(
        output_root / "supplementary_manifest.json",
        {
            "contract_version": "habitbench.supplementary_batch.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_suite": str(suite_root),
            "source_suites_by_domain": {
                domain: str(domain_suite_roots[domain])
                for domain in sorted(selected_domains)
            },
            "output_root": str(output_root),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "domains": sorted(selected_domains),
            "methods": sorted({str(row["method"]) for row in records}),
            "runs": records,
        },
    )
    print(
        json.dumps(
            {"output_root": str(output_root), "runs": len(records)},
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument(
        "--domain-suite-root",
        action="append",
        default=None,
        help=(
            "Use DOMAIN=PATH to take one domain from a newer completed suite; "
            "repeat for multiple overrides."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--domains",
        help="Comma-separated; default is every dataset in the suite manifest.",
    )
    parser.add_argument("--methods", help="Comma-separated; default discovers completed methods.")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--utility-lambda", type=float, action="append", default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
