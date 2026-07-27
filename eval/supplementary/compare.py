#!/usr/bin/env python
"""Compare completed methods with paired, user-clustered inference.

Inputs are existing ``scored_predictions.jsonl`` files.  All methods must cover
the same probes.  The script emits separate supplementary outputs and never
changes HABIT-Bench's primary scorer or metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.core.dataset import DatasetBundle, DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_csv, write_json
from eval.supplementary.analyze import (
    cluster_bootstrap_ci,
    user_accuracy_rows,
    validate_scored_rows,
)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be METHOD=PATH")
    method, raw_path = value.split("=", 1)
    method = method.strip()
    path = Path(raw_path).expanduser()
    if not method or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must be METHOD=PATH")
    return method, path


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_user_values(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    bundle: DatasetBundle,
) -> dict[str, float]:
    probes = {probe["probe_id"]: probe for probe in bundle.probes}
    values: dict[str, list[float]] = defaultdict(list)
    for probe_id in sorted(left):
        user_id = str(probes[probe_id]["user_id"])
        values[user_id].append(
            float(bool(left[probe_id]["correct"]))
            - float(bool(right[probe_id]["correct"]))
        )
    return {
        user_id: sum(user_values) / len(user_values)
        for user_id, user_values in values.items()
    }


def paired_cluster_bootstrap(
    user_differences: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    users = sorted(user_differences)
    if not users:
        raise DatasetContractError("No paired users are available")
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected = [generator.choice(users) for _ in users]
        draws.append(
            sum(user_differences[user_id] for user_id in selected) / len(selected)
        )
    nonpositive = sum(value <= 0 for value in draws)
    nonnegative = sum(value >= 0 for value in draws)
    p_value = min(
        1.0,
        2
        * min(
            (nonpositive + 1) / (samples + 1),
            (nonnegative + 1) / (samples + 1),
        ),
    )
    return {
        "user_macro_accuracy_difference": _rounded(
            sum(user_differences.values()) / len(users)
        ),
        "ci95_low": _rounded(_percentile(draws, 0.025)),
        "ci95_high": _rounded(_percentile(draws, 0.975)),
        "two_sided_bootstrap_p": _rounded(p_value),
        "samples": samples,
        "clusters": len(users),
        "seed": seed,
        "resampling_unit": "user",
    }


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def mcnemar_exact(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    left_only = sum(
        bool(left[probe_id]["correct"]) and not bool(right[probe_id]["correct"])
        for probe_id in left
    )
    right_only = sum(
        bool(right[probe_id]["correct"]) and not bool(left[probe_id]["correct"])
        for probe_id in left
    )
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(left_only, right_only)
        log_terms = [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2)
            for k in range(tail + 1)
        ]
        p_value = min(1.0, 2 * math.exp(_logsumexp(log_terms)))
    return {
        "left_correct_right_wrong": left_only,
        "right_correct_left_wrong": right_only,
        "discordant_probes": discordant,
        "two_sided_exact_p": _rounded(p_value),
    }


def holm_adjust(rows: list[dict[str, Any]], field: str) -> None:
    ranked = sorted(
        enumerate(rows), key=lambda item: float(item[1][field])
    )
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ranked):
        adjusted = min(1.0, (total - rank) * float(row[field]))
        running = max(running, adjusted)
        rows[original_index][f"{field}_holm"] = round(running, 6)


def compare_runs(
    runs: dict[str, list[dict[str, Any]]],
    bundle: DatasetBundle,
    *,
    bootstrap_samples: int,
    seed: int,
    allow_partial: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(runs) < 2:
        raise DatasetContractError("At least two --run inputs are required")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: set[str] | None = None
    for method, rows in runs.items():
        by_id = validate_scored_rows(rows, bundle, allow_partial=allow_partial)
        if coverage is None:
            coverage = set(by_id)
        elif set(by_id) != coverage:
            raise DatasetContractError(
                f"All compared methods must cover identical probes; {method} differs"
            )
        indexed[method] = by_id
    assert coverage is not None

    method_rows: list[dict[str, Any]] = []
    for method, by_id in sorted(indexed.items()):
        rows = [by_id[probe_id] for probe_id in sorted(coverage)]
        per_user = user_accuracy_rows(rows, bundle)
        user_values = {
            str(row["user_id"]): float(row["accuracy"]) for row in per_user
        }
        correct = sum(bool(row["correct"]) for row in rows)
        bootstrap = cluster_bootstrap_ci(
            user_values, samples=bootstrap_samples, seed=seed
        )
        method_rows.append(
            {
                "method": method,
                "probes": len(rows),
                "users": len(user_values),
                "correct": correct,
                "micro_accuracy": round(correct / len(rows), 6),
                "user_macro_accuracy": bootstrap["estimate"],
                "user_macro_ci95_low": bootstrap["ci95_low"],
                "user_macro_ci95_high": bootstrap["ci95_high"],
            }
        )

    pair_rows: list[dict[str, Any]] = []
    methods = sorted(indexed)
    for left_index, left_name in enumerate(methods):
        for pair_index, right_name in enumerate(methods[left_index + 1 :]):
            left = indexed[left_name]
            right = indexed[right_name]
            paired = paired_cluster_bootstrap(
                _paired_user_values(left, right, bundle),
                samples=bootstrap_samples,
                seed=seed + left_index * len(methods) + pair_index + 1,
            )
            mcnemar = mcnemar_exact(left, right)
            left_micro = sum(bool(row["correct"]) for row in left.values()) / len(
                left
            )
            right_micro = sum(bool(row["correct"]) for row in right.values()) / len(
                right
            )
            pair_rows.append(
                {
                    "left_method": left_name,
                    "right_method": right_name,
                    "micro_accuracy_difference": round(
                        left_micro - right_micro, 6
                    ),
                    **paired,
                    **mcnemar,
                }
            )
    holm_adjust(pair_rows, "two_sided_bootstrap_p")
    holm_adjust(pair_rows, "two_sided_exact_p")
    payload = {
        "contract_version": "habitbench.supplementary_comparison.v1",
        "analysis_role": (
            "paired supplementary inference; primary benchmark metrics are unchanged"
        ),
        "dataset": bundle.manifest,
        "coverage": {"probes": len(coverage), "methods": len(runs)},
        "inference": {
            "primary_resampling_unit": "user",
            "cluster_bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "secondary_test": "two-sided exact McNemar on paired probes",
            "multiple_comparison_correction": "Holm family-wise correction",
        },
        "methods": method_rows,
        "pairs": pair_rows,
    }
    return payload, method_rows, pair_rows


def run(args: argparse.Namespace) -> None:
    bundle = load_dataset(args.dataset_dir, domain_filter=args.domain_filter)
    parsed = [_parse_run(value) for value in args.run]
    names = [name for name, _ in parsed]
    if len(names) != len(set(names)):
        raise DatasetContractError("Duplicate method name in --run")
    runs = {name: read_jsonl(path) for name, path in parsed}
    payload, methods, pairs = compare_runs(
        runs,
        bundle,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        allow_partial=args.allow_partial,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "supplementary_comparison.json", payload)
    write_csv(args.output_dir / "supplementary_comparison_methods.csv", methods)
    write_csv(args.output_dir / "supplementary_comparison_pairs.csv", pairs)
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="METHOD=/path/to/scored_predictions.jsonl; repeat at least twice.",
    )
    parser.add_argument("--domain-filter")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
