from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import DatasetBundle, DatasetContractError
from .io import write_csv, write_json, write_jsonl
from .retrieval_scoring import (
    RETRIEVAL_METRIC_DEFINITIONS,
    aggregate_retrieval_scores,
    balanced_decision_metrics,
    score_retrieval_predictions,
)


GROUP_FIELDS = (
    "domain",
    "probe_type",
    "capability_group",
    "habit_family",
    "stress_variant",
    "split",
)


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _dimension(probe: dict[str, Any], key: dict[str, Any], field: str) -> str:
    if field == "domain":
        return str(probe.get("domain", "unknown"))
    if field == "split":
        return str(probe.get("split", "unknown"))
    if field == "stress_variant":
        return str(key.get(field) or (probe.get("metadata") or {}).get(field) or "unknown")
    return str(key.get(field, "unknown"))


def validate_predictions(
    predictions: list[dict[str, Any]], bundle: DatasetBundle
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        probe_id = str(row.get("probe_id", ""))
        if not probe_id:
            raise DatasetContractError("Prediction missing probe_id")
        if probe_id in by_id:
            raise DatasetContractError(f"Duplicate prediction for {probe_id}")
        by_id[probe_id] = row
    expected = {probe["probe_id"] for probe in bundle.probes}
    missing = expected - set(by_id)
    extra = set(by_id) - expected
    if missing or extra:
        raise DatasetContractError(
            f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    for probe in bundle.probes:
        valid = {choice["choice_id"] for choice in probe["choices"]}
        choice_id = str(by_id[probe["probe_id"]].get("choice_id", ""))
        if choice_id not in valid:
            raise DatasetContractError(
                f"Invalid choice_id {choice_id!r} for probe {probe['probe_id']}; valid={sorted(valid)}"
            )
    return by_id


def _metric_row(
    method_name: str,
    group_field: str,
    group: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    low, high = _wilson_interval(correct, len(rows))
    memory_tokens = [
        int(row.get("answer", {}).get("memory_tokens_used") or 0) for row in rows
    ]
    return {
        "method_name": method_name,
        "group_field": group_field,
        "group": group,
        "correct": correct,
        "total": len(rows),
        "accuracy": round(correct / len(rows), 6),
        "accuracy_ci95_low": round(low, 6),
        "accuracy_ci95_high": round(high, 6),
        "avg_memory_tokens_used": round(sum(memory_tokens) / len(memory_tokens), 2),
        **aggregate_retrieval_scores(rows),
    }


def _capability_panels(
    detailed: list[dict[str, Any]],
    method_name: str,
) -> dict[str, dict[str, Any]]:
    probe_types = {row["probe_type"] for row in detailed}
    definitions: dict[str, set[str]] = {}
    if {"direct_use", "boundary", "exception", "explicit_retrieval"} & probe_types:
        definitions = {
            "habit_induction": {"direct_use"},
            "explicit_history_retrieval": {"explicit_retrieval"},
            "boundary_calibration": {"boundary"},
            "exception_retention": {"exception"},
        }
    elif any(
        row.get("retrieval", {}).get("gold_semantics") == "decisive_habit_evidence"
        for row in detailed
    ):
        definitions = {
            "temporal_and_drift_resolution": {
                "dual_asof_reversal",
                "scope_temporal_pair",
                "triple_asof_interleaved",
            },
            "provenance_and_decoy_rejection": {
                "suggestion_rejection_pair",
                "surface_decoy_pair",
                "provenance_weighted_triple",
            },
            "reference_case_reconstruction": {"reference_case_reconstruction"},
        }

    panels: dict[str, dict[str, Any]] = {}
    for panel_name, selected_types in definitions.items():
        rows = [row for row in detailed if row["probe_type"] in selected_types]
        if not rows:
            continue
        panel = _metric_row(method_name, "capability_panel", panel_name, rows)
        if panel_name == "boundary_calibration":
            panel["false_personalization_cost"] = round(1 - panel["accuracy"], 6)
        elif panel_name == "exception_retention":
            panel["exception_failure_rate"] = round(1 - panel["accuracy"], 6)
        elif panel_name == "habit_induction":
            panel["habit_induction_failure_rate"] = round(
                1 - panel["accuracy"], 6
            )
        panels[panel_name] = panel

    unseen_rows = [
        row for row in detailed if row.get("stress_variant") == "unseen_paraphrase"
    ]
    if unseen_rows:
        panels["unseen_paraphrase_robustness"] = _metric_row(
            method_name,
            "capability_panel",
            "unseen_paraphrase_robustness",
            unseen_rows,
        )
    return panels


def score_predictions(
    predictions: list[dict[str, Any]], bundle: DatasetBundle, method_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_id = validate_predictions(predictions, bundle)
    retrieval_scores = score_retrieval_predictions(by_id, bundle, method_name)
    detailed: list[dict[str, Any]] = []
    probes_by_id = {probe["probe_id"]: probe for probe in bundle.probes}
    for probe_id in [probe["probe_id"] for probe in bundle.probes]:
        prediction = by_id[probe_id]
        probe = probes_by_id[probe_id]
        key = bundle.keys[probe_id]
        row = {
            **prediction,
            "method_name": method_name,
            "gold_choice_id": key["gold_choice_id"],
            "correct": prediction["choice_id"] == key["gold_choice_id"],
            "retrieval": retrieval_scores[probe_id],
        }
        for field in GROUP_FIELDS:
            row[field] = _dimension(probe, key, field)
        detailed.append(row)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    buckets[("overall", "overall")] = detailed
    for row in detailed:
        for field in GROUP_FIELDS:
            buckets[(field, str(row[field]))].append(row)

    metric_rows: list[dict[str, Any]] = []
    for (group_field, group), rows in sorted(buckets.items()):
        metric_rows.append(_metric_row(method_name, group_field, group, rows))

    overall = next(row for row in metric_rows if row["group_field"] == "overall")
    balanced = balanced_decision_metrics(detailed, bundle)
    overall.update(balanced)
    capability_panels = _capability_panels(detailed, method_name)
    metrics = {
        "contract_version": "habitbench.choice_and_retrieval.v2",
        "method_name": method_name,
        "dataset": bundle.manifest,
        "primary_metric": "accuracy",
        "retrieval_primary_metric": (
            "evidence_recall_at_5_macro"
            if overall.get("retrieval_mode") == "ranked"
            else "context_evidence_recall_macro"
            if overall.get("retrieval_mode") == "context"
            else None
        ),
        "overall": overall,
        "balanced_aggregates": balanced,
        "capability_panels": capability_panels,
        "retrieval_metric_definitions": RETRIEVAL_METRIC_DEFINITIONS,
        "groups": metric_rows,
    }
    return detailed, metrics, metric_rows


def write_score_outputs(
    output_dir: Path,
    detailed: list[dict[str, Any]],
    metrics: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> None:
    write_jsonl(output_dir / "scored_predictions.jsonl", detailed)
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "metrics_by_group.csv", metric_rows)
    retrieval_keys = {
        key
        for row in metric_rows
        for key in row
        if (
            key.startswith("retrieval_")
            or "evidence" in key
            or "attribution" in key
            or "component" in key
            or "nonbinding" in key
            or "decoy" in key
            or key
            in {
                "method_name",
                "group_field",
                "group",
                "total",
                "accuracy",
                "temporal_context_recall_at_5",
                "temporal_context_recall_in_context",
                "clean_grounded_answer_rate_at_5",
            }
        )
    }
    retrieval_rows = [
        {key: value for key, value in row.items() if key in retrieval_keys}
        for row in metric_rows
    ]
    write_json(
        output_dir / "retrieval_metrics.json",
        {
            "contract_version": "habitbench.retrieval_metrics.v1",
            "method_name": metrics["method_name"],
            "dataset": metrics["dataset"],
            "primary_metric": metrics["retrieval_primary_metric"],
            "overall": {
                key: value
                for key, value in metrics["overall"].items()
                if key in retrieval_keys or key.startswith("decision_")
            },
            "balanced_aggregates": metrics["balanced_aggregates"],
            "capability_panels": metrics["capability_panels"],
            "metric_definitions": metrics["retrieval_metric_definitions"],
            "groups": retrieval_rows,
        },
    )
    write_csv(output_dir / "retrieval_metrics_by_group.csv", retrieval_rows)
