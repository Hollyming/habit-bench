from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import DatasetBundle, DatasetContractError
from .io import write_csv, write_json, write_jsonl


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


def score_predictions(
    predictions: list[dict[str, Any]], bundle: DatasetBundle, method_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_id = validate_predictions(predictions, bundle)
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
        correct = sum(bool(row["correct"]) for row in rows)
        low, high = _wilson_interval(correct, len(rows))
        memory_tokens = [
            int(row.get("answer", {}).get("memory_tokens_used") or 0) for row in rows
        ]
        metric_rows.append(
            {
                "method_name": method_name,
                "group_field": group_field,
                "group": group,
                "correct": correct,
                "total": len(rows),
                "accuracy": round(correct / len(rows), 6),
                "accuracy_ci95_low": round(low, 6),
                "accuracy_ci95_high": round(high, 6),
                "avg_memory_tokens_used": round(sum(memory_tokens) / len(memory_tokens), 2),
            }
        )

    overall = next(row for row in metric_rows if row["group_field"] == "overall")
    metrics = {
        "contract_version": "habitbench.choice_accuracy.v1",
        "method_name": method_name,
        "dataset": bundle.manifest,
        "primary_metric": "accuracy",
        "overall": overall,
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
