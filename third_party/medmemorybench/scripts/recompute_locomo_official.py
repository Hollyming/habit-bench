#!/usr/bin/env python3
"""Recompute official LoCoMo QA scores from persisted model answers.

This is intentionally a non-destructive postprocessor: the input report and
its original scores are preserved, while the output adds ``official_score``
and ``official_is_correct`` fields.  The implementation mirrors
snap-research/locomo ``task_eval/evaluation.py`` at main commit
3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 (checked 2026-07-23).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.locomo_metrics import compute_f1, compute_multi_hop_f1


OFFICIAL_SOURCE = (
    "https://github.com/snap-research/locomo/blob/"
    "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py"
)
OFFICIAL_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"


def official_score(row: dict[str, Any]) -> float:
    """Return the official per-question LoCoMo QA score."""
    details = row.get("evaluation_details") or {}
    category = int(details.get("category"))
    prediction = str(row.get("model_output", ""))
    answer = str(row.get("expected_answer", ""))

    if category == 3:
        answer = answer.split(";", 1)[0].strip()
    if category == 1:
        return float(compute_multi_hop_f1(prediction, answer))
    if category in (2, 3, 4):
        return float(compute_f1(prediction, answer))
    if category == 5:
        lowered = prediction.lower()
        return float(
            "no information available" in lowered or "not mentioned" in lowered
        )
    raise ValueError(f"Unsupported LoCoMo category: {category!r}")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("query_type", "unknown"))].append(row)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        correct = sum(bool(item["official_is_correct"]) for item in items)
        return {
            "total": total,
            "mean_official_score": (
                sum(float(item["official_score"]) for item in items) / total
                if total
                else 0.0
            ),
            # This thresholded value is a MedMemoryBench diagnostic.  The
            # official LoCoMo headline QA metric is the mean score above.
            "threshold_0_5_correct": correct,
            "threshold_0_5_accuracy": correct / total if total else 0.0,
        }

    summary = stats(rows)
    summary["by_type"] = {
        query_type: stats(items) for query_type, items in sorted(by_type.items())
    }
    return summary


def recompute(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("dataset_name") != "locomo":
        raise ValueError(
            f"Expected dataset_name='locomo', found {payload.get('dataset_name')!r}"
        )
    rows = copy.deepcopy(payload.get("queries", []))
    if not rows:
        raise ValueError("Input query report contains no queries")

    for row in rows:
        score = official_score(row)
        row["official_score"] = score
        row["official_is_correct"] = score >= 0.5

    output = copy.deepcopy(payload)
    output["queries"] = rows
    output["official_locomo"] = {
        "contract": "locomo.qa.official_postprocess.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": OFFICIAL_COMMIT,
        "source_url": OFFICIAL_SOURCE,
        "headline_metric": "mean_official_score",
        "summary": _summarize(rows),
        "input_path": str(input_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output["official_locomo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(recompute(args.input, args.output), ensure_ascii=False, indent=2))
