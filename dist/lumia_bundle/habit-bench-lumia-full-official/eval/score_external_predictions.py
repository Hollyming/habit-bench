#!/usr/bin/env python
"""Score an existing external-method prediction JSONL on HABIT-Bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_baselines import (  # noqa: E402
    evaluate_predictions,
    load_dataset,
    make_diagnostic_summary,
    write_csv,
    write_jsonl,
    write_report,
)
from run_external_baseline import rewrite_official_report  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, probes, keys = load_dataset(args.dataset_dir)
    raw_predictions = read_jsonl(args.predictions)

    probe_ids = {probe["probe_id"] for probe in probes}
    seen = {pred.get("probe_id") for pred in raw_predictions}
    missing = probe_ids - seen
    extra = seen - probe_ids
    if missing or extra:
        raise SystemExit(
            f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    predictions = [
        {
            "baseline": args.method_name,
            "probe_id": pred["probe_id"],
            "choice_id": pred["choice_id"],
            "scores": pred.get("scores", {}),
            "evidence_session_ids": pred.get("evidence_session_ids", []),
            "debug": pred.get("debug", {}),
            "cost": pred.get("cost", {}),
        }
        for pred in raw_predictions
    ]

    detailed, summary_rows = evaluate_predictions(predictions, keys)
    for row in summary_rows:
        row["baseline"] = args.method_name
        row["elapsed_sec"] = None
    diagnostic_rows = make_diagnostic_summary(summary_rows)

    write_jsonl(args.output_dir / f"{args.method_name}_scored_predictions.jsonl", detailed)
    write_csv(args.output_dir / f"{args.method_name}_metrics_summary.csv", summary_rows)
    write_csv(args.output_dir / f"{args.method_name}_diagnostic_summary.csv", diagnostic_rows)
    write_report(
        args.output_dir / f"{args.method_name}_baseline_report.md",
        [(args.method_name, 0.0, summary_rows)],
        diagnostic_rows,
    )
    rewrite_official_report(
        args.output_dir / f"{args.method_name}_baseline_report.md",
        args.method_name,
        args.adapter_note,
    )
    print(
        json.dumps(
            {
                "method_name": args.method_name,
                "predictions": len(raw_predictions),
                "output_dir": str(args.output_dir),
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument(
        "--adapter-note",
        default="External command used official code under the HABIT-Bench prediction contract.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
