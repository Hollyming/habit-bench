#!/usr/bin/env python
"""Run an external official baseline command and score it on HABIT-Bench.

The external command receives two file paths via placeholders:

    --command "python official_method.py --input {input} --output {output}"

Input JSON format:

{
  "sessions_by_user": {"user_0001": [...]},
  "probes": [...]
}

The command must write JSONL predictions:

{"probe_id": "...", "choice_id": "A", "evidence_session_ids": ["..."]}

This runner validates coverage and computes the same metrics used by the
lightweight baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rewrite_official_report(path: Path, method_name: str, adapter_note: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "These are lightweight method-inspired baselines, not official package integrations.\n"
        "They are intended to validate the benchmark/evaluator loop before scaling.",
        "This is an external official-code adapter run, not a lightweight proxy baseline.\n"
        f"Adapter note: {adapter_note}",
    )
    text = text.replace(
        "The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.",
        f"The table reports HABIT-Bench metrics for `{method_name}` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.",
    )
    text = text.replace(
        "Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.",
        "Official conclusions should be made only for the exact method configuration represented by this adapter run.",
    )
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions_by_user, probes, keys = load_dataset(args.dataset_dir)

    input_path = args.output_dir / f"{args.method_name}_official_input.json"
    raw_predictions_path = args.output_dir / f"{args.method_name}_raw_predictions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "method_name": args.method_name,
                "sessions_by_user": sessions_by_user,
                "probes": probes,
                "prediction_contract": {
                    "required_fields": ["probe_id", "choice_id"],
                    "optional_fields": ["evidence_session_ids", "debug", "cost"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = args.command.format(input=str(input_path), output=str(raw_predictions_path))
    stdout_path = args.output_dir / f"{args.method_name}_stdout.txt"
    stderr_path = args.output_dir / f"{args.method_name}_stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(
            shlex.split(command, posix=os.name != "nt"),
            cwd=args.cwd,
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            timeout=args.timeout_sec,
        )
    if completed.returncode != 0:
        raise SystemExit(f"External baseline failed with code {completed.returncode}; see stderr file")
    if not raw_predictions_path.exists():
        raise SystemExit(f"External baseline did not write {raw_predictions_path}")

    raw_predictions = read_jsonl(raw_predictions_path)
    probe_ids = {probe["probe_id"] for probe in probes}
    seen = {pred.get("probe_id") for pred in raw_predictions}
    missing = probe_ids - seen
    extra = seen - probe_ids
    if missing or extra:
        raise SystemExit(
            f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    predictions = []
    for pred in raw_predictions:
        predictions.append(
            {
                "baseline": args.method_name,
                "probe_id": pred["probe_id"],
                "choice_id": pred["choice_id"],
                "scores": pred.get("scores", {}),
                "evidence_session_ids": pred.get("evidence_session_ids", []),
                "debug": pred.get("debug", {}),
                "cost": pred.get("cost", {}),
            }
        )

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
    print(f"Wrote official baseline metrics for {args.method_name} to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--adapter-note",
        default="External command used official code under the HABIT-Bench prediction contract.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
