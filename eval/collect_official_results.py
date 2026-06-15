#!/usr/bin/env python
"""Collect HABIT-Bench official-result directories into summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_method_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and first_matching(path, "_diagnostic_summary.csv") is not None
    )


def first_matching(path: Path, suffix: str) -> Path | None:
    matches = sorted(path.glob(f"*{suffix}"))
    return matches[0] if matches else None


def collect_method(path: Path) -> Dict[str, Any]:
    diagnostic_path = first_matching(path, "_diagnostic_summary.csv")
    metrics_path = first_matching(path, "_metrics_summary.csv")
    report_path = first_matching(path, "_baseline_report.md")
    stderr_path = first_matching(path, "_stderr.txt")
    raw_path = first_matching(path, "_raw_predictions.jsonl")
    scored_path = first_matching(path, "_scored_predictions.jsonl")
    config_paths = sorted(path.glob("*config*.json"))
    runtime_paths = sorted(path.glob("*runtime*.json"))

    row: Dict[str, Any] = {
        "method_dir": path.name,
        "status": "missing_diagnostic",
        "overall_accuracy": "",
        "explicit_retrieval_accuracy": "",
        "habit_direct_accuracy": "",
        "habit_stress_accuracy_weighted": "",
        "explicit_minus_stress_gap": "",
        "false_personalization_control_accuracy": "",
        "avg_retrieved_tokens_est": "",
        "avg_stored_items_est": "",
        "metrics_path": str(metrics_path) if metrics_path else "",
        "diagnostic_path": str(diagnostic_path) if diagnostic_path else "",
        "report_path": str(report_path) if report_path else "",
        "raw_predictions_path": str(raw_path) if raw_path else "",
        "scored_predictions_path": str(scored_path) if scored_path else "",
        "config_paths": ";".join(str(p) for p in config_paths),
        "runtime_paths": ";".join(str(p) for p in runtime_paths),
        "stderr_nonempty": False,
    }
    if stderr_path and stderr_path.exists():
        row["stderr_nonempty"] = bool(stderr_path.read_text(encoding="utf-8", errors="replace").strip())
    if diagnostic_path is None:
        return row

    diagnostic_rows = read_csv(diagnostic_path)
    if diagnostic_rows:
        row.update(diagnostic_rows[0])
        row["status"] = "ok"

    if metrics_path:
        for metric in read_csv(metrics_path):
            if metric.get("group_field") == "overall" and metric.get("group") == "overall":
                row["overall_accuracy"] = metric.get("accuracy", "")
                break
    return row


def markdown_table(rows: List[Dict[str, Any]], dataset_dir: Path, results_dir: Path) -> str:
    lines = [
        "# Official Results Collection",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Results dir: `{results_dir}`",
        f"- Methods found: {len(rows)}",
        "",
        "| method | status | overall | explicit | direct | stress | gap | false-pers ctrl | retrieved toks | stored items | config | runtime | stderr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method_dir} | {status} | {overall_accuracy} | {explicit_retrieval_accuracy} | {habit_direct_accuracy} | {habit_stress_accuracy_weighted} | {explicit_minus_stress_gap} | {false_personalization_control_accuracy} | {avg_retrieved_tokens_est} | {avg_stored_items_est} | {config_present} | {runtime_present} | {stderr_nonempty} |".format(
                config_present=bool(row.get("config_paths")),
                runtime_present=bool(row.get("runtime_paths")),
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "Rows produced by current `official_adapters` are official-code storage/retrieval",
            "adapter runs unless their method note explicitly states that full LLM-backed",
            "write/update/reasoning paths were enabled. Do not cite adapter rows as full",
            "paper reproductions.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or args.dataset_dir / "official_results"
    out_dir = args.out_dir or results_dir / "collected"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_method(path) for path in find_method_dirs(results_dir)]
    write_csv(out_dir / "official_results_collected.csv", rows)
    (out_dir / "official_results_collected.md").write_text(
        markdown_table(rows, args.dataset_dir, results_dir),
        encoding="utf-8",
    )
    (out_dir / "official_results_collected.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"methods": len(rows), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
