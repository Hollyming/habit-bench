#!/usr/bin/env python3
"""Promote the validated false-probe candidate into a new v0.5 release only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V04 = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
SOURCE = V04 / "candidates" / "false_repair_selected_r2_r3"
OUT = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_5"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing release: {OUT}")
    source_manifest = json.loads((SOURCE / "candidate_manifest.json").read_text(encoding="utf-8"))
    public_probes = read_jsonl(SOURCE / "public" / "probes.jsonl")
    for probe in public_probes:
        metadata = probe.setdefault("metadata", {})
        metadata["dataset_version"] = "taskmaster_planning_defaults_v0_5"
        metadata.pop("candidate_only", None)
        metadata.pop("selection_policy", None)
    private_key = read_jsonl(SOURCE / "private" / "probe_key.jsonl")
    lifelines = read_jsonl(SOURCE / "public" / "lifelines.jsonl")
    sessions = read_jsonl(SOURCE / "private" / "sessions_with_annotations.jsonl")
    write_jsonl(OUT / "public" / "lifelines.jsonl", lifelines)
    write_jsonl(OUT / "public" / "probes.jsonl", public_probes)
    write_jsonl(OUT / "private" / "probe_key.jsonl", private_key)
    write_jsonl(OUT / "private" / "sessions_with_annotations.jsonl", sessions)
    report = {
        "release": "taskmaster_planning_defaults_v0_5",
        "base_release": "taskmaster_planning_defaults_v0_4",
        "sessions_unchanged_from_v0_4": sha256(OUT / "private" / "sessions_with_annotations.jsonl") == sha256(V04 / "private" / "sessions_with_annotations.jsonl"),
        "lifelines_unchanged_from_v0_4": sha256(OUT / "public" / "lifelines.jsonl") == sha256(V04 / "public" / "lifelines.jsonl"),
        "replacement_count": source_manifest["replacement_count"],
        "replacements": source_manifest["replacements"],
        "retained_original_false_probes": source_manifest["retained_original_false_probes"],
        "no_memory_calibration": {
            "model": "Qwen3-8B",
            "method": "no_memory",
            "total_accuracy": 0.267516,
            "total_correct": 42,
            "total_probes": 157,
            "false_personalization_accuracy": 0.235294,
            "false_personalization_correct": 4,
            "false_personalization_probes": 17,
        },
        "source_candidate_hashes": {
            "public_probes": sha256(SOURCE / "public" / "probes.jsonl"),
            "private_probe_key": sha256(SOURCE / "private" / "probe_key.jsonl"),
        },
    }
    report_path = OUT / "reports" / "v0_5_release_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "public" / "README.md").write_text(
        "# Taskmaster Planning Defaults v0.5\n\n"
        "v0.5 retains the v0.4 sessions and positive probes. It replaces six calibrated false-personalization probes after independent semantic audits and a Qwen3-8B no-memory calibration. Gold keys remain private.\n",
        encoding="utf-8",
    )
    print(json.dumps({"release": str(OUT), "probes": len(public_probes), "sessions": len(sessions), "replacements": source_manifest["replacement_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
