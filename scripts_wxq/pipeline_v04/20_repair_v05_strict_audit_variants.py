#!/usr/bin/env python3
"""Swap two v0.5 rows to their non-ambiguous independently audited variants."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V05 = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_5"
SCREEN = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4" / "candidates" / "false_repair_variant_screen"
TARGETS = {
    "tm_pd_v04_user_002_fp_005": "tm_pd_v04_user_002_c04_repl_02",
    "tm_pd_v04_user_003_fp_008": "ctrl_false_open_jaw_default_v5_gulf_coast_msy_roundtrip",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    manifest = json.loads((SCREEN / "screen_manifest.json").read_text())
    variant_to_screen = {item["variant_id"]: item["screen_probe_id"] for item in manifest}
    screen_public = {row["probe_id"]: row for row in read_jsonl(SCREEN / "public" / "probes.jsonl")}
    screen_private = {row["probe_id"]: row for row in read_jsonl(SCREEN / "private" / "probe_key.jsonl")}
    replacements = {}
    for target_id, variant_id in TARGETS.items():
        screen_id = variant_to_screen[variant_id]
        public_row = copy.deepcopy(screen_public[screen_id]); private_row = copy.deepcopy(screen_private[screen_id])
        public_row["probe_id"] = target_id; private_row["probe_id"] = target_id
        public_row["metadata"]["dataset_version"] = "taskmaster_planning_defaults_v0_5"
        public_row["metadata"].pop("candidate_only", None)
        public_row["metadata"].pop("selection_policy", None)
        replacements[target_id] = (public_row, private_row, variant_id, screen_id)
    public = read_jsonl(V05 / "public" / "probes.jsonl")
    private = read_jsonl(V05 / "private" / "probe_key.jsonl")
    write_jsonl(V05 / "public" / "probes.jsonl", [replacements.get(row["probe_id"], (row,))[0] for row in public])
    write_jsonl(V05 / "private" / "probe_key.jsonl", [replacements.get(row["probe_id"], (None, row))[1] for row in private])
    report_path = V05 / "reports" / "v0_5_release_manifest.json"
    report = json.loads(report_path.read_text())
    for item in report["replacements"]:
        if item["replaces_probe_id"] in replacements:
            item["variant_id"] = replacements[item["replaces_probe_id"]][2]
            item["screen_probe_id"] = replacements[item["replaces_probe_id"]][3]
    report["no_memory_calibration"] = {"status": "pending_rerun_after_strict_audit_variant_swap"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"strict_variant_swaps": TARGETS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
