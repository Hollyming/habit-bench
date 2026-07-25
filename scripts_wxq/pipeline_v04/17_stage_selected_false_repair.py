#!/usr/bin/env python3
"""Stage a quality-first false-probe repair chosen from screened audited variants."""
from __future__ import annotations

import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
SCREEN = DATASET / "candidates" / "false_repair_variant_screen"
OUT = DATASET / "candidates" / "false_repair_selected_r2_r3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    public = read_jsonl(DATASET / "public" / "probes.jsonl")
    private = {row["probe_id"]: row for row in read_jsonl(DATASET / "private" / "probe_key.jsonl")}
    screen_public = {row["probe_id"]: row for row in read_jsonl(SCREEN / "public" / "probes.jsonl")}
    screen_private = {row["probe_id"]: row for row in read_jsonl(SCREEN / "private" / "probe_key.jsonl")}
    screen_scored = {row["probe_id"]: row for row in read_jsonl(SCREEN / "evaluation_results" / "no_memory_variant_screen_20260725" / "taskmaster_v04" / "no_memory" / "merged" / "scored_predictions.jsonl")}
    manifest = json.loads((SCREEN / "screen_manifest.json").read_text())

    # Keep the first semantically audited variant that Qwen did not solve with
    # empty memory. This is an aggregate calibration aid, not a semantic gate:
    # every selected row already passed the independent dual audit.
    selected: dict[str, dict[str, str]] = {}
    for item in manifest:
        screen_id = item["screen_probe_id"]
        if screen_scored[screen_id]["correct"] is False:
            selected.setdefault(item["replaces_probe_id"], item)
    if len(selected) != 6:
        raise SystemExit(f"Expected six screened replacements, found {len(selected)}")

    out_public: list[dict[str, Any]] = []
    out_private: list[dict[str, Any]] = []
    for original in public:
        probe_id = original["probe_id"]
        item = selected.get(probe_id)
        if item is None:
            out_public.append(original)
            out_private.append(private[probe_id])
            continue
        screen_id = item["screen_probe_id"]
        p = copy.deepcopy(screen_public[screen_id]); k = copy.deepcopy(screen_private[screen_id])
        p["probe_id"] = probe_id; k["probe_id"] = probe_id
        p["metadata"]["candidate_only"] = True
        p["metadata"]["selection_policy"] = "dual_audit_then_aggregate_no_memory_calibration"
        out_public.append(p); out_private.append(k)

    if OUT.exists(): shutil.rmtree(OUT)
    write_jsonl(OUT / "public" / "lifelines.jsonl", read_jsonl(DATASET / "public" / "lifelines.jsonl"))
    write_jsonl(OUT / "public" / "probes.jsonl", out_public)
    write_jsonl(OUT / "private" / "probe_key.jsonl", out_private)
    write_jsonl(OUT / "private" / "sessions_with_annotations.jsonl", read_jsonl(DATASET / "private" / "sessions_with_annotations.jsonl"))
    (OUT / "candidate_manifest.json").write_text(json.dumps({
        "candidate_only": True, "replacements": list(selected.values()), "replacement_count": len(selected),
        "retained_original_false_probes": 11, "selection_policy": "dual_audit_then_aggregate_no_memory_calibration",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": str(OUT), "replacements": len(selected), "probes": len(out_public)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
