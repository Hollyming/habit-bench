#!/usr/bin/env python3
"""Create a query-only Qwen screening set from all dual-audited repair variants."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
OUT = DATASET / "candidates" / "false_repair_variant_screen"

spec = importlib.util.spec_from_file_location("stage_candidate", Path(__file__).with_name("15_stage_false_repair_candidate.py"))
assert spec and spec.loader
stage_candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_candidate)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    public = read_jsonl(DATASET / "public" / "probes.jsonl")
    private = {row["probe_id"]: row for row in read_jsonl(DATASET / "private" / "probe_key.jsonl")}
    old_scored = {row["probe_id"]: row for row in read_jsonl(DATASET / "evaluation_results" / "no_memory_v04_full_20260724" / "taskmaster_v04" / "no_memory" / "merged" / "scored_predictions.jsonl")}
    defective = [row for row in public if row.get("metadata", {}).get("probe_type") == "false_personalization" and old_scored[row["probe_id"]]["correct"]]
    candidates = stage_candidate.accepted_candidates()
    seed = (DATASET / "reports" / "generation_provenance" / "probe_shuffle_seed.txt").read_text().strip()
    staged_public: list[dict[str, Any]] = []
    staged_private: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    for old_probe in defective:
        old_key = private[old_probe["probe_id"]]
        control_id = old_key["control_id"]
        for ordinal, (revision, row, query_audit, history_audit) in enumerate(candidates.get(control_id, [])):
            probe_id = f"{old_probe['probe_id']}__v{ordinal:02d}"
            mapping = stage_candidate.stable_permutation(seed, probe_id)
            choices = [dict(choice, choice_id=mapping[choice["choice_id"]]) for choice in row["choices"]]
            choices.sort(key=lambda choice: choice["choice_id"])
            staged_public.append({
                "probe_id": probe_id, "user_id": old_probe["user_id"], "split": "test", "query": row["query"], "choices": choices,
                "visible_history_scope": old_probe["visible_history_scope"], "evaluation_contract": old_probe["evaluation_contract"],
                "metadata": {"dataset_version": "false_variant_screen", "probe_type": "false_personalization", "control_id": control_id, "candidate_only": True, "source_revision": revision},
            })
            staged_private.append(stage_candidate.remap({
                "probe_id": probe_id, "user_id": old_probe["user_id"], "probe_type": "false_personalization", "target_kind": "negative_control", "control_id": control_id,
                "gold_choice_id": row["gold_choice_id"], "gold_action": row.get("gold_action"), "gold_evidence_session_ids": row["gold_evidence_session_ids"],
                "label_rationale": row.get("label_rationale"), "negative_control": old_key["negative_control"], "query_only_judge": query_audit, "independent_gold_judge": history_audit,
                "label_source": "variant_screen_gpt55_xhigh_gpt56_terra_xhigh_dual_audit",
            }, mapping))
            manifest.append({"screen_probe_id": probe_id, "replaces_probe_id": old_probe["probe_id"], "control_id": control_id, "revision": revision, "variant_id": row["variant_id"]})
    if not staged_public:
        raise SystemExit("No dual-audited variants found")
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    write_jsonl(OUT / "public" / "lifelines.jsonl", read_jsonl(DATASET / "public" / "lifelines.jsonl"))
    write_jsonl(OUT / "public" / "probes.jsonl", staged_public)
    write_jsonl(OUT / "private" / "probe_key.jsonl", staged_private)
    write_jsonl(OUT / "private" / "sessions_with_annotations.jsonl", read_jsonl(DATASET / "private" / "sessions_with_annotations.jsonl"))
    (OUT / "screen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": str(OUT), "screen_variants": len(staged_public), "controls": len({row['control_id'] for row in manifest})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
