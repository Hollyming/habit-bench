#!/usr/bin/env python3
"""Stage, but never promote, an audited false-personalization repair candidate.

The candidate replaces only released false probes that Qwen3-8B answered with
an empty memory context.  It keeps every other released probe unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
STAGE = DATASET / "candidates" / "false_repair_r2_r3"
REVISIONS = ["v4_qwen_no_memory_repair_r2", "v4_qwen_no_memory_repair_r3"]
LABELS = ["A", "B", "C", "D"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def stable_permutation(seed: str, probe_id: str) -> dict[str, str]:
    ranked = sorted(LABELS, key=lambda old: hashlib.sha256(f"{seed}\0{probe_id}\0{old}".encode()).hexdigest())
    return {old: new for old, new in zip(ranked, LABELS)}


def remap(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [remap(item, mapping) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"choice_id", "gold_choice_id", "closest_distractor_choice_id", "generator_closest_distractor_choice_id"} and isinstance(item, str):
            out[key] = mapping.get(item, item)
        elif key == "plausible_choice_ids" and isinstance(item, list):
            out[key] = [mapping.get(x, x) for x in item]
        else:
            out[key] = remap(item, mapping)
    return out


def accepted_candidates() -> dict[str, list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    selected: dict[str, list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for revision in REVISIONS:
        report = json.loads((DATASET / "reports" / f"false_personalization_probe_generation_{revision}.json").read_text())
        allowed = {item["candidate_id"] for item in report["candidates"] if item.get("accepted")}
        work = DATASET / "work" / "false_personalization"
        for generated_path in sorted((work / f"probe_generation_{revision}").glob("*.json")):
            user_id = generated_path.stem
            generated = json.loads(generated_path.read_text()).get("probes", [])
            query_audit = {
                item.get("candidate_id"): item
                for item in json.loads((work / f"probe_query_audit_{revision}" / f"{user_id}.json").read_text()).get("answers", [])
            }
            history_audit: dict[str, dict[str, Any]] = {}
            for path in sorted((work / f"probe_history_audit_{revision}").glob(f"{user_id}_*.json")):
                history_audit.update({item.get("candidate_id"): item for item in json.loads(path.read_text()).get("answers", [])})
            for row in generated:
                candidate_id = f"{row.get('control_id')}::{row.get('variant_id')}"
                if candidate_id not in allowed:
                    continue
                selected.setdefault(row["control_id"], []).append((revision, row, query_audit[candidate_id], history_audit[candidate_id]))
    return selected


def main() -> None:
    public = read_jsonl(DATASET / "public" / "probes.jsonl")
    private = read_jsonl(DATASET / "private" / "probe_key.jsonl")
    keys = {row["probe_id"]: row for row in private}
    scored = read_jsonl(DATASET / "evaluation_results" / "no_memory_v04_full_20260724" / "taskmaster_v04" / "no_memory" / "merged" / "scored_predictions.jsonl")
    defective_ids = {
        row["probe_id"] for row in scored
        if row.get("probe_type") == "false_personalization" and row.get("correct") is True
    }
    if len(defective_ids) != 10:
        raise SystemExit(f"Expected 10 no-memory-solvable false probes, found {len(defective_ids)}")
    candidates = accepted_candidates()
    seed = (DATASET / "reports" / "generation_provenance" / "probe_shuffle_seed.txt").read_text().strip()

    # The only released defect without a same-control passing candidate receives
    # a second independently audited variant from the same user, rather than an
    # unreviewed or failed item.
    fallback_control = "ctrl_false_open_jaw_default"
    used_candidate_controls: set[str] = set()
    staged_public: list[dict[str, Any]] = []
    staged_private: list[dict[str, Any]] = []
    replacements: list[dict[str, str]] = []
    for probe in public:
        probe_id = probe["probe_id"]
        key = keys[probe_id]
        if probe_id not in defective_ids:
            staged_public.append(probe)
            staged_private.append(key)
            continue
        control_id = key["control_id"]
        chosen_control = control_id if control_id in candidates else fallback_control
        if chosen_control not in candidates:
            raise SystemExit(f"No audited candidate for {control_id}")
        # The fallback must be a distinct audited variant, not a duplicated
        # question already used for the open-jaw replacement.
        variant_index = 1 if chosen_control == fallback_control and control_id != fallback_control else 0
        if len(candidates[chosen_control]) <= variant_index:
            raise SystemExit(f"No distinct audited fallback variant for {control_id}")
        revision, row, query_audit, history_audit = candidates[chosen_control][variant_index]
        # Reuse the original released probe ID so coverage and user allocation
        # remain identical; remap choice positions deterministically.
        mapping = stable_permutation(seed, probe_id)
        choices = [copy.deepcopy(choice) for choice in row["choices"]]
        for choice in choices:
            choice["choice_id"] = mapping[choice["choice_id"]]
        choices.sort(key=lambda choice: choice["choice_id"])
        public_row = {
            **{field: probe[field] for field in ("probe_id", "user_id", "split", "visible_history_scope", "evaluation_contract")},
            "query": row["query"],
            "choices": choices,
            "metadata": {
                "dataset_version": "taskmaster_planning_defaults_v0_4_false_candidate",
                "probe_type": "false_personalization",
                "capability": "false_personalization",
                "control_id": chosen_control,
                "generated_by": "gpt-5.5",
                "source_revision": revision,
                "candidate_only": True,
            },
        }
        private_row = remap({
            "probe_id": probe_id,
            "user_id": probe["user_id"],
            "probe_type": "false_personalization",
            "target_kind": "negative_control",
            "control_id": chosen_control,
            "gold_choice_id": row["gold_choice_id"],
            "gold_action": row.get("gold_action"),
            "gold_evidence_session_ids": row["gold_evidence_session_ids"],
            "label_rationale": row.get("label_rationale"),
            "generator_difficulty_rationale": row.get("difficulty_rationale"),
            "negative_control": key["negative_control"],
            "query_only_judge": query_audit,
            "independent_gold_judge": history_audit,
            "label_source": "candidate_only_gpt55_xhigh_gpt56_terra_xhigh_dual_audit",
            "choice_position_balancing": "private_seeded_per_probe_shuffle",
            "choice_reference_remapped": True,
        }, mapping)
        staged_public.append(public_row)
        staged_private.append(private_row)
        replacements.append({"probe_id": probe_id, "old_control_id": control_id, "candidate_control_id": chosen_control, "revision": revision})
        used_candidate_controls.add(chosen_control)

    if STAGE.exists():
        shutil.rmtree(STAGE)
    write_jsonl(STAGE / "public" / "lifelines.jsonl", read_jsonl(DATASET / "public" / "lifelines.jsonl"))
    write_jsonl(STAGE / "public" / "probes.jsonl", staged_public)
    write_jsonl(STAGE / "private" / "probe_key.jsonl", staged_private)
    write_jsonl(STAGE / "private" / "sessions_with_annotations.jsonl", read_jsonl(DATASET / "private" / "sessions_with_annotations.jsonl"))
    (STAGE / "candidate_manifest.json").write_text(json.dumps({
        "candidate_only": True,
        "base_dataset": str(DATASET),
        "replaced_no_memory_solvable_false_probes": replacements,
        "retained_original_false_probes": 7,
        "total_probes": len(staged_public),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": str(STAGE), "replacements": len(replacements), "probes": len(staged_public)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
