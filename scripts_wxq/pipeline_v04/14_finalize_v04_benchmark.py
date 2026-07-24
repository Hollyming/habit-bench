#!/usr/bin/env python3
"""Release-gate and merge the validated v0.4 positive and false-personalization probes.

This is deliberately a data-only finalization step: it never edits sessions,
dossiers, controls, or prior probe files.  It accepts a false probe only when
both independent audit artifacts still satisfy the release contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
LABELS = ["A", "B", "C", "D"]

SOURCES = [
    ("v1", "false_personalization_probe_generation_v1.json", "probe_generation", "probe_query_audit", "probe_history_audit", "false_personalization_controls_v1.jsonl"),
    ("v2", "false_personalization_probe_generation_v2_repair.json", "probe_generation_v2_repair", "probe_query_audit_v2_repair", "probe_history_audit_v2_repair", "false_personalization_controls_v1.jsonl"),
    ("v3", "false_personalization_probe_generation_v3_feedback_repair.json", "probe_generation_v3_feedback_repair", "probe_query_audit_v3_feedback_repair", "probe_history_audit_v3_feedback_repair", "false_personalization_controls_v1.jsonl"),
    ("u005_recovery", "false_personalization_probe_generation_u005_probe_recovery_v2.json", "probe_generation_u005_probe_recovery_v2", "probe_query_audit_u005_probe_recovery_v2", "probe_history_audit_u005_probe_recovery_v2", "false_personalization_controls_u005_recovery_v2.jsonl"),
]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

def candidate_id(row: dict[str, Any]) -> str:
    return f"{row.get('control_id')}::{row.get('variant_id')}"

def audit_answers(root: Path, user_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    qpath = root / "probe_query_audit" / f"{user_id}.json"
    qraw = read_json(qpath)
    q = {x.get("candidate_id"): x for x in qraw.get("answers", []) if isinstance(x, dict)}
    h: dict[str, Any] = {}
    for path in sorted((root / "probe_history_audit").glob(f"{user_id}_*.json")):
        h.update({x.get("candidate_id"): x for x in read_json(path).get("answers", []) if isinstance(x, dict)})
    return q, h

def validate(row: dict[str, Any], query: dict[str, Any], history: dict[str, Any], valid_sessions: set[str], episode_by_session: dict[str, str | None]) -> list[str]:
    errors: list[str] = []
    choices = row.get("choices", [])
    ids = [x.get("choice_id") for x in choices if isinstance(x, dict)]
    gold = row.get("gold_choice_id")
    lengths = [len(str(x.get("text", "")).strip()) for x in choices if isinstance(x, dict)]
    cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
    if len(choices) != 4 or set(ids) != set(LABELS) or gold not in LABELS: errors.append("choice_schema")
    if not lengths or min(lengths) < 30 or max(lengths) > 2.5 * min(lengths): errors.append("choice_length")
    if len(cited) < 4 or not set(cited).issubset(valid_sessions): errors.append("evidence_ids")
    elif len({episode_by_session[s] for s in cited}) < 3: errors.append("evidence_episodes")
    plausible = query.get("plausible_choice_ids")
    if query.get("choice_id") != "UNRESOLVED" or query.get("answerable_without_history") is not False or query.get("generic_best_exists") is not False or not isinstance(plausible, list) or len(set(plausible)) < 2 or gold not in plausible:
        errors.append("query_only_contract")
    if history.get("choice_id") != gold or history.get("ambiguous") is not False or history.get("stable_preference_supported") is not False or history.get("difficulty") not in {"medium", "hard"}:
        errors.append("history_contract")
    return errors

def stable_permutation(seed: str, namespace: str) -> dict[str, str]:
    ranked = sorted(LABELS, key=lambda old: hashlib.sha256(f"{seed}\0{namespace}\0{old}".encode()).hexdigest())
    return {old: new for old, new in zip(ranked, LABELS)}

def remap(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list): return [remap(x, mapping) for x in value]
    if not isinstance(value, dict): return value
    out = {}
    for key, item in value.items():
        if key in {"choice_id", "gold_choice_id", "closest_distractor_choice_id", "generator_closest_distractor_choice_id"} and isinstance(item, str):
            out[key] = mapping.get(item, item)
        elif key == "plausible_choice_ids" and isinstance(item, list):
            out[key] = [mapping.get(x, x) for x in item]
        else:
            out[key] = remap(item, mapping)
    return out

def main() -> None:
    sessions = read_jsonl(DATASET / "private" / "sessions_with_annotations.jsonl")
    valid_sessions = {s["session_id"] for s in sessions}
    episode_by_session = {s["session_id"]: s.get("memory_annotations", {}).get("episode_id") for s in sessions}
    controls: dict[str, dict[str, Any]] = {}
    selected: dict[tuple[str, str], tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    rejections: list[dict[str, Any]] = []

    for tag, report_name, gen_dir, q_dir, h_dir, control_name in SOURCES:
        controls.update({x["control_id"]: x for x in read_jsonl(DATASET / "private" / control_name)})
        report = read_json(DATASET / "reports" / report_name)
        by_user = defaultdict(list)
        for item in report.get("candidates", []):
            if not item.get("errors"):
                by_user[item["user_id"]].append(item["candidate_id"])
        for user_id, allowed_ids in by_user.items():
            rows = read_json(DATASET / "work" / "false_personalization" / gen_dir / f"{user_id}.json").get("probes", [])
            q, h = audit_answers(DATASET / "work" / "false_personalization" / q_dir, user_id) if False else ({}, {})
            # Audit artifacts live in separate roots, so load them directly here.
            qraw = read_json(DATASET / "work" / "false_personalization" / q_dir / f"{user_id}.json")
            q = {x.get("candidate_id"): x for x in qraw.get("answers", []) if isinstance(x, dict)}
            h = {}
            for path in sorted((DATASET / "work" / "false_personalization" / h_dir).glob(f"{user_id}_*.json")):
                h.update({x.get("candidate_id"): x for x in read_json(path).get("answers", []) if isinstance(x, dict)})
            for row in rows:
                cid = candidate_id(row)
                if cid not in allowed_ids: continue
                errors = validate(row, q.get(cid, {}), h.get(cid, {}), valid_sessions, episode_by_session)
                key = (user_id, row["control_id"])
                if errors:
                    rejections.append({"source": tag, "candidate_id": cid, "errors": errors})
                elif key not in selected:
                    selected[key] = (tag, copy.deepcopy(row), copy.deepcopy(q[cid]), copy.deepcopy(h[cid]))

    missing_users = sorted({s["user_id"] for s in sessions} - {user for user, _ in selected})
    if rejections or missing_users:
        raise SystemExit(json.dumps({"release_gate": "fail", "rejections": rejections, "missing_negative_users": missing_users}, ensure_ascii=False))

    seed = (DATASET / "private" / "probe_shuffle_seed.txt").read_text(encoding="utf-8").strip()
    public_rows, private_rows = [], []
    for ordinal, ((user_id, control_id), (tag, row, q, h)) in enumerate(sorted(selected.items())):
        probe_id = f"{user_id}_fp_{ordinal:03d}"
        mapping = stable_permutation(seed, probe_id)
        choices = []
        for item in row["choices"]:
            item = copy.deepcopy(item); item["choice_id"] = mapping[item["choice_id"]]; choices.append(item)
        choices.sort(key=lambda x: x["choice_id"])
        public_rows.append({
            "probe_id": probe_id, "user_id": user_id, "split": "test", "query": row["query"], "choices": choices,
            "visible_history_scope": {"max_session_index": max(s["session_index"] for s in sessions if s["user_id"] == user_id)},
            "evaluation_contract": {"answer_format": "return one choice_id", "validator_type": "choice_equals"},
            "metadata": {"dataset_version": "taskmaster_planning_defaults_v0_4", "probe_type": "false_personalization", "capability": "false_personalization", "control_id": control_id, "generated_by": "gpt-5.5", "source_revision": tag},
        })
        private_rows.append(remap({
            "probe_id": probe_id, "user_id": user_id, "probe_type": "false_personalization", "target_kind": "negative_control", "control_id": control_id,
            "gold_choice_id": row["gold_choice_id"], "gold_action": row.get("gold_action"), "gold_evidence_session_ids": row["gold_evidence_session_ids"],
            "label_rationale": row.get("label_rationale"), "generator_difficulty_rationale": row.get("difficulty_rationale"),
            "negative_control": controls[control_id], "query_only_judge": q, "independent_gold_judge": h,
            "label_source": "gpt55_xhigh_generation_gpt56_terra_xhigh_dual_audit", "choice_position_balancing": "private_seeded_per_probe_shuffle", "choice_reference_remapped": True,
        }, mapping))

    positive_public = read_jsonl(DATASET / "public" / "probes.jsonl")
    positive_private = read_jsonl(DATASET / "private" / "probe_key.jsonl")
    public_ids = {x["probe_id"] for x in public_rows}
    if public_ids & {x["probe_id"] for x in positive_public}: raise SystemExit("probe_id collision")
    write_jsonl(DATASET / "public" / "false_personalization_probes_v2.jsonl", public_rows)
    write_jsonl(DATASET / "private" / "false_personalization_probe_key_v2.jsonl", private_rows)
    write_jsonl(DATASET / "public" / "benchmark_probes_v0_4.jsonl", positive_public + public_rows)
    write_jsonl(DATASET / "private" / "benchmark_probe_key_v0_4.jsonl", positive_private + private_rows)
    report = {"release_gate": "pass", "sessions": len(sessions), "users": len({s["user_id"] for s in sessions}), "positive_probe_count": len(positive_public), "false_personalization_probe_count": len(public_rows), "benchmark_probe_count": len(positive_public) + len(public_rows), "false_probes_by_user": {u: sum(x["user_id"] == u for x in public_rows) for u in sorted({x["user_id"] for x in public_rows})}, "public_private_id_match": sorted(x["probe_id"] for x in public_rows) == sorted(x["probe_id"] for x in private_rows), "sessions_modified": False, "source_revisions": sorted({x["metadata"]["source_revision"] for x in public_rows})}
    write_json(DATASET / "reports" / "final_benchmark_release_qa_v0_4.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
