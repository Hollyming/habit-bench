#!/usr/bin/env python3
"""Replace weak pilot probes with matched positive/negative calibration probes.

The resulting 40-item pilot keeps 25 positive action probes and five existing
false-memory-interference controls.  It replaces one positive item per habit
and the five easy negative controls with ten surface-matched calibration
items: five histories support a reusable default and five do not.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
REVISION = "v04_matched_preference_calibration_r1"
LABELS = {"A", "B", "C", "D"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def call(args: argparse.Namespace, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    return post_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": system + " Return strict JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        transport=args.transport,
        reasoning_effort=args.reasoning_effort,
    )["json"]


def habit_evidence(habit_id: str, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for session in sessions:
        annotations = session.get("memory_annotations", {})
        signal = next(
            (
                item for item in annotations.get("verified_habit_signals", [])
                if item.get("habit_id") == habit_id
            ),
            None,
        )
        if not signal:
            continue
        rows.append({
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "episode_id": annotations.get("episode_id"),
            "signal_type": signal.get("signal_type"),
            "evidence_quote": signal.get("evidence_quote"),
        })
    return rows


def candidate_error(row: Any, assignment: dict[str, Any]) -> str | None:
    if not isinstance(row, dict) or row.get("assignment_id") != assignment["assignment_id"]:
        return "assignment_mismatch"
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return "invalid_choices"
    ids = {item.get("choice_id") for item in choices if isinstance(item, dict)}
    if ids != LABELS or row.get("gold_choice_id") not in LABELS:
        return "invalid_choice_ids"
    if row.get("closest_distractor_choice_id") not in LABELS or row.get("closest_distractor_choice_id") == row.get("gold_choice_id"):
        return "invalid_closest_distractor"
    texts = [str(item.get("text", "")).strip() for item in choices]
    if min(map(len, texts)) < 25 or max(map(len, texts)) > 2.4 * min(map(len, texts)):
        return "choice_length_leakage"
    if len(str(row.get("query", "")).strip()) < 60:
        return "query_too_short"
    valid = {item["session_id"] for item in assignment["history_evidence"]}
    cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
    if len(cited) < 3 or not set(cited).issubset(valid):
        return "invalid_evidence"
    episodes = {
        item.get("episode_id") for item in assignment["history_evidence"]
        if item["session_id"] in cited
    }
    if len(episodes - {None}) < 3:
        return "evidence_not_three_episodes"
    return None


def choose_replaced_positives(keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_habit: dict[str, list[dict[str, Any]]] = {}
    for row in keys:
        if row.get("target_kind") == "negative_control":
            continue
        by_habit.setdefault(row["habit_id"], []).append(row)
    chosen = []
    for habit_id, rows in sorted(by_habit.items()):
        # Preserve boundary/exception/hard items; replace the least diagnostic one.
        ranked = sorted(rows, key=lambda row: (
            row.get("independent_gold_judge", {}).get("difficulty") not in {"easy", "low"},
            row.get("probe_type") in {"boundary", "exception", "conflict_resolution", "evidence_disambiguation"},
        ))
        chosen.append(ranked[0])
    if len(chosen) != 5:
        raise ValueError(f"expected five testable habits, got {len(chosen)}")
    return chosen


def build_assignments(
    profile: dict[str, Any], sessions: list[dict[str, Any]], keys: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    habits = {row["habit_id"]: row for row in profile["habits"]}
    replaced_positive = choose_replaced_positives(keys)
    easy_negatives = [
        row for row in keys
        if row.get("target_kind") == "negative_control"
        and (
            str(row.get("query_only_judge", {}).get("difficulty", "")).lower() in {"easy", "low"}
            or str(row.get("independent_gold_judge", {}).get("difficulty", "")).lower() in {"easy", "low"}
        )
    ]
    if len(easy_negatives) != 5:
        raise ValueError(f"expected five easy negative controls, got {len(easy_negatives)}")
    control_by_id = {row["control_id"]: row for row in controls}
    assignments = []
    for old in replaced_positive:
        habit = habits[old["habit_id"]]
        assignments.append({
            "assignment_id": f"positive::{old['probe_id']}",
            "polarity": "supported_default",
            "replacement_probe_id": old["probe_id"],
            "target_id": old["habit_id"],
            "target_description": habit,
            "history_evidence": habit_evidence(old["habit_id"], sessions),
        })
    for old in easy_negatives:
        control = control_by_id[old["memory_target_id"]]
        assignments.append({
            "assignment_id": f"negative::{old['probe_id']}",
            "polarity": "unsupported_default",
            "replacement_probe_id": old["probe_id"],
            "target_id": control["control_id"],
            "target_description": control,
            "history_evidence": control["evidence"],
        })
    return assignments, replaced_positive, easy_negatives


def generate(args: argparse.Namespace, assignments: list[dict[str, Any]], existing_queries: list[str]) -> dict[str, Any]:
    cache_payload = {
        "revision": REVISION, "model": args.model, "reasoning_effort": args.reasoning_effort,
        "assignments": assignments, "existing_queries": existing_queries,
    }
    fingerprint = hashlib.sha256(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache = args.dataset / "work" / "matched_calibration" / f"generated_{fingerprint}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = call(args, "You author rigorous matched habit-calibration benchmark questions.", {
        "task": "Write three distinct candidate probes for each assignment. Positive and negative assignments must have indistinguishable surface form.",
        "assignments": assignments,
        "existing_queries_to_avoid": existing_queries,
        "requirements": [
            "Return {probes:[{assignment_id,variant_id,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
            "Each query is a realistic new travel-planning situation asking what standing guidance, default, or memory policy the assistant should carry into comparable future decisions.",
            "Every item must offer four plausible memory policies: concrete defaults or scoped rules plus one calibrated keep-unset/ask-again/current-context-only policy. Vary their wording and position naturally.",
            "For supported_default assignments, a scoped reusable habit is gold and the keep-unset policy is a strong distractor. For unsupported_default assignments, the keep-unset or context-only policy is gold and at least two tempting concrete defaults reflect noisy history.",
            "Never reveal assignment polarity, hidden labels, habit IDs, evidence, or phrases such as benchmark, positive probe, negative probe, false personalization, ground truth, or insufficient evidence.",
            "The query and options alone must leave at least two policies defensible. Generic caution must not uniquely identify the answer; only longitudinal history may resolve whether a reusable rule exists.",
            "Use the same range of wording, detail, difficulty, and option structure for both polarities. Do not make the keep-unset choice longer, more qualified, more ethical-sounding, or more obviously cautious than the other choices.",
            "Policies must be scoped and operational, not vague claims. Use a fresh destination/context and do not copy a prior transaction.",
            "Cite at least three supplied evidence session IDs from three distinct episodes.",
            "Gold and closest distractor must be medium/hard to distinguish with history and comparable in length and specificity.",
        ],
    }, 30000)
    write_json(cache, raw)
    return raw


def judge(args: argparse.Namespace, assignments: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment_by_id = {row["assignment_id"]: row for row in assignments}
    blind_items = [{
        "candidate_id": f"{row['assignment_id']}::{row['variant_id']}",
        "query": row["query"], "choices": row["choices"],
    } for row in rows]
    history_items = [{
        "candidate_id": f"{row['assignment_id']}::{row['variant_id']}",
        "history_evidence": assignment_by_id[row["assignment_id"]]["history_evidence"],
        "query": row["query"], "choices": row["choices"],
    } for row in rows]
    blind = call(args, "You audit preference-calibration questions without access to user history or proposed labels.", {
        "task": "Determine whether query and choices alone reveal a unique memory policy.",
        "probes": blind_items,
        "requirements": [
            "Return {answers:[{candidate_id,choice_id,answerable_without_history,generic_best_exists,plausible_choice_ids,rationale,leakage_signals}]}",
            "Return choice_id=UNRESOLVED, answerable_without_history=false, and generic_best_exists=false only when at least two policies remain defensible.",
            "Treat a conspicuously cautious, longer, or specially worded keep-unset choice as leakage and mark answerable_without_history=true.",
        ],
    }, 14000)
    history = call(args, "You solve preference-calibration questions from quoted longitudinal user evidence. Proposed answers and assignment polarity are hidden.", {
        "task": "Infer whether a scoped reusable default is supported, then select the unique policy.",
        "probes": history_items,
        "requirements": [
            "Return {answers:[{candidate_id,choice_id,evidence_session_ids,rationale,difficulty,ambiguous,closest_distractor_choice_id,closest_distractor_rationale,stable_preference_supported}]}",
            "Use repeated evidence across episodes. Mixed, trip-specific, assistant-led, or contradictory observations do not establish a stable default.",
            "Return ambiguous=true if history cannot uniquely distinguish the best policy. Difficulty must reflect history-aware distinction, not prose complexity.",
        ],
    }, 18000)
    return blind, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")

    release = args.dataset / "release_single_user_pilot"
    public_path = release / "public" / "probes.jsonl"
    key_path = release / "private" / "probe_key.jsonl"
    public = read_jsonl(public_path)
    keys = read_jsonl(key_path)
    profile = read_jsonl(release / "private" / "user_dossiers.jsonl")[0]
    sessions = read_jsonl(release / "private" / "sessions_with_annotations.jsonl")
    controls = read_jsonl(release / "private" / "false_personalization_controls.jsonl")
    if len(public) != 40 or len(keys) != 40:
        raise SystemExit("expected frozen 40-probe pilot")
    assignments, replaced_positive, replaced_negative = build_assignments(profile, sessions, keys, controls)
    raw = generate(args, assignments, [row["query"] for row in public])
    rows = raw.get("probes", []) if isinstance(raw, dict) else []
    expected = Counter({row["assignment_id"]: 3 for row in assignments})
    actual = Counter(row.get("assignment_id") for row in rows if isinstance(row, dict))
    if actual != expected:
        raise ValueError(f"generator assignment counts mismatch: {actual} != {expected}")
    assignment_by_id = {row["assignment_id"]: row for row in assignments}
    structurally_valid = [
        row for row in rows if candidate_error(row, assignment_by_id.get(row.get("assignment_id"), {})) is None
    ]
    blind_raw, history_raw = judge(args, assignments, structurally_valid)
    blind = {row.get("candidate_id"): row for row in blind_raw.get("answers", [])}
    history = {row.get("candidate_id"): row for row in history_raw.get("answers", [])}
    selected: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    rejections = {}
    for row in structurally_valid:
        assignment = assignment_by_id[row["assignment_id"]]
        candidate_id = f"{row['assignment_id']}::{row['variant_id']}"
        b, h = blind.get(candidate_id), history.get(candidate_id)
        reasons = []
        if not b or b.get("choice_id") != "UNRESOLVED" or b.get("answerable_without_history") is not False or b.get("generic_best_exists") is not False or len(b.get("plausible_choice_ids", [])) < 2:
            reasons.append("query_only_not_unresolved")
        expected_stable = assignment["polarity"] == "supported_default"
        if not h or h.get("choice_id") != row["gold_choice_id"] or h.get("ambiguous") is not False or h.get("stable_preference_supported") is not expected_stable or str(h.get("difficulty", "")).lower() not in {"medium", "hard"}:
            reasons.append("history_judge_failed")
        if reasons:
            rejections[candidate_id] = reasons
            continue
        selected.setdefault(row["assignment_id"], (row, b, h))
    missing = sorted(set(assignment_by_id) - set(selected))
    report_path = args.dataset / "reports" / "matched_calibration_rebuild.json"
    if missing:
        write_json(report_path, {"status": "failed", "revision": REVISION, "missing": missing, "rejections": rejections})
        raise ValueError(f"no accepted matched calibration candidate for: {missing}")

    public_by_id = {row["probe_id"]: copy.deepcopy(row) for row in public}
    key_by_id = {row["probe_id"]: copy.deepcopy(row) for row in keys}
    for assignment_id, (row, b, h) in selected.items():
        assignment = assignment_by_id[assignment_id]
        probe_id = assignment["replacement_probe_id"]
        public_by_id[probe_id].update({"query": row["query"], "choices": row["choices"]})
        public_by_id[probe_id]["metadata"] = {
            "dataset_version": "taskmaster_planning_defaults_v0_4",
            "probe_type": "preference_calibration",
            "generated_by": args.model,
            "calibration_revision": REVISION,
        }
        key = key_by_id[probe_id]
        key.update({
            "probe_type": "preference_calibration",
            "gold_choice_id": row["gold_choice_id"],
            "gold_action": row.get("gold_action"),
            "gold_evidence_session_ids": row["gold_evidence_session_ids"],
            "label_rationale": row.get("label_rationale"),
            "generator_difficulty_rationale": row.get("difficulty_rationale"),
            "generator_closest_distractor_choice_id": row["closest_distractor_choice_id"],
            "query_only_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **b},
            "independent_gold_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **h},
            "label_source": "gpt55_xhigh_matched_preference_calibration",
            "calibration_polarity": assignment["polarity"],
            "calibration_revision": REVISION,
        })

    final_public = sorted(public_by_id.values(), key=lambda row: row["probe_id"])
    final_keys = sorted(key_by_id.values(), key=lambda row: row["probe_id"])
    backup = args.dataset / "work" / "archive" / REVISION
    backup.mkdir(parents=True, exist_ok=True)
    for path in [public_path, key_path]:
        target = backup / path.name
        if not target.exists():
            shutil.copy2(path, target)
    write_jsonl(public_path, final_public)
    write_jsonl(key_path, final_keys)

    # Keep canonical root files synchronized with the frozen pilot release.
    write_jsonl(args.dataset / "public" / "probes.jsonl", final_public)
    write_jsonl(args.dataset / "private" / "probe_key.jsonl", final_keys)
    write_jsonl(args.dataset / "public" / "habit_probes_final.jsonl", [
        row for row in final_public if key_by_id[row["probe_id"]].get("target_kind") != "negative_control"
    ])
    write_jsonl(args.dataset / "private" / "habit_probe_key_final.jsonl", [
        row for row in final_keys if row.get("target_kind") != "negative_control"
    ])
    write_jsonl(args.dataset / "public" / "false_personalization_probes_final.jsonl", [
        row for row in final_public if key_by_id[row["probe_id"]].get("target_kind") == "negative_control"
    ])
    write_jsonl(args.dataset / "private" / "false_personalization_probe_key_final.jsonl", [
        row for row in final_keys if row.get("target_kind") == "negative_control"
    ])
    report = {
        "status": "pass", "revision": REVISION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_count": 40,
        "positive_action_count": 25,
        "positive_calibration_count": 5,
        "negative_calibration_count": 5,
        "negative_interference_count": 5,
        "replaced_positive_probe_ids": [row["probe_id"] for row in replaced_positive],
        "replaced_negative_probe_ids": [row["probe_id"] for row in replaced_negative],
        "query_only_unresolved_calibration_count": 10,
        "calibration_gold_positions": dict(Counter(
            key_by_id[assignment["replacement_probe_id"]]["gold_choice_id"] for assignment in assignments
        )),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
