#!/usr/bin/env python3
"""Complete and freeze the one-user v0.4 generation pilot at 40 probes.

The long lifeline is immutable here.  GPT-5.5 xhigh authors the missing probe
content end to end from the controlled habit graph and verified evidence.  The
orchestrator only specifies capability/count constraints, validates traceable
evidence, and balances answer positions after semantic judging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from importlib import import_module
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_generator = import_module("04_generate_benchmark")
from api_client import write_json, write_jsonl

DEFAULT_DATASET = _generator.DEFAULT_DATASET
api_json = _generator.api_json
read_jsonl = _generator.read_jsonl


REVISION = "v04_single_user_completion_r1_40_probe_generation_only"
LABELS = ["A", "B", "C", "D"]
POSITIVE_TARGET = 6
NEGATIVE_TARGET = 10


def evidence_for_habit(habit_id: str, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for session in sessions:
        annotation = session.get("memory_annotations", {})
        signals = {
            row.get("habit_id"): row
            for row in annotation.get("verified_habit_signals", [])
            if isinstance(row, dict)
        }
        signal = signals.get(habit_id)
        if not signal:
            continue
        rows.append({
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "episode_id": annotation.get("episode_id"),
            "signal_type": signal.get("signal_type"),
            "evidence_quote": signal.get("evidence_quote"),
            "user_turns": [
                message["content"] for message in session["messages"]
                if message.get("role") == "user"
            ],
        })
    return rows


def choose_existing_positive(
    public: list[dict[str, Any]], keys: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    public_by_id = {row["probe_id"]: row for row in public}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keys:
        grouped[row["habit_id"]].append(row)
    kept_keys = []
    deficits = {}
    for habit_id, rows in sorted(grouped.items()):
        # Keep type diversity first, then harder independently judged items.
        ranked = sorted(
            rows,
            key=lambda row: (
                row.get("independent_gold_judge", {}).get("difficulty") == "hard",
                row.get("probe_type") in {"boundary", "exception", "evidence_disambiguation"},
            ),
            reverse=True,
        )
        chosen, seen_types = [], set()
        for row in ranked:
            if row.get("probe_type") not in seen_types:
                chosen.append(row)
                seen_types.add(row.get("probe_type"))
            if len(chosen) == POSITIVE_TARGET:
                break
        for row in ranked:
            if len(chosen) == POSITIVE_TARGET:
                break
            if row not in chosen:
                chosen.append(row)
        kept_keys.extend(chosen)
        deficits[habit_id] = max(0, POSITIVE_TARGET - len(chosen))
    kept_public = [public_by_id[row["probe_id"]] for row in kept_keys]
    return kept_public, kept_keys, deficits


def validate_candidate(
    row: dict[str, Any], valid_session_ids: set[str], session_episode: dict[str, str]
) -> str | None:
    choices = row.get("choices", [])
    ids = [choice.get("choice_id") for choice in choices if isinstance(choice, dict)]
    if len(choices) != 4 or set(ids) != set(LABELS):
        return "invalid_choices"
    if row.get("gold_choice_id") not in LABELS:
        return "invalid_gold"
    if row.get("closest_distractor_choice_id") not in LABELS or row.get("closest_distractor_choice_id") == row.get("gold_choice_id"):
        return "invalid_closest_distractor"
    cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
    if len(cited) < 3 or not set(cited).issubset(valid_session_ids):
        return "invalid_evidence"
    if len({session_episode[session_id] for session_id in cited}) < 3:
        return "evidence_not_three_episodes"
    lengths = [len(str(choice.get("text", "")).strip()) for choice in choices]
    if min(lengths) < 12 or max(lengths) > 2.7 * min(lengths):
        return "choice_length_giveaway"
    if len(str(row.get("query", "")).strip()) < 50:
        return "query_too_short"
    return None


def normalize_choice_ids(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a reversible choice_a/option_a schema drift without changing text."""
    normalized = copy.deepcopy(row)
    choices = normalized.get("choices", [])
    if not isinstance(choices, list) or len(choices) != 4:
        return normalized
    raw_ids = [str(choice.get("choice_id", "")) for choice in choices if isinstance(choice, dict)]
    aliases = {
        "a": "A", "b": "B", "c": "C", "d": "D",
        "choice_a": "A", "choice_b": "B", "choice_c": "C", "choice_d": "D",
        "option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D",
    }
    mapping = {raw_id: aliases.get(raw_id.lower(), raw_id) for raw_id in raw_ids}
    if set(mapping.values()) != set(LABELS):
        return normalized
    for choice in choices:
        choice["choice_id"] = mapping[str(choice["choice_id"])]
    for field in ["gold_choice_id", "closest_distractor_choice_id"]:
        if str(normalized.get(field, "")) in mapping:
            normalized[field] = mapping[str(normalized[field])]
    return normalized


def generate_positive_additions(
    args: argparse.Namespace,
    profile: dict[str, Any],
    sessions: list[dict[str, Any]],
    existing_public: list[dict[str, Any]],
    existing_keys: list[dict[str, Any]],
    deficits: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    habits = {row["habit_id"]: row for row in profile["habits"]}
    existing_types: dict[str, Counter] = defaultdict(Counter)
    for row in existing_keys:
        existing_types[row["habit_id"]][row["probe_type"]] += 1
    jobs, evidence_by_habit = [], {}
    for habit_id, count in deficits.items():
        if count <= 0:
            continue
        evidence = evidence_for_habit(habit_id, sessions)
        evidence_by_habit[habit_id] = evidence
        jobs.append({
            "habit_id": habit_id,
            "missing_count": count,
            "candidate_count": max(3, count * 3),
            "hidden_habit": habits[habit_id],
            "existing_type_counts": dict(existing_types[habit_id]),
            "verified_evidence": evidence,
        })
    if not jobs:
        return [], [], {"generator": "not_needed"}
    attempt_path = args.dataset / "work" / "single_user_positive_completion_attempt.json"
    cached_attempt = (
        json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt_path.exists() else None
    )
    raw = cached_attempt.get("generator") if cached_attempt else None
    if not isinstance(raw, dict) or not isinstance(raw.get("probes"), list):
        raw = api_json(args, "You directly author rigorous long-history travel-habit benchmark probes. You do not use sentence templates or paraphrase an existing probe.", {
        "task": "Write exactly the requested number of new positive latent-habit probes for each job.",
        "user_id": profile["user_id"],
        "jobs": jobs,
        "requirements": [
            "Return {probes:[{candidate_id,habit_id,probe_type,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
            "Return exactly candidate_count probes per habit_id and unique candidate_id values. The orchestrator will retain only missing_count independently accepted items.",
            "Each probe is a fresh realistic travel decision written end to end, not a slot-filled or paraphrased version of an existing item.",
            "Use the existing type distribution only to add semantic variety; do not follow a repeated wording or option layout.",
            "The current query must leave at least two options genuinely defensible without private history. No option may generically dominate on feasibility, price, time, routing, location, quality, or convenience.",
            "Private longitudinal evidence must uniquely resolve the gold by inducing the scoped, probabilistic habit and then applying its boundary or exception.",
            "Do not reveal the preference, threshold, default, evidence, or words such as memory, habit, usual, normally, prior, or pattern in the query.",
            "All four options must be concrete, plausible, similarly detailed actions with real tradeoffs; include a strong closest distractor.",
            "Use a new transaction and destination rather than copying an evidence event or an existing probe.",
            "Cite at least three supplied evidence session_ids from three distinct episode_ids.",
        ],
        }, 22000)
    rows = raw.get("probes", []) if isinstance(raw, dict) else []
    write_json(args.dataset / "work" / "single_user_positive_generator_raw.json", raw)
    wanted_by_habit = Counter({job["habit_id"]: job["candidate_count"] for job in jobs})
    wanted = sum(wanted_by_habit.values())
    if not isinstance(rows, list) or len(rows) != wanted:
        raise ValueError(f"positive generator returned {len(rows) if isinstance(rows, list) else 0}, expected {wanted}")
    by_habit = Counter(row.get("habit_id") for row in rows if isinstance(row, dict))
    if by_habit != wanted_by_habit:
        raise ValueError(f"positive habit counts mismatch: {by_habit} != {wanted_by_habit}")
    session_episode = {
        row["session_id"]: row.get("memory_annotations", {}).get("episode_id") for row in sessions
    }
    valid_by_habit = {
        habit_id: {row["session_id"] for row in evidence}
        for habit_id, evidence in evidence_by_habit.items()
    }
    structurally_valid_rows = []
    structural_rejections: dict[str, list[str]] = {}
    for row in rows:
        error = validate_candidate(row, valid_by_habit[row["habit_id"]], session_episode)
        if error:
            structural_rejections[str(row.get("candidate_id"))] = [f"structural:{error}"]
            continue
        structurally_valid_rows.append(row)
    judge_payload = []
    blind_payload = []
    for row in structurally_valid_rows:
        cited = set(row["gold_evidence_session_ids"])
        judge_payload.append({
            "candidate_id": row["candidate_id"],
            "history_evidence": [item for item in evidence_by_habit[row["habit_id"]] if item["session_id"] in cited],
            "query": row["query"],
            "choices": row["choices"],
        })
        blind_payload.append({"candidate_id": row["candidate_id"], "query": row["query"], "choices": row["choices"]})

    def history_call() -> dict[str, Any]:
        return api_json(args, "You independently solve travel probes from quoted user history. Hidden habits and proposed answers are unavailable.", {
            "task": "Infer the applicable scoped policy and select the uniquely history-supported action.",
            "probes": judge_payload,
            "requirements": [
                "Return {answers:[{candidate_id,choice_id,evidence_session_ids,rationale,difficulty,ambiguous,closest_distractor_choice_id,closest_distractor_rationale}]}",
                "Use repeated weak evidence rather than a single quote; distinguish soft defaults, boundaries, and one-trip exceptions.",
                "Return UNRESOLVED if the supplied history does not uniquely resolve the decision.",
            ],
        }, 12000)

    def blind_call() -> dict[str, Any]:
        return api_json(args, "You audit travel questions using only the current query and choices. No user history or proposed answer is available.", {
            "task": "Check generic dominance and whether multiple choices remain defensible.",
            "probes": blind_payload,
            "requirements": [
                "Return {answers:[{candidate_id,choice_id,answerable_without_history,generic_best_exists,plausible_choice_ids,rationale,leakage_signals}]}",
                "Return UNRESOLVED when at least two options remain defensible. Do not guess an unstated preference.",
                "A forced A/B/C/D guess is diagnostic, but generic_best_exists=true or answerable_without_history=true means the item structurally fails.",
            ],
        }, 10000)

    history_raw = cached_attempt.get("history_judge") if cached_attempt else None
    blind_raw = cached_attempt.get("query_only_audit") if cached_attempt else None
    if not isinstance(history_raw, dict) or not isinstance(blind_raw, dict):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            history_future = pool.submit(history_call)
            blind_future = pool.submit(blind_call)
            history_raw = history_future.result()
            blind_raw = blind_future.result()
    history = {row.get("candidate_id"): row for row in history_raw.get("answers", []) if isinstance(row, dict)}
    blind = {row.get("candidate_id"): row for row in blind_raw.get("answers", []) if isinstance(row, dict)}
    public, private = [], []
    accepted_counts: Counter = Counter()
    rejection_reasons: dict[str, list[str]] = dict(structural_rejections)
    existing_queries = [row["query"].lower() for row in existing_public]
    difficulty_rank = {"hard": 3, "medium": 2, "moderate": 2, "easy": 1, "low": 1}
    ranked_rows = sorted(
        structurally_valid_rows,
        key=lambda row: (
            difficulty_rank.get(str(history.get(row["candidate_id"], {}).get("difficulty", "")).lower(), 0),
            row.get("probe_type") in {"boundary", "exception", "evidence_disambiguation"},
        ),
        reverse=True,
    )
    for row in ranked_rows:
        candidate_id = row["candidate_id"]
        h, b = history.get(candidate_id), blind.get(candidate_id)
        reasons = []
        if not h or h.get("choice_id") != row["gold_choice_id"] or h.get("ambiguous") is not False:
            reasons.append("history_judge_not_unique_gold")
        if not b or b.get("answerable_without_history") is not False or b.get("generic_best_exists") is not False:
            reasons.append("query_only_structural_dominance")
        if any(SequenceMatcher(None, row["query"].lower(), old).ratio() >= 0.72 for old in existing_queries):
            reasons.append("too_similar_to_existing_query")
        habit_id = row["habit_id"]
        if accepted_counts[habit_id] >= deficits[habit_id]:
            reasons.append("surplus_after_target_filled")
        if reasons:
            rejection_reasons[candidate_id] = reasons
            continue
        probe_id = f"{habit_id}_add{sum(1 for key in private if key['habit_id'] == habit_id):02d}"
        public.append({
            "probe_id": probe_id,
            "user_id": profile["user_id"],
            "split": "test",
            "query": row["query"],
            "choices": row["choices"],
            "visible_history_scope": {"max_session_index": max(item["session_index"] for item in sessions)},
            "metadata": {"dataset_version": "taskmaster_planning_defaults_v0_4", "probe_type": row["probe_type"], "generated_by": args.model, "completion_revision": REVISION},
            "evaluation_contract": {"answer_format": "return one choice_id", "validator_type": "choice_equals"},
        })
        private.append({
            "probe_id": probe_id,
            "user_id": profile["user_id"],
            "target_kind": "latent_habit",
            "memory_target_id": habit_id,
            "habit_id": habit_id,
            "habit_family": habits[habit_id]["family"],
            "probe_type": row["probe_type"],
            "gold_choice_id": row["gold_choice_id"],
            "gold_action": row.get("gold_action"),
            "gold_evidence_session_ids": row["gold_evidence_session_ids"],
            "label_rationale": row.get("label_rationale"),
            "generator_difficulty_rationale": row.get("difficulty_rationale"),
            "generator_closest_distractor_choice_id": row["closest_distractor_choice_id"],
            "hidden_habit_graph": habits[habit_id],
            "query_only_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **b},
            "independent_gold_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **h},
            "label_source": "gpt55_xhigh_e2e_generation_and_independent_history_adjudication",
        })
        accepted_counts[habit_id] += 1
        existing_queries.append(row["query"].lower())
    shortfalls = {
        habit_id: deficits[habit_id] - accepted_counts[habit_id]
        for habit_id in deficits if accepted_counts[habit_id] < deficits[habit_id]
    }
    diagnostics = {
        "generator": raw,
        "history_judge": history_raw,
        "query_only_audit": blind_raw,
        "accepted_counts": dict(accepted_counts),
        "rejection_reasons": rejection_reasons,
        "shortfalls": shortfalls,
    }
    write_json(args.dataset / "work" / "single_user_positive_completion_attempt.json", diagnostics)
    if shortfalls:
        raise ValueError(f"positive candidate pool still has shortfalls: {shortfalls}")
    return public, private, diagnostics


def generate_negative_additions(
    args: argparse.Namespace,
    profile: dict[str, Any],
    sessions: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    existing_public: list[dict[str, Any]],
    existing_keys: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    missing = NEGATIVE_TARGET - len(existing_keys)
    if missing <= 0:
        return [], [], {"generator": "not_needed"}
    used = Counter(row["memory_target_id"] for row in existing_keys)
    ordered = sorted(controls, key=lambda row: (used[row["control_id"]], row["control_id"]))
    selected_controls = [ordered[index % len(ordered)] for index in range(missing)]
    assignments = [
        {
            "assignment_id": f"neg_add_{slot:02d}_c{variant:02d}",
            "slot_id": f"neg_add_{slot:02d}",
            "negative_control": control,
        }
        for slot, control in enumerate(selected_controls)
        for variant in range(3)
    ]
    raw_path = args.dataset / "work" / "single_user_negative_generator_raw.json"
    attempt_path = args.dataset / "work" / "single_user_negative_completion_attempt.json"
    cached_attempt = (
        json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt_path.exists() else None
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
    if not isinstance(raw, dict) or not isinstance(raw.get("probes"), list):
        raw = api_json(args, "You directly author false-personalization travel probes from adjudicated non-habits. You do not use sentence templates or paraphrase existing probes.", {
            "task": "Write one fresh probe for each listed assignment, preserving assignment_id.",
            "assignments": assignments,
            "existing_queries_to_avoid": [row["query"] for row in existing_public],
            "requirements": [
                "Return {probes:[{assignment_id,control_id,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
                "Use choice_id values exactly A, B, C, and D, and distribute intended gold positions across them rather than always using A.",
                "Write a realistic new travel decision end to end. Do not mention history, memory, preferences, patterns, insufficient evidence, or personalization.",
                "The current request, not a purported personal rule, must make the gold the best medium/hard action after a nontrivial tradeoff.",
                "The gold and closest distractor must both satisfy every hard constraint. Neither may dominate on feasibility, room setup, timing, routing, price, location, breakfast, or convenience.",
                "Make the pair differ through at least three conflicting soft objectives so selecting the gold requires weighing them together; never let one option uniquely provide a requirement named in the query.",
                "Avoid easy constructions where one option has every requested benefit or every rival violates an explicit need. Each of the four options needs a credible reason to choose it.",
                "At least two distractors should be tempting if an agent overgeneralizes noisy prior behavior into the supplied false inference.",
                "All choices must be concrete comparable actions with balanced length, meaningful pros and cons, and no obvious worst filler.",
                "Cite at least four supplied evidence session_ids across at least three episodes.",
                "The three candidates sharing a slot_id must use distinct contexts and tradeoff structures; they are a candidate pool, not paraphrases.",
                "Use contexts and option structures unlike the existing probes and unlike other assignments.",
            ],
        }, 30000)
        write_json(raw_path, raw)
    rows = [normalize_choice_ids(row) for row in raw.get("probes", []) if isinstance(row, dict)]
    write_json(args.dataset / "work" / "single_user_negative_generator_normalized.json", {"probes": rows})
    if not isinstance(rows, list) or len(rows) != len(assignments):
        raise ValueError(f"negative generator returned {len(rows) if isinstance(rows, list) else 0}, expected {len(assignments)}")
    control_by_id = {row["control_id"]: row for row in controls}
    session_episode = {row["session_id"]: row.get("memory_annotations", {}).get("episode_id") for row in sessions}
    expected_assignments = {
        assignment["assignment_id"]: assignment["negative_control"]["control_id"]
        for assignment in assignments
    }
    structurally_valid_rows = []
    rejection_reasons: dict[str, list[str]] = {}
    for row in rows:
        if expected_assignments.get(row.get("assignment_id")) != row.get("control_id"):
            rejection_reasons[str(row.get("assignment_id"))] = ["assignment_mismatch"]
            continue
        control = control_by_id[row["control_id"]]
        error = validate_candidate(row, {item["session_id"] for item in control["evidence"]}, session_episode)
        if error:
            rejection_reasons[row["assignment_id"]] = [f"structural:{error}"]
            continue
        structurally_valid_rows.append(row)
    query_items = [{"assignment_id": row["assignment_id"], "query": row["query"], "choices": row["choices"]} for row in structurally_valid_rows]
    history_items = []
    for row in structurally_valid_rows:
        control = control_by_id[row["control_id"]]
        history_items.append({
            "assignment_id": row["assignment_id"],
            "history_evidence": control["evidence"],
            "query": row["query"],
            "choices": row["choices"],
        })

    def query_call() -> dict[str, Any]:
        return api_json(args, "You solve travel multiple-choice decisions only from the current request. No personal history or proposed answer is available.", {
            "task": "Choose the best current-trip action after weighing all explicit constraints.",
            "probes": query_items,
            "requirements": ["Return {answers:[{assignment_id,choice_id,rationale,difficulty,ambiguous,closest_distractor_choice_id,closest_distractor_rationale}]}", "Do not invent personal preferences."],
        }, 10000)

    def history_call() -> dict[str, Any]:
        return api_json(args, "You independently solve false-personalization probes from current requests plus noisy history. Proposed answers and negative-control labels are unavailable.", {
            "task": "Choose the best action and decide whether history supports a stable preference on the tested dimension.",
            "probes": history_items,
            "requirements": [
                "Return {answers:[{assignment_id,choice_id,evidence_session_ids,rationale,difficulty,ambiguous,closest_distractor_choice_id,closest_distractor_rationale,stable_preference_supported}]}",
                "Do not turn mixed, trip-specific, or assistant-led observations into a durable personal rule.",
            ],
        }, 12000)

    query_raw = cached_attempt.get("query_judge") if cached_attempt else None
    history_raw = cached_attempt.get("history_judge") if cached_attempt else None
    if not isinstance(query_raw, dict) or not isinstance(history_raw, dict):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            query_future = pool.submit(query_call)
            history_future = pool.submit(history_call)
            query_raw = query_future.result()
            history_raw = history_future.result()
    query = {row.get("assignment_id"): row for row in query_raw.get("answers", []) if isinstance(row, dict)}
    history = {row.get("assignment_id"): row for row in history_raw.get("answers", []) if isinstance(row, dict)}
    public, private = [], []
    all_old_queries = [row["query"].lower() for row in existing_public]
    difficulty_rank = {"hard": 3, "medium": 2, "moderate": 2, "easy": 1, "low": 1}
    ranked_rows = sorted(
        structurally_valid_rows,
        key=lambda row: difficulty_rank.get(
            str(query.get(row["assignment_id"], {}).get("difficulty", "")).lower(), 0
        ),
        reverse=True,
    )
    selected_by_slot: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for row in ranked_rows:
        assignment_id = row["assignment_id"]
        q, h = query.get(assignment_id), history.get(assignment_id)
        reasons = []
        if (
            not q
            or q.get("choice_id") != row["gold_choice_id"]
            or str(q.get("difficulty", "")).lower() not in {"easy", "low", "medium", "moderate", "hard"}
            or q.get("ambiguous") is not False
        ):
            reasons.append("current_query_not_unique_gold")
        if not h or h.get("choice_id") != row["gold_choice_id"] or h.get("stable_preference_supported") is not False or h.get("ambiguous") is not False:
            reasons.append("history_judge_failed_nonpersonalization")
        if any(SequenceMatcher(None, row["query"].lower(), old).ratio() >= 0.72 for old in all_old_queries):
            reasons.append("too_similar_to_existing_query")
        slot_id = assignment_id.rsplit("_c", 1)[0]
        if slot_id in selected_by_slot:
            reasons.append("surplus_after_slot_filled")
        if reasons:
            rejection_reasons[assignment_id] = reasons
            continue
        selected_by_slot[slot_id] = (row, q, h)
        all_old_queries.append(row["query"].lower())
    expected_slots = {f"neg_add_{slot:02d}" for slot in range(missing)}
    shortfalls = sorted(expected_slots - set(selected_by_slot))
    diagnostics = {
        "generator": raw,
        "query_judge": query_raw,
        "history_judge": history_raw,
        "selected_assignment_ids": {
            slot_id: item[0]["assignment_id"] for slot_id, item in selected_by_slot.items()
        },
        "rejection_reasons": rejection_reasons,
        "shortfalls": shortfalls,
    }
    write_json(args.dataset / "work" / "single_user_negative_completion_attempt.json", diagnostics)
    if shortfalls:
        raise ValueError(f"negative candidate pool still has shortfalls: {shortfalls}")
    for offset, slot_id in enumerate(sorted(selected_by_slot)):
        row, q, h = selected_by_slot[slot_id]
        index = len(existing_keys) + offset
        control = control_by_id[row["control_id"]]
        probe_id = f"{profile['user_id']}_fp_p{index:02d}"
        public.append({
            "probe_id": probe_id, "user_id": profile["user_id"], "split": "test",
            "query": row["query"], "choices": row["choices"],
            "visible_history_scope": {"max_session_index": max(item["session_index"] for item in sessions)},
            "metadata": {"dataset_version": "taskmaster_planning_defaults_v0_4", "probe_type": "false_personalization", "capability": "false_personalization", "control_id": control["control_id"], "generated_by": args.model, "completion_revision": REVISION},
            "evaluation_contract": {"answer_format": "return one choice_id", "validator_type": "choice_equals"},
        })
        private.append({
            "probe_id": probe_id, "user_id": profile["user_id"], "target_kind": "negative_control",
            "memory_target_id": control["control_id"], "habit_id": None, "habit_family": control["family"],
            "probe_type": "false_personalization", "gold_choice_id": row["gold_choice_id"], "gold_action": row.get("gold_action"),
            "gold_evidence_session_ids": row["gold_evidence_session_ids"], "label_rationale": row.get("label_rationale"),
            "generator_difficulty_rationale": row.get("difficulty_rationale"), "generator_closest_distractor_choice_id": row["closest_distractor_choice_id"],
            "query_only_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **q},
            "independent_gold_judge": {"model": args.model, "reasoning_effort": args.reasoning_effort, **h},
            "negative_control": control, "label_source": "gpt55_xhigh_e2e_false_personalization_and_independent_dual_adjudication",
        })
    return public, private, diagnostics


def remap_choice_fields(key: dict[str, Any], mapping: dict[str, str]) -> None:
    key["gold_choice_id"] = mapping[key["gold_choice_id"]]
    for field in ["generator_closest_distractor_choice_id"]:
        if key.get(field) in mapping:
            key[field] = mapping[key[field]]
    for judge_name in ["query_only_judge", "independent_gold_judge"]:
        judge = key.get(judge_name, {})
        for field in ["choice_id", "closest_distractor_choice_id"]:
            if judge.get(field) in mapping:
                judge[field] = mapping[judge[field]]
        if isinstance(judge.get("plausible_choice_ids"), list):
            judge["plausible_choice_ids"] = [mapping.get(item, item) for item in judge["plausible_choice_ids"]]


def balance_positions(public: list[dict[str, Any]], keys: list[dict[str, Any]]) -> None:
    public_by_id = {row["probe_id"]: row for row in public}
    for index, key in enumerate(sorted(keys, key=lambda row: row["probe_id"])):
        probe = public_by_id[key["probe_id"]]
        old_choices = copy.deepcopy(probe["choices"])
        gold_text = next(row["text"] for row in old_choices if row["choice_id"] == key["gold_choice_id"])
        distractors = [row["text"] for row in old_choices if row["choice_id"] != key["gold_choice_id"]]
        target = LABELS[index % 4]
        texts = list(distractors)
        texts.insert(LABELS.index(target), gold_text)
        probe["choices"] = [{"choice_id": label, "text": text} for label, text in zip(LABELS, texts)]
        new_by_text = {row["text"]: row["choice_id"] for row in probe["choices"]}
        mapping = {row["choice_id"]: new_by_text[row["text"]] for row in old_choices}
        remap_choice_fields(key, mapping)
        key["choice_position_balancing"] = "round_robin_after_all_semantic_generation_and_judging"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--transport", choices=["curl", "curl_stream", "urllib"], default="curl_stream")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")
    root = args.dataset
    profiles = read_jsonl(root / "private" / "user_dossiers.jsonl")
    sessions = read_jsonl(root / "private" / "sessions_with_annotations.jsonl")
    if len(profiles) != 1 or not sessions:
        raise SystemExit("completion expects the frozen one-user v0.4 pilot")
    profile = profiles[0]
    base_pos_public = read_jsonl(root / "public" / "habit_probes.jsonl")
    base_pos_keys = read_jsonl(root / "private" / "habit_probe_key.jsonl")
    neg_public = read_jsonl(root / "public" / "false_personalization_probes.jsonl")
    neg_keys = read_jsonl(root / "private" / "false_personalization_probe_key.jsonl")
    controls = read_jsonl(root / "private" / "false_personalization_controls.jsonl")
    pos_public, pos_keys, deficits = choose_existing_positive(base_pos_public, base_pos_keys)
    positive_add_public, positive_add_keys, positive_raw = generate_positive_additions(
        args, profile, sessions, pos_public, pos_keys, deficits
    )
    pos_public += positive_add_public
    pos_keys += positive_add_keys
    negative_add_public, negative_add_keys, negative_raw = generate_negative_additions(
        args, profile, sessions, controls, neg_public, neg_keys
    )
    neg_public += negative_add_public
    neg_keys += negative_add_keys
    public, keys = pos_public + neg_public, pos_keys + neg_keys
    habit_counts = Counter(row.get("habit_id") for row in keys if row.get("target_kind") != "negative_control")
    expected_habits = {row["habit_id"] for row in profile["habits"] if row.get("testable", True)}
    issues = []
    if len(public) != 40 or len(keys) != 40:
        issues.append(f"expected_40_got_{len(public)}_{len(keys)}")
    if set(habit_counts) != expected_habits or any(habit_counts[habit_id] != POSITIVE_TARGET for habit_id in expected_habits):
        issues.append(f"positive_habit_counts:{dict(habit_counts)}")
    if sum(row.get("target_kind") == "negative_control" for row in keys) != NEGATIVE_TARGET:
        issues.append("negative_count_not_10")
    if len({row["probe_id"] for row in public}) != len(public):
        issues.append("duplicate_probe_ids")
    if set(row["probe_id"] for row in public) != set(row["probe_id"] for row in keys):
        issues.append("public_private_mismatch")
    queries = [row["query"].lower() for row in public]
    near_duplicates = []
    for i, left in enumerate(queries):
        for j in range(i + 1, len(queries)):
            score = SequenceMatcher(None, left, queries[j]).ratio()
            if score >= 0.76:
                near_duplicates.append({"left": public[i]["probe_id"], "right": public[j]["probe_id"], "ratio": score})
    if near_duplicates:
        issues.append(f"near_duplicate_count:{len(near_duplicates)}")
    if issues:
        write_json(root / "work" / "single_user_completion_failed.json", {
            "revision": REVISION, "issues": issues, "positive_raw": positive_raw,
            "negative_raw": negative_raw, "near_duplicates": near_duplicates,
        })
        raise ValueError(f"single-user completion failed before release writes: {issues}")
    balance_positions(public, keys)
    public.sort(key=lambda row: row["probe_id"])
    keys.sort(key=lambda row: row["probe_id"])
    write_jsonl(root / "public" / "probes.jsonl", public)
    write_jsonl(root / "private" / "probe_key.jsonl", keys)
    write_jsonl(root / "public" / "habit_probes_final.jsonl", [row for row in public if row["metadata"].get("probe_type") != "false_personalization"])
    write_jsonl(root / "private" / "habit_probe_key_final.jsonl", [row for row in keys if row.get("target_kind") != "negative_control"])
    write_jsonl(root / "public" / "false_personalization_probes_final.jsonl", [row for row in public if row["metadata"].get("probe_type") == "false_personalization"])
    write_jsonl(root / "private" / "false_personalization_probe_key_final.jsonl", [row for row in keys if row.get("target_kind") == "negative_control"])
    report = {
        "status": "pass", "revision": REVISION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_scope": "probe completion only; frozen GPT-authored lifeline unchanged",
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "user_count": 1, "session_count": len(sessions),
        "episode_count": len({row.get("memory_annotations", {}).get("episode_id") for row in sessions}),
        "probe_count": len(public), "positive_probe_count": 30, "false_personalization_probe_count": 10,
        "habit_counts": dict(habit_counts),
        "negative_control_counts": dict(Counter(row.get("memory_target_id") for row in keys if row.get("target_kind") == "negative_control")),
        "probe_type_counts": dict(Counter(row.get("probe_type") for row in keys)),
        "gold_position_counts": dict(Counter(row["gold_choice_id"] for row in keys)),
        "near_duplicate_count": 0,
        "methodology_note": "No candidate was selected or rewritten because one forced-choice GPT baseline happened to hit or miss its gold answer.",
    }
    write_json(root / "reports" / "single_user_generation_release.json", report)
    write_json(root / "work" / "single_user_completion_raw.json", {"positive": positive_raw, "negative": negative_raw})
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
