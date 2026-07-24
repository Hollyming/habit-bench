#!/usr/bin/env python3
"""Stage-aware release audit for Taskmaster planning_defaults v0.4."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
DEFAULT_TOKENIZER = Path("/data1/public/hf/Qwen/Qwen3-8B")


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalized_session(row: dict[str, Any]) -> str:
    text = " ".join(message["content"] for message in row.get("messages", []))
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--tokenizer-python", type=Path, default=Path("/home/xqwang/miniconda3/envs/grpo/bin/python"))
    args = parser.parse_args()
    errors, warnings = [], []
    bundles = rows(args.dataset / "sources" / "habit_evidence_bundles.jsonl")
    if not bundles: errors.append("missing source bundles")
    for bundle in bundles:
        examples = bundle.get("source_examples", [])
        if len({x.get("conversation_id") for x in examples}) < 3 or len({x.get("instruction_id") for x in examples}) < 3:
            errors.append(f"non-independent bundle {bundle.get('bundle_id')}")

    summary_path = args.dataset / "reports" / "habit_induction_summary.json"
    if not summary_path.exists(): errors.append("missing habit induction summary")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("model") != "gpt-5.5" or summary.get("reasoning_effort") != "xhigh" or summary.get("processed") != len(bundles):
            errors.append("habit induction is not complete gpt-5.5/xhigh")

    curated = rows(args.dataset / "private" / "curated_habit_pool.jsonl")
    profiles = rows(args.dataset / "private" / "user_dossiers.jsonl")
    arcs = rows(args.dataset / "private" / "chronological_arc.jsonl")
    versions = rows(args.dataset / "private" / "habit_version_history.jsonl")
    sessions = rows(args.dataset / "private" / "sessions_with_annotations.jsonl")
    public_sessions = rows(args.dataset / "public" / "lifelines.jsonl")
    probes = rows(args.dataset / "public" / "probes.jsonl")
    keys = rows(args.dataset / "private" / "probe_key.jsonl")
    if not curated: warnings.append("semantic review/curated pool not complete")
    if not profiles: warnings.append("user dossiers not complete")
    if not arcs: warnings.append("chronological arcs not generated")
    if not sessions: warnings.append("longitudinal sessions not generated")
    if not probes: warnings.append("held-out probes not generated")

    profile_by_user = {row["user_id"]: row for row in profiles}
    curated_ids = {row.get("habit_instance_id") for row in curated}
    target_counts = []
    all_habit_ids = set()
    for profile in profiles:
        habits = profile.get("habits", [])
        if not 5 <= len(habits) <= 8: errors.append(f"unnatural habit count {profile['user_id']}")
        if sum(bool(habit.get("testable")) for habit in habits) < 5:
            errors.append(f"fewer than five testable habits {profile['user_id']}")
        families = defaultdict(set)
        for habit in habits:
            all_habit_ids.add(habit["habit_id"])
            if habit.get("source_habit_instance_id") not in curated_ids:
                errors.append(f"dossier cites non-curated habit {profile['user_id']}:{habit.get('source_habit_instance_id')}")
            families[habit["family"]].add(habit.get("scope_key"))
            if len(families[habit["family"]]) > 2: errors.append(f"too many same-family habits {profile['user_id']}")
        if not 4 <= len(families) <= 7: errors.append(f"unnatural family count {profile['user_id']}")
        target = int(profile.get("longitudinal_plan", {}).get("target_sessions", 0))
        target_counts.append(target)
        if not 100 <= target <= 150: errors.append(f"invalid target sessions {profile['user_id']}")
        target_characters = int(profile.get("longitudinal_plan", {}).get("target_history_characters", 0))
        if not 220_000 <= target_characters <= 400_000: errors.append(f"invalid history character target {profile['user_id']}")
    if len(target_counts) > 1 and len(set(target_counts)) == 1: errors.append("all users have the same history length")

    arcs_by_user = defaultdict(list)
    for event in arcs: arcs_by_user[event["user_id"]].append(event)
    for user_id, profile in profile_by_user.items():
        user_arcs = sorted(arcs_by_user[user_id], key=lambda row: row["session_index"])
        target = profile["longitudinal_plan"]["target_sessions"]
        if arcs and [row["session_index"] for row in user_arcs] != list(range(target)):
            errors.append(f"arc indices/count mismatch {user_id}")
        gaps, episodes, revision_ids = [], set(), set()
        for event in user_arcs:
            gaps.append(event.get("days_after_previous")); episodes.add(event.get("episode_id"))
            if not event.get("event_first_provenance", {}).get("dossier_conditioned_without_signal_labels"):
                errors.append(f"arc does not follow dossier-conditioned event-first route {user_id}:{event['session_index']}")
            if any(key in event for key in ["habit_signals", "linked_habit_ids", "state_updates", "habit_mapping_provenance"]):
                errors.append(f"arc contains forbidden pre-dialogue habit labels {user_id}:{event['session_index']}")
            if not event.get("grounding_source_ids"):
                errors.append(f"arc lacks Taskmaster grounding {user_id}:{event['session_index']}")
            signal_ids = set()
            for signal in event.get("habit_signals", []):
                if signal.get("habit_id") not in all_habit_ids or signal.get("habit_id") in signal_ids: errors.append(f"invalid per-habit signal {user_id}:{event['session_index']}")
                signal_ids.add(signal.get("habit_id"))
                if signal.get("signal_type") == "revision": revision_ids.add((event["session_index"], signal["habit_id"]))
            update_ids = {(event["session_index"], update.get("habit_id")) for update in event.get("state_updates", [])}
            if {(event["session_index"], hid) for idx, hid in revision_ids if idx == event["session_index"]} != update_ids:
                errors.append(f"revision/update mismatch {user_id}:{event['session_index']}")
        if user_arcs and (len(set(gaps)) < 4 or len(episodes) < 4): errors.append(f"mechanical episode timeline {user_id}")

    sessions_by_user = defaultdict(list)
    session_by_id = {}
    normalized_seen = {}
    for session in sessions:
        sessions_by_user[session["user_id"]].append(session); session_by_id[session["session_id"]] = session
        messages = session.get("messages", [])
        if len(messages) % 2 or not messages or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
            errors.append(f"malformed alternation {session.get('session_id')}")
        annotation = session.get("memory_annotations", {})
        if "verified_habit_signals" not in annotation or "independent_verification" not in annotation:
            errors.append(f"unverified session {session.get('session_id')}")
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user").lower()
        for signal in annotation.get("verified_habit_signals", []):
            if str(signal.get("evidence_quote", "")).lower() not in user_text: errors.append(f"untraceable evidence quote {session.get('session_id')}")
        norm = normalized_session(session)
        if norm in normalized_seen: errors.append(f"exact duplicate sessions {normalized_seen[norm]} {session.get('session_id')}")
        normalized_seen[norm] = session.get("session_id")
    if sessions and len(public_sessions) != len(sessions): errors.append("public/private session count mismatch")

    token_counts = {}
    for user_id, user_sessions in sessions_by_user.items():
        ordered = sorted(user_sessions, key=lambda row: row["session_index"])
        if len(ordered) != profile_by_user[user_id]["longitudinal_plan"]["target_sessions"]: errors.append(f"session count mismatch {user_id}")
        actual_characters = sum(len(message["content"]) for row in ordered for message in row["messages"])
        if actual_characters < profile_by_user[user_id]["longitudinal_plan"]["target_history_characters"]:
            errors.append(f"history below dossier character target {user_id}:{actual_characters}")
        stamps = [datetime.fromisoformat(row["timestamp"]) for row in ordered]
        if any(b <= a for a, b in zip(stamps, stamps[1:])): errors.append(f"timestamps not strictly increasing {user_id}")
        if sessions:
            helper = Path(__file__).with_name("count_qwen_tokens.py")
            texts = ["\n".join(m["content"] for m in row["messages"]) for row in ordered]
            proc = subprocess.run(
                [str(args.tokenizer_python), str(helper), str(args.tokenizer)],
                input=json.dumps({"texts": texts}, ensure_ascii=False), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if proc.returncode != 0:
                errors.append(f"cannot run exact Qwen tokenizer for {user_id}: {proc.stderr[-500:]}")
            else:
                token_count = int(json.loads(proc.stdout)["total"])
                token_counts[user_id] = token_count
                if token_count < 51_200: errors.append(f"history below 1.25x Qwen context {user_id}:{token_count}")
        sets = [(row["session_id"], shingles(normalized_session(row))) for row in ordered]
        for i, (left_id, left) in enumerate(sets):
            if len(left) < 8: continue
            for right_id, right in sets[i + 1:]:
                union = len(left | right)
                if union and len(left & right) / union >= 0.82: errors.append(f"semantic near duplicate {left_id} {right_id}")

    key_by_id = {row["probe_id"]: row for row in keys}
    if probes and len(probes) != len(keys): errors.append("public/private probe count mismatch")
    gold_positions = Counter()
    gold_sequence = []
    probe_queries_by_user = defaultdict(list)
    for probe in probes:
        key = key_by_id.get(probe["probe_id"])
        if not key or "independent_gold_judge" not in key: errors.append(f"unjudged probe {probe['probe_id']}"); continue
        gold_positions[key["gold_choice_id"]] += 1
        gold_sequence.append(key["gold_choice_id"])
        if key.get("choice_position_balancing") != "private_seeded_per_probe_shuffle":
            errors.append(f"non-private probe choice shuffle {probe['probe_id']}")
        if key.get("choice_reference_remapped") is not True:
            errors.append(f"unremapped choice references {probe['probe_id']}")
        if key["gold_choice_id"] not in key.get("query_only_judge", {}).get("plausible_choice_ids", []):
            errors.append(f"query-only plausible set omits gold {probe['probe_id']}")
        query_norm = re.sub(r"[^a-z0-9 ]+", " ", probe.get("query", "").lower()).strip()
        probe_queries_by_user[probe["user_id"]].append((probe["probe_id"], shingles(query_norm, 3)))
        if re.search(r"\b(hidden habit|memory system|benchmark|gold answer)\b", probe.get("query", ""), re.I): errors.append(f"probe leaks benchmark framing {probe['probe_id']}")
        lengths = [len(choice.get("text", "")) for choice in probe.get("choices", [])]
        if lengths and min(lengths) and max(lengths) / min(lengths) > 2.5: errors.append(f"choice length giveaway {probe['probe_id']}")
        if any(session_id not in session_by_id for session_id in key.get("gold_evidence_session_ids", [])): errors.append(f"bad probe evidence {probe['probe_id']}")
    if probes and max(gold_positions.values(), default=0) > len(probes) * 0.35: errors.append(f"gold position imbalance {dict(gold_positions)}")
    for period in range(1, min(9, len(gold_sequence))):
        if len(gold_sequence) >= period * 4 and all(
            value == gold_sequence[index % period] for index, value in enumerate(gold_sequence)
        ):
            errors.append(f"periodic gold-position leakage period={period}")
            break
    for user_id, query_sets in probe_queries_by_user.items():
        for index, (left_id, left) in enumerate(query_sets):
            for right_id, right in query_sets[index + 1:]:
                union = len(left | right)
                if union and len(left & right) / union >= 0.78: errors.append(f"semantic near-duplicate probes {left_id} {right_id}")

    complete = bool(profiles and arcs and sessions and probes and keys)
    if args.require_complete and not complete: errors.append("release artifacts are incomplete")
    status = "fail" if errors else "pass" if complete else "in_progress"
    report = {
        "status": status,
        "complete_release": complete,
        "errors": errors, "warnings": warnings, "counts": {
            "source_bundles": len(bundles), "curated_habits": len(curated), "users": len(profiles),
            "arc_events": len(arcs), "sessions": len(sessions), "probes": len(probes),
        }, "qwen_token_counts": token_counts,
    }
    path = args.dataset / "reports" / "v04_release_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(1 if report["status"] == "fail" else 0)


if __name__ == "__main__": main()
