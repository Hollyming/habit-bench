#!/usr/bin/env python3
"""Regenerate only positive probes rejected by the full-lifeline independent audit.

The script never edits sessions or dossiers.  It writes replacements only if every
selected target passes a new deterministic, model-separated audit.  Generation
uses gpt-5.5 xhigh; audit uses gpt-5.6-terra xhigh at temperature zero.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from api_client import write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
REVISION = "positive_semantic_repair_r1"
LABELS = ["A", "B", "C", "D"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def fingerprint(*parts: Any) -> str:
    text = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def streaming_json_call(args: argparse.Namespace, model: str, system: str, payload: dict[str, Any], max_tokens: int, temperature: float) -> dict[str, Any]:
    """Use SSE so long xhigh reasoning keeps the API connection alive."""
    request = {
        "model": model,
        "messages": [{"role": "system", "content": system + " Return strict JSON only."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "temperature": temperature, "max_tokens": max_tokens,
        "reasoning_effort": args.reasoning_effort, "response_format": {"type": "json_object"}, "stream": True,
    }
    encoded = json.dumps(request, ensure_ascii=False)
    url = args.base_url.rstrip("/") + "/chat/completions"
    last_error = None
    for attempt in range(args.retries + 1):
        proc = subprocess.run(
            ["curl", "-sS", "-N", "--http1.1", "--connect-timeout", "30", "--max-time", str(args.timeout),
             "-H", f"Authorization: Bearer {args.api_key}", "-H", "Content-Type: application/json", "--data-binary", "@-", url],
            input=encoded, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        chunks = []
        try:
            for line in proc.stdout.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(json.dumps(event["error"], ensure_ascii=False))
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if isinstance(content, str):
                        chunks.append(content)
            content = "".join(chunks).strip()
            if proc.returncode == 0 and content:
                return json.loads(content)
            if proc.returncode == 0 and proc.stdout.lstrip().startswith("{"):
                body = json.loads(proc.stdout)
                return json.loads(body["choices"][0]["message"]["content"])
            last_error = f"stream_exit_{proc.returncode}:{proc.stderr[:500]}:content_chars={len(content)}"
        except Exception as exc:
            last_error = f"stream_parse:{type(exc).__name__}:{str(exc)[:500]}:tail={proc.stdout[-500:]}"
        if attempt < args.retries:
            time.sleep(min(2 * (attempt + 1), 10))
    raise RuntimeError(last_error or "stream_request_failed")


def cached_call(args: argparse.Namespace, stage: str, cache_id: str, system: str, payload: dict[str, Any], max_tokens: int, *, temperature: float) -> dict[str, Any]:
    cache = args.dataset / "work" / "positive_semantic_repair" / stage / f"{cache_id}.json"
    fp = fingerprint(REVISION, stage, args.generator_model if stage == "generation" else args.audit_model, payload, temperature)
    if cache.exists():
        row = json.loads(cache.read_text(encoding="utf-8"))
        if row.get("fingerprint") == fp:
            return row["response"]
    model = args.generator_model if stage == "generation" else args.audit_model
    response = streaming_json_call(args, model, system, payload, max_tokens, temperature)
    write_json(cache, {"fingerprint": fp, "response": response})
    return response


def evidence_for_habit(habit_id: str, sessions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str], dict[str, str | None]]:
    evidence, valid, episodes = [], set(), {}
    for session in sessions:
        annotations = session.get("memory_annotations", {})
        signal = next((item for item in annotations.get("verified_habit_signals", []) if item.get("habit_id") == habit_id), None)
        if not signal:
            continue
        sid = session["session_id"]
        valid.add(sid)
        episodes[sid] = annotations.get("episode_id")
        evidence.append({
            "session_id": sid, "session_index": session["session_index"],
            "episode_id": annotations.get("episode_id"), "signal_type": signal.get("signal_type"),
            "evidence_quote": signal.get("evidence_quote"),
            "user_turn_samples": [m["content"] for m in session["messages"] if m["role"] == "user"][-2:],
        })
    return evidence, valid, episodes


def candidate_error(row: Any, assignment: dict[str, Any]) -> str | None:
    if not isinstance(row, dict) or row.get("assignment_id") != assignment["assignment_id"]:
        return "assignment_mismatch"
    choices = row.get("choices")
    ids = [item.get("choice_id") for item in choices] if isinstance(choices, list) else []
    if len(ids) != 4 or set(ids) != set(LABELS) or row.get("gold_choice_id") not in ids:
        return "invalid_choices"
    if row.get("closest_distractor_choice_id") not in ids or row.get("closest_distractor_choice_id") == row.get("gold_choice_id"):
        return "invalid_closest_distractor"
    lengths = [len(str(item.get("text", "")).strip()) for item in choices]
    if not lengths or min(lengths) < 30 or max(lengths) > 2.4 * min(lengths):
        return "choice_length_leakage"
    cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
    if len(cited) < 3 or not set(cited).issubset(assignment["valid_evidence_ids"]):
        return "invalid_evidence_citations"
    if len({assignment["session_to_episode"][sid] for sid in cited}) < 3:
        return "evidence_not_three_episodes"
    if len(str(row.get("query", "")).strip()) < 60:
        return "query_too_short"
    return None


def query_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "For every item, decide whether a unique best answer exists without private user history.",
        "items": [
            {"opaque_item_id": row["opaque_item_id"], "query": row["query"], "choices": row["choices"]}
            for row in rows
        ],
        "rules": [
            "Use only query and choices; do not invent a preference.",
            "If two or more choices are comparably defensible, set choice_id=UNRESOLVED.",
            "Set generic_best_exists=true only when ordinary travel reasoning supplies a clearly dominant answer.",
            "List every genuinely plausible choice_id.",
            "Return {answers:[{opaque_item_id,choice_id,answerable_without_history,generic_best_exists,plausible_choice_ids,rationale,leakage_signals}]}",
        ],
    }


def history_payload(rows: list[dict[str, Any]], lifeline: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "For every item, select the uniquely best action from the complete raw user lifeline.",
        "items": [
            {"opaque_item_id": row["opaque_item_id"], "query": row["query"], "choices": row["choices"]}
            for row in rows
        ],
        "complete_raw_visible_lifeline": lifeline,
        "rules": [
            "No habit graph, annotations, selected evidence, label, or proposed answer is available.",
            "Infer only repeatedly supported, scoped user behavior and retain genuine exceptions.",
            "Set ambiguous=true if history cannot distinguish a unique best choice.",
            "Return {answers:[{opaque_item_id,choice_id,ambiguous,difficulty,evidence_session_ids,rationale,unsupported_assumptions}]}",
            "difficulty must be easy, medium, or hard for a capable agent with this history.",
        ],
    }


def valid_query(verdict: dict[str, Any], gold: str) -> list[str]:
    plausible = verdict.get("plausible_choice_ids")
    errors = []
    if verdict.get("choice_id") != "UNRESOLVED": errors.append("resolved_without_history")
    if verdict.get("answerable_without_history") is not False: errors.append("answerable_without_history")
    if verdict.get("generic_best_exists") is not False: errors.append("generic_best_exists")
    if not isinstance(plausible, list) or len(set(plausible)) < 2: errors.append("fewer_than_two_plausible")
    if not isinstance(plausible, list) or gold not in plausible: errors.append("gold_not_query_only_plausible")
    return errors


def valid_history(verdict: dict[str, Any], gold: str) -> list[str]:
    errors = []
    if verdict.get("choice_id") != gold: errors.append("gold_disagreement")
    if verdict.get("ambiguous") is not False: errors.append("ambiguous_with_history")
    if verdict.get("difficulty") not in {"medium", "hard"}: errors.append("not_medium_or_hard")
    if not isinstance(verdict.get("rationale"), str) or len(verdict["rationale"].strip()) < 40: errors.append("missing_rationale")
    return errors


def remap_text(value: Any, mapping: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(r"\b(choice|option)\s+([ABCD])\b", lambda m: f"{m.group(1)} {mapping.get(m.group(2), m.group(2))}", value, flags=re.I)


def private_seed(dataset: Path) -> str:
    path = dataset / "private" / "probe_shuffle_seed.txt"
    seed = path.read_text(encoding="utf-8").strip()
    if len(seed) < 32:
        raise ValueError("missing valid private probe shuffle seed")
    return seed


def rank(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--generator-model", default="gpt-5.5")
    parser.add_argument("--audit-model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--only-probes", nargs="*", default=None)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retire-shortfalls",
        action="store_true",
        help="Explicitly retire probes that remain invalid after all configured repair rounds; never retain them with a weak label.",
    )
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")

    public = read_jsonl(args.dataset / "public" / "probes.jsonl")
    keys = read_jsonl(args.dataset / "private" / "probe_key.jsonl")
    audit = json.loads((args.dataset / "reports" / "independent_positive_probe_audit.json").read_text(encoding="utf-8"))
    rejected = {item["probe_id"]: item for item in audit["items"] if item["status"] == "reject"}
    if args.only_probes:
        rejected = {pid: rejected[pid] for pid in args.only_probes if pid in rejected}
    if not rejected:
        raise SystemExit("No rejected probes selected")
    public_by_id, key_by_id = {row["probe_id"]: row for row in public}, {row["probe_id"]: row for row in keys}
    dossiers = {row["user_id"]: row for row in read_jsonl(args.dataset / "private" / "user_dossiers.jsonl")}
    sessions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in read_jsonl(args.dataset / "private" / "sessions_with_annotations.jsonl"):
        sessions_by_user[session["user_id"]].append(session)
    for rows in sessions_by_user.values():
        rows.sort(key=lambda row: row["session_index"])

    assignments, seen_queries = [], [row["query"] for row in public]
    for probe_id, audit_item in sorted(rejected.items()):
        key, probe = key_by_id[probe_id], public_by_id[probe_id]
        user_id, habit_id = probe["user_id"], key["habit_id"]
        evidence, valid, episodes = evidence_for_habit(habit_id, sessions_by_user[user_id])
        assignments.append({
            "assignment_id": probe_id, "probe_id": probe_id, "user_id": user_id,
            "habit_id": habit_id, "probe_type": key["probe_type"],
            "hidden_target_habit": key["hidden_habit_graph"],
            "other_habits": [habit for habit in dossiers[user_id]["habits"] if habit["habit_id"] != habit_id],
            "evidence_sessions": evidence, "valid_evidence_ids": valid, "session_to_episode": episodes,
            "prior_failures": {"query_only": audit_item["query_only_errors"], "history_aware": audit_item["history_aware_errors"]},
        })
    assignment_by_id = {row["assignment_id"]: row for row in assignments}

    generated = []
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments: by_user[assignment["user_id"]].append(assignment)
    for user_id, user_assignments in sorted(by_user.items()):
        for offset in range(0, len(user_assignments), 4):
            chunk = user_assignments[offset:offset + 4]
            payload_assignments = []
            for assignment in chunk:
                payload_assignments.append({
                    key: assignment[key] for key in ["assignment_id", "probe_type", "hidden_target_habit", "other_habits", "evidence_sessions", "prior_failures"]
                })
            raw = cached_call(args, "generation", f"{user_id}_{offset:03d}",
                "You repair semantically invalid long-history travel-memory benchmark probes.", {
                    "task": "Write exactly three entirely new candidate probes per assignment. They will be checked against the full raw lifeline by a different model.",
                    "assignments": payload_assignments,
                    "existing_queries_to_avoid": seen_queries,
                    "requirements": [
                        "Return {probes:[{assignment_id,variant_id,probe_type,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
                        "Do not preserve prior wording: rebuild the decision from the target habit, its evidence, and the other active habits.",
                        "The target policy must be the unique best answer after history, without contradicting any listed other habit, boundary, exception, or trip-specific condition.",
                        "Before writing, form a gold/near-miss pair with comparable ordinary travel utility. The query alone must leave at least two choices genuinely defensible; neither option may dominate on explicit price, timing, routing, location, amenities, feasibility, or convenience.",
                        "Use a new realistic travel context. The query must not state the user's preference, a target threshold, or a constraint that itself decides the tradeoff.",
                        "All four choices must be operationally viable and similarly specific. The near miss should reflect a competing user policy, a boundary, or an exception rather than a straw man.",
                        "Cite at least three supplied evidence session IDs from three episodes. Make the resulting history-aware distinction medium or hard, not merely verbose.",
                        "Do not mention benchmark, labels, habit, evidence, memory, or insufficient information.",
                    ],
                }, 12000, temperature=0.7)
            generated.extend(row for row in raw.get("probes", []) if isinstance(row, dict))

    candidates = []
    structural_errors: dict[str, str] = {}
    for row in generated:
        assignment = assignment_by_id.get(row.get("assignment_id"))
        candidate_id = f"{row.get('assignment_id')}::{row.get('variant_id')}::r0"
        error = candidate_error(row, assignment) if assignment else "unknown_assignment"
        if error:
            structural_errors[candidate_id] = error
            continue
        row = copy.deepcopy(row)
        row["candidate_id"] = candidate_id
        row["opaque_item_id"] = hashlib.sha256(candidate_id.encode()).hexdigest()[:16]
        candidates.append(row)

    query_verdicts: dict[str, dict[str, Any]] = {}
    for start in range(0, len(candidates), 12):
        chunk = candidates[start:start + 12]
        raw = cached_call(args, "query_audit", f"b{start // 12:03d}",
            "You are an adversarial independent auditor of multiple-choice question answerability.",
            query_payload(chunk), 4000, temperature=0.0)
        by_opaque = {row.get("opaque_item_id"): row for row in raw.get("answers", []) if isinstance(row, dict)}
        for row in chunk: query_verdicts[row["candidate_id"]] = by_opaque.get(row["opaque_item_id"], {})

    history_verdicts: dict[str, dict[str, Any]] = {}
    by_user_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates: by_user_candidates[assignment_by_id[row["assignment_id"]]["user_id"]].append(row)
    for user_id, rows in sorted(by_user_candidates.items()):
        raw_lifeline = [
            {"session_id": session["session_id"], "session_index": session["session_index"], "timestamp": session["timestamp"],
             "messages": [{"role": msg["role"], "content": msg["content"]} for msg in session["messages"]]}
            for session in sessions_by_user[user_id]
        ]
        for start in range(0, len(rows), 4):
            chunk = rows[start:start + 4]
            raw = cached_call(args, "history_audit", f"{user_id}_b{start // 4:03d}",
                "You are an independent adversarial auditor of longitudinal user-memory questions.",
                history_payload(chunk, raw_lifeline), 7000, temperature=0.0)
            by_opaque = {row.get("opaque_item_id"): row for row in raw.get("answers", []) if isinstance(row, dict)}
            for row in chunk: history_verdicts[row["candidate_id"]] = by_opaque.get(row["opaque_item_id"], {})

    def select_candidates() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        selected: dict[str, dict[str, Any]] = {}
        records = []
        for candidate in candidates:
            q = query_verdicts.get(candidate["candidate_id"], {})
            h = history_verdicts.get(candidate["candidate_id"], {})
            errors = valid_query(q, candidate["gold_choice_id"]) + valid_history(h, candidate["gold_choice_id"])
            record = {"candidate_id": candidate["candidate_id"], "assignment_id": candidate["assignment_id"], "errors": errors, "query_only": q, "history_aware": h}
            records.append(record)
            if not errors and candidate["assignment_id"] not in selected:
                selected[candidate["assignment_id"]] = {"row": candidate, "query": q, "history": h}
        return selected, records

    accepted, decisions = select_candidates()
    for repair_round in range(1, args.max_repair_rounds + 1):
        shortfalls = sorted(set(assignment_by_id) - set(accepted))
        if not shortfalls:
            break
        retry_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for assignment_id in shortfalls:
            assignment = assignment_by_id[assignment_id]
            failed = []
            for candidate in candidates:
                if candidate["assignment_id"] != assignment_id:
                    continue
                record = next(item for item in decisions if item["candidate_id"] == candidate["candidate_id"])
                failed.append({
                    "candidate": {key: candidate.get(key) for key in ["probe_type", "query", "choices", "gold_choice_id", "gold_action", "gold_evidence_session_ids"]},
                    "audit_errors": record["errors"], "query_only_verdict": record["query_only"], "history_aware_verdict": record["history_aware"],
                })
            # By round two an assignment can contain six lengthy failed candidates.
            # The newest three already encode the first repair attempt and are the
            # most actionable feedback.  Keeping all six can exhaust reasoning
            # output before the model emits its required JSON.
            feedback = failed[-3:] if repair_round >= 2 else failed
            retry_by_user[assignment["user_id"]].append({"assignment": assignment, "failed": feedback})
        fresh_rows = []
        for user_id, retry_rows in sorted(retry_by_user.items()):
            for offset in range(0, len(retry_rows), 4):
                chunk = retry_rows[offset:offset + 4]
                payload_assignments = []
                for item in chunk:
                    assignment = item["assignment"]
                    payload_assignments.append({
                        key: assignment[key] for key in ["assignment_id", "probe_type", "hidden_target_habit", "other_habits", "evidence_sessions"]
                    } | {"failed_candidates_with_independent_audit": item["failed"]})
                retry_requirements = [
                    "Return {probes:[{assignment_id,variant_id,probe_type,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
                            "The history-aware audit sees the complete lifeline and will reject a gold option that loses on price, timing, route, airport, or convenience under the target habit's stated boundary or exception.",
                            "For a soft preference, give the gold and closest distractor genuinely comparable ordinary utility; never use the preference to override a material generic disadvantage.",
                            "If a prior candidate was easy, make the decisive difference depend on a scoped preference plus a second competing personal consideration, while the query alone remains unresolved.",
                    "All choices must be viable, balanced, and distinct from the failed candidates. Cite at least three supplied evidence sessions from three episodes.",
                ]
                if repair_round >= 3:
                    retry_requirements.insert(1, "Every candidate must contain exactly four choices, with choice_id values A, B, C, and D each appearing exactly once. Do not omit D or return a three-choice question.")
                raw = cached_call(args, "regeneration", f"r{repair_round}_{user_id}_{offset:03d}",
                    "You repair benchmark candidates after a blind full-lifeline adversarial audit.", {
                        "task": "Write exactly three replacement candidates per assignment. Every listed prior candidate failed; address its specific audit feedback rather than paraphrasing it.",
                        "assignments": payload_assignments,
                        "requirements": retry_requirements,
                    }, 16000 if repair_round >= 2 else 12000, temperature=0.7)
                fresh_rows.extend(row for row in raw.get("probes", []) if isinstance(row, dict))
        fresh_candidates = []
        for row in fresh_rows:
            assignment = assignment_by_id.get(row.get("assignment_id"))
            candidate_id = f"{row.get('assignment_id')}::{row.get('variant_id')}::r{repair_round}"
            error = candidate_error(row, assignment) if assignment else "unknown_assignment"
            if error:
                structural_errors[candidate_id] = error
                continue
            row = copy.deepcopy(row)
            row["candidate_id"] = candidate_id
            row["opaque_item_id"] = hashlib.sha256(candidate_id.encode()).hexdigest()[:16]
            fresh_candidates.append(row)
        for start in range(0, len(fresh_candidates), 12):
            chunk = fresh_candidates[start:start + 12]
            raw = cached_call(args, "query_audit", f"r{repair_round}_b{start // 12:03d}",
                "You are an adversarial independent auditor of multiple-choice question answerability.", query_payload(chunk), 4000, temperature=0.0)
            by_opaque = {row.get("opaque_item_id"): row for row in raw.get("answers", []) if isinstance(row, dict)}
            for row in chunk: query_verdicts[row["candidate_id"]] = by_opaque.get(row["opaque_item_id"], {})
        fresh_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in fresh_candidates: fresh_by_user[assignment_by_id[row["assignment_id"]]["user_id"]].append(row)
        for user_id, rows in sorted(fresh_by_user.items()):
            raw_lifeline = [
                {"session_id": session["session_id"], "session_index": session["session_index"], "timestamp": session["timestamp"],
                 "messages": [{"role": msg["role"], "content": msg["content"]} for msg in session["messages"]]}
                for session in sessions_by_user[user_id]
            ]
            for start in range(0, len(rows), 4):
                chunk = rows[start:start + 4]
                raw = cached_call(args, "history_audit", f"{user_id}_r{repair_round}_b{start // 4:03d}",
                    "You are an independent adversarial auditor of longitudinal user-memory questions.", history_payload(chunk, raw_lifeline), 7000, temperature=0.0)
                by_opaque = {row.get("opaque_item_id"): row for row in raw.get("answers", []) if isinstance(row, dict)}
                for row in chunk: history_verdicts[row["candidate_id"]] = by_opaque.get(row["opaque_item_id"], {})
        candidates.extend(fresh_candidates)
        accepted, decisions = select_candidates()

    shortfalls = sorted(set(assignment_by_id) - set(accepted))
    retired_probe_ids = shortfalls if args.retire_shortfalls else []
    report = {
        "status": "pass" if not shortfalls else ("pass_with_retirements" if args.retire_shortfalls else "reject"), "revision": REVISION,
        "target_probe_count": len(assignments), "candidate_count": len(candidates),
        "accepted_replacement_count": len(accepted), "shortfalls": shortfalls,
        "retired_probe_ids": retired_probe_ids,
        "structural_errors": structural_errors,
        "error_counts": dict(Counter(error for row in decisions for error in row["errors"])),
        "decisions": decisions,
    }
    write_json(args.dataset / "reports" / "positive_semantic_repair_r1.json", report)
    if (shortfalls and not args.retire_shortfalls) or args.dry_run:
        print(json.dumps({key: value for key, value in report.items() if key != "decisions"}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if args.dry_run else 1)

    archive = args.dataset / "reports" / "pre_positive_semantic_repair_r1"
    archive.mkdir(parents=True, exist_ok=True)
    for path in [args.dataset / "public" / "probes.jsonl", args.dataset / "private" / "probe_key.jsonl"]:
        target = archive / path.name
        if not target.exists(): shutil.copy2(path, target)
    seed = private_seed(args.dataset)
    for probe_id in retired_probe_ids:
        public_by_id.pop(probe_id, None)
        key_by_id.pop(probe_id, None)
    for probe_id, replacement in accepted.items():
        row, q, h = replacement["row"], replacement["query"], replacement["history"]
        old_public, old_key = public_by_id[probe_id], key_by_id[probe_id]
        ordered = sorted(row["choices"], key=lambda choice: rank(seed, f"probe_choices:{probe_id}", choice["choice_id"]))
        mapping = {choice["choice_id"]: label for label, choice in zip(LABELS, ordered)}
        old_public["query"] = row["query"]
        old_public["choices"] = [{"choice_id": label, "text": choice["text"]} for label, choice in zip(LABELS, ordered)]
        old_public.setdefault("metadata", {})["semantic_repair_revision"] = REVISION
        old_key.update({
            "probe_type": row["probe_type"], "gold_choice_id": mapping[row["gold_choice_id"]],
            "gold_action": row.get("gold_action"), "gold_evidence_session_ids": row["gold_evidence_session_ids"],
            "label_rationale": remap_text(row.get("label_rationale"), mapping),
            "generator_difficulty_rationale": remap_text(row.get("difficulty_rationale"), mapping),
            "generator_closest_distractor_choice_id": mapping[row["closest_distractor_choice_id"]],
            "query_only_judge": {**q, "plausible_choice_ids": [mapping.get(item, item) for item in q.get("plausible_choice_ids", [])], "rationale": remap_text(q.get("rationale"), mapping), "leakage_signals": [remap_text(item, mapping) for item in q.get("leakage_signals", [])]},
            "independent_gold_judge": {**h, "choice_id": mapping.get(h.get("choice_id"), h.get("choice_id")), "rationale": remap_text(h.get("rationale"), mapping)},
            "label_source": "gpt55_xhigh_targeted_regeneration__gpt56_terra_xhigh_full_lifeline_audit",
            "semantic_repair_revision": REVISION, "choice_position_balancing": "private_seeded_per_probe_shuffle", "choice_reference_remapped": True,
        })
    repaired_public = sorted(public_by_id.values(), key=lambda row: rank(seed, "probe_row", row["probe_id"]))
    repaired_keys = sorted(key_by_id.values(), key=lambda row: rank(seed, "probe_row", row["probe_id"]))
    write_jsonl(args.dataset / "public" / "probes.jsonl", repaired_public)
    write_jsonl(args.dataset / "private" / "probe_key.jsonl", repaired_keys)
    print(json.dumps({"status": "pass", "repaired_probe_count": len(accepted), "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
