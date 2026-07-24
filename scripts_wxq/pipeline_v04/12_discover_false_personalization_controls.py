#!/usr/bin/env python3
"""Discover and independently audit multi-user false-personalization controls.

Controls are dimensions with repeated but context-dependent observations.  They
are not negative answers: their correct policy is to avoid storing a fixed
default and use current-trip conditions instead.  This stage never edits
sessions or public probes.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

from api_client import write_json, write_jsonl


_pipeline = import_module("04_generate_benchmark")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
REVISION = "false_personalization_control_discovery_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def call(args: argparse.Namespace, model: str, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    request_args = copy.copy(args)
    request_args.model = model
    return _pipeline.api_json(request_args, system, payload, max_tokens)


def raw_lifeline(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": row["session_id"], "session_index": row["session_index"],
            "timestamp": row["timestamp"],
            "messages": [{"role": item["role"], "content": item["content"]} for item in row["messages"]],
        }
        for row in sorted(sessions, key=lambda row: row["session_index"])
    ]


def valid_control(row: Any, session_episode: dict[str, str | None]) -> str | None:
    if not isinstance(row, dict):
        return "not_object"
    required = [
        "control_id", "family", "decision_dimension", "opportunity_description",
        "strongest_false_inference", "tempting_false_inferences", "correct_memory_policy",
        "why_no_stable_habit", "evidence",
    ]
    if any(not row.get(field) for field in required):
        return "missing_required_field"
    evidence = row.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 4:
        return "insufficient_evidence"
    ids = [item.get("session_id") for item in evidence if isinstance(item, dict)]
    if len(set(ids)) < 4 or not set(ids).issubset(session_episode):
        return "invalid_evidence_ids"
    if len({session_episode[sid] for sid in ids}) < 3:
        return "evidence_not_three_episodes"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--generator-model", default="gpt-5.5")
    parser.add_argument("--audit-model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--transport", choices=["curl_stream", "curl", "urllib"], default="curl_stream")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--only-users", default="")
    parser.add_argument("--revision", default=REVISION,
                        help="Cache/report namespace for a rediscovery pass; does not alter sessions.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSONL destination. Required for targeted rediscovery so the frozen control set is preserved.")
    parser.add_argument("--target-per-user", type=int, default=5,
                        help="Minimum independently accepted controls per selected user.")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")

    dossiers = {row["user_id"]: row for row in read_jsonl(args.dataset / "private" / "user_dossiers.jsonl")}
    sessions_by_user: dict[str, list[dict[str, Any]]] = {user_id: [] for user_id in dossiers}
    for row in read_jsonl(args.dataset / "private" / "sessions_with_annotations.jsonl"):
        sessions_by_user[row["user_id"]].append(row)
    selected = {item.strip() for item in args.only_users.split(",") if item.strip()} or set(dossiers)
    if unknown := selected - set(dossiers):
        raise SystemExit(f"unknown users: {sorted(unknown)}")

    accepted, reports = [], []
    for user_id in sorted(selected):
        sessions = sessions_by_user[user_id]
        episode_by_session = {row["session_id"]: row.get("memory_annotations", {}).get("episode_id") for row in sessions}
        discovery_cache = args.dataset / "work" / "false_personalization" / f"control_discovery_{args.revision}" / f"{user_id}.json"
        if discovery_cache.exists():
            raw = json.loads(discovery_cache.read_text(encoding="utf-8"))
        else:
            dossier = dossiers[user_id]
            raw = call(args, args.generator_model, "You discover rigorous false-personalization controls in long travel-assistant lifelines.", {
                "task": "Find exactly eight candidate decision dimensions where a naive memory system might infer a durable user preference, but the complete lifeline does not support storing a fixed default.",
                "user_id": user_id,
                "known_supported_habits_to_avoid": dossier["habits"],
                "complete_raw_lifeline": raw_lifeline(sessions),
                "requirements": [
                    "Return {controls:[{control_id,family,decision_dimension,opportunity_description,strongest_false_inference,tempting_false_inferences,correct_memory_policy,why_no_stable_habit,evidence:[{session_id,evidence_quote,role_in_nonhabit}]}]}",
                    "A control must have repeated, tempting but mixed/context-conditioned observations across at least four sessions and three episodes; a one-off absence is not a control.",
                    "Do not duplicate, weaken, negate, or re-label any listed supported habit. Avoid a mere threshold variant of a supported default.",
                    "correct_memory_policy must be operational: keep this dimension unset or condition it on current trip facts. It must not say merely 'refuse' or 'insufficient information'.",
                    "The false inference should be a plausible but overconfident personalization a memory agent could make from noisy history.",
                    "Use only raw user/assistant messages; do not invent facts, labels, or events.",
                ],
            }, 22000)
            write_json(discovery_cache, raw)
        candidates = raw.get("controls", []) if isinstance(raw, dict) else []
        valid, invalid = [], {}
        for row in candidates:
            error = valid_control(row, episode_by_session)
            if error: invalid[str(row.get("control_id"))] = error
            else: valid.append(row)
        audit_cache = args.dataset / "work" / "false_personalization" / f"control_audit_{args.revision}" / f"{user_id}.json"
        if audit_cache.exists():
            audit = json.loads(audit_cache.read_text(encoding="utf-8"))
        else:
            audit = call(args, args.audit_model, "You independently audit alleged non-habits from complete longitudinal travel conversations.", {
                "task": "For every proposed dimension, decide whether the raw lifeline supports a reusable stable preference or instead supports a context-conditioned non-personalization policy.",
                "complete_raw_lifeline": raw_lifeline(sessions),
                "candidate_controls": valid,
                "requirements": [
                    "You are not given the user's habit graph, proposed labels, or generator rationale beyond the candidate descriptions.",
                    "Return {reviews:[{control_id,stable_preference_supported,correct_memory_policy_valid,evidence_session_ids,rationale,confidence}]}",
                    "Set stable_preference_supported=false only if the apparent pattern remains mixed, trip-specific, assistant-led, or conditional after inspecting the complete lifeline.",
                    "Reject a control if the lifeline actually supports a stable preference on that same decision dimension.",
                ],
            }, 14000)
            write_json(audit_cache, audit)
        review_by_id = {row.get("control_id"): row for row in audit.get("reviews", []) if isinstance(row, dict)}
        user_accepted = []
        used_families = set()
        for control in valid:
            review = review_by_id.get(control["control_id"], {})
            if review.get("stable_preference_supported") is not False or review.get("correct_memory_policy_valid") is not True:
                continue
            if control["family"] in used_families:
                continue
            control = copy.deepcopy(control)
            control["user_id"] = user_id
            control["adjudication"] = {"model": args.audit_model, "reasoning_effort": args.reasoning_effort, **review}
            control["control_revision"] = REVISION
            user_accepted.append(control)
            used_families.add(control["family"])
            if len(user_accepted) == args.target_per_user:
                break
        accepted.extend(user_accepted)
        reports.append({
            "user_id": user_id, "generated_count": len(candidates), "structural_invalid": invalid,
            "independently_accepted_count": len(user_accepted),
            "accepted_control_ids": [row["control_id"] for row in user_accepted],
        })
        print(f"false_control_progress {user_id} accepted={len(user_accepted)}", flush=True)

    status = "pass" if all(row["independently_accepted_count"] >= args.target_per_user for row in reports) else "reject"
    report_name = f"false_personalization_control_discovery_{args.revision}.json"
    write_json(args.dataset / "reports" / report_name, {
        "status": status, "revision": args.revision, "generator_model": args.generator_model,
        "audit_model": args.audit_model, "target_per_user": args.target_per_user, "reports": reports,
    })
    if status == "pass":
        destination = args.output or (args.dataset / "private" / "false_personalization_controls_v1.jsonl")
        write_jsonl(destination, accepted)
    print(json.dumps({"status": status, "accepted_control_count": len(accepted), "reports": reports}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
