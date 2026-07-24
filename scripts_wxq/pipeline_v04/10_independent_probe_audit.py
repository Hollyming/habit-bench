#!/usr/bin/env python3
"""Independently audit positive probes with a model different from the generator.

Each probe receives two logically isolated judgments:
1. query-only, with no user history or proposed label;
2. history-aware, with the complete raw visible lifeline but no annotations, habit graph, cited-evidence
   selection, or gold. History judgments are batched only within one user to avoid repeatedly sending the
   same ultra-long lifeline; every answer remains keyed by an opaque item ID.

Per-call caches make the audit safely resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
AUDIT_REVISION = "independent_positive_probe_audit_v3_full_lifeline_deterministic"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fingerprint(model: str, phase: str, payload: dict[str, Any]) -> str:
    material = json.dumps(
        {"revision": AUDIT_REVISION, "model": model, "phase": phase, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def raw_history_excerpt(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "session_index": session["session_index"],
        "timestamp": session["timestamp"],
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in session["messages"]
        ],
    }


def call_cached(
    args: argparse.Namespace,
    phase: str,
    probe_id: str,
    system: str,
    payload: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    cache = args.dataset / "work" / "independent_probe_audit" / args.model / phase / f"{probe_id}.json"
    fp = fingerprint(args.model, phase, payload)
    if cache.exists():
        row = json.loads(cache.read_text(encoding="utf-8"))
        if row.get("fingerprint") == fp:
            return row
    response = post_chat(
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
        transport="curl_stream",
        reasoning_effort=args.reasoning_effort,
        temperature=0.0,
    )
    row = {
        "fingerprint": fp,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "phase": phase,
        "probe_id": probe_id,
        "verdict": response["json"],
        "usage": response.get("usage", {}),
    }
    write_json(cache, row)
    return row


def query_only_payload(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Decide whether this travel multiple-choice question has a unique best answer without any private user history.",
        "opaque_item_id": hashlib.sha256(probe["probe_id"].encode()).hexdigest()[:16],
        "query": probe["query"],
        "choices": probe["choices"],
        "rules": [
            "Use only the query and choices. Do not invent a user preference.",
            "If two or more choices remain comparably defensible, choice_id must be UNRESOLVED.",
            "generic_best_exists is true only when ordinary travel reasoning yields one clearly dominant option.",
            "List every genuinely plausible choice_id, not merely one favorite.",
            "Return {choice_id,answerable_without_history,generic_best_exists,plausible_choice_ids,rationale,leakage_signals}.",
        ],
    }


def history_batch_payload(probes: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "For each item, choose the uniquely best action using only the complete raw longitudinal conversation history supplied below.",
        "items": [
            {
                "opaque_item_id": hashlib.sha256(probe["probe_id"].encode()).hexdigest()[:16],
                "query": probe["query"],
                "choices": probe["choices"],
            }
            for probe in probes
        ],
        "complete_raw_visible_lifeline": [raw_history_excerpt(session) for session in sessions],
        "rules": [
            "You are not given a habit graph, annotations, selected evidence, rationale, or proposed gold answer.",
            "Infer only preferences supported repeatedly by the raw user messages.",
            "Respect scope, soft boundaries, current-trip constraints, and exceptions.",
            "Set ambiguous=true if two choices remain comparably defensible even after reading the excerpts.",
            "Return {answers:[{opaque_item_id,choice_id,ambiguous,difficulty,evidence_session_ids,rationale,unsupported_assumptions}]}",
            "difficulty must be easy, medium, or hard for a capable model with these excerpts.",
        ],
    }


def valid_query_verdict(verdict: dict[str, Any], gold_choice_id: str) -> tuple[bool, list[str]]:
    errors = []
    plausible = verdict.get("plausible_choice_ids")
    if verdict.get("choice_id") != "UNRESOLVED": errors.append("resolved_without_history")
    if verdict.get("answerable_without_history") is not False: errors.append("answerable_without_history")
    if verdict.get("generic_best_exists") is not False: errors.append("generic_best_exists")
    if not isinstance(plausible, list) or len(set(plausible)) < 2: errors.append("fewer_than_two_plausible")
    if not isinstance(plausible, list) or gold_choice_id not in plausible: errors.append("gold_not_query_only_plausible")
    return not errors, errors


def valid_history_verdict(verdict: dict[str, Any], gold_choice_id: str) -> tuple[bool, list[str]]:
    errors = []
    if verdict.get("choice_id") != gold_choice_id: errors.append("gold_disagreement")
    if verdict.get("ambiguous") is not False: errors.append("ambiguous_with_history")
    if verdict.get("difficulty") not in {"medium", "hard"}: errors.append("not_medium_or_hard")
    if not isinstance(verdict.get("rationale"), str) or len(verdict["rationale"].strip()) < 40:
        errors.append("missing_rationale")
    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--history-batch-size", type=int, default=4)
    parser.add_argument(
        "--only-probes",
        nargs="*",
        default=None,
        help="Audit only these probe IDs (intended for smoke tests or targeted reruns).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N selected probes.")
    parser.add_argument(
        "--report-name",
        default="independent_positive_probe_audit.json",
        help="Filename written below the dataset reports directory.",
    )
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")

    probes = read_jsonl(args.dataset / "public" / "probes.jsonl")
    if args.only_probes:
        requested = set(args.only_probes)
        known = {probe["probe_id"] for probe in probes}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"Unknown probe IDs: {missing}")
        probes = [probe for probe in probes if probe["probe_id"] in requested]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        probes = probes[:args.limit]
    if not probes:
        raise SystemExit("No probes selected")
    keys = {row["probe_id"]: row for row in read_jsonl(args.dataset / "private" / "probe_key.jsonl")}
    sessions = {row["session_id"]: row for row in read_jsonl(args.dataset / "private" / "sessions_with_annotations.jsonl")}
    probe_by_id = {row["probe_id"]: row for row in probes}
    query_tasks = [("query_only", probe["probe_id"], [probe["probe_id"]]) for probe in probes]
    history_tasks = []
    for user_id in sorted({probe["user_id"] for probe in probes}):
        user_probes = [probe for probe in probes if probe["user_id"] == user_id]
        for start in range(0, len(user_probes), args.history_batch_size):
            probe_ids = [probe["probe_id"] for probe in user_probes[start:start + args.history_batch_size]]
            batch_id = f"{user_id}_b{start // args.history_batch_size:03d}"
            history_tasks.append(("history_aware", batch_id, probe_ids))
    tasks = query_tasks + history_tasks
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    failures = []

    def run(task: tuple[str, str, list[str]]) -> tuple[tuple[str, str, list[str]], dict[str, Any]]:
        phase, item_id, probe_ids = task
        if phase == "query_only":
            probe = probe_by_id[probe_ids[0]]
            row = call_cached(
                args, phase, item_id,
                "You are an independent adversarial auditor of benchmark question answerability.",
                query_only_payload(probe), 1800,
            )
        else:
            batch_probes = [probe_by_id[probe_id] for probe_id in probe_ids]
            user_id = batch_probes[0]["user_id"]
            max_visible_index = max(probe["visible_history_scope"]["max_session_index"] for probe in batch_probes)
            complete_history = sorted(
                (
                    session for session in sessions.values()
                    if session["user_id"] == user_id and session["session_index"] <= max_visible_index
                ),
                key=lambda session: session["session_index"],
            )
            row = call_cached(
                args, "history_aware_full_lifeline", item_id,
                "You are an independent adversarial auditor of longitudinal user-memory questions.",
                history_batch_payload(batch_probes, complete_history), 7000,
            )
        return task, row

    query_workers = max(1, min(2, args.workers - 1))
    history_workers = max(1, args.workers - query_workers)
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=query_workers) as query_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=history_workers) as history_executor,
    ):
        future_map = {
            **{query_executor.submit(run, task): task for task in query_tasks},
            **{history_executor.submit(run, task): task for task in history_tasks},
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            task = future_map[future]
            try:
                completed_task, row = future.result()
                phase, _item_id, probe_ids = completed_task
                if phase == "query_only":
                    outputs[(phase, probe_ids[0])] = row
                else:
                    answers = row.get("verdict", {}).get("answers", [])
                    by_opaque_id = {
                        answer.get("opaque_item_id"): answer
                        for answer in answers if isinstance(answer, dict)
                    }
                    for probe_id in probe_ids:
                        opaque_id = hashlib.sha256(probe_id.encode()).hexdigest()[:16]
                        outputs[(phase, probe_id)] = {
                            **row,
                            "verdict": by_opaque_id.get(opaque_id, {}),
                        }
            except Exception as exc:
                failures.append({"phase": task[0], "item_id": task[1], "probe_ids": task[2], "error": str(exc)})
            print(f"independent_audit_progress {index}/{len(tasks)} failures={len(failures)}", flush=True)

    items = []
    for probe in probes:
        probe_id = probe["probe_id"]
        key = keys[probe_id]
        query_row = outputs.get(("query_only", probe_id))
        history_row = outputs.get(("history_aware", probe_id))
        query_verdict = query_row.get("verdict", {}) if query_row else {}
        history_verdict = history_row.get("verdict", {}) if history_row else {}
        query_pass, query_errors = valid_query_verdict(query_verdict, key["gold_choice_id"])
        history_pass, history_errors = valid_history_verdict(history_verdict, key["gold_choice_id"])
        items.append({
            "probe_id": probe_id,
            "user_id": probe["user_id"],
            "habit_id": key["habit_id"],
            "probe_type": key["probe_type"],
            "gold_choice_id": key["gold_choice_id"],
            "query_only_pass": query_pass,
            "query_only_errors": query_errors,
            "query_only_verdict": query_verdict,
            "history_aware_pass": history_pass,
            "history_aware_errors": history_errors,
            "history_aware_verdict": history_verdict,
            "status": "pass" if query_pass and history_pass else "reject",
        })

    status_counts = Counter(row["status"] for row in items)
    report = {
        "status": "pass" if not failures and status_counts.get("reject", 0) == 0 else "reject",
        "audit_revision": AUDIT_REVISION,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "probe_count": len(probes),
        "physical_call_count": len(tasks),
        "logical_judgment_count": len(probes) * 2,
        "history_scope": "complete_raw_visible_lifeline",
        "history_batch_size": args.history_batch_size,
        "completed_call_count": len(outputs),
        "failed_call_count": len(failures),
        "call_failures": failures,
        "status_counts": dict(status_counts),
        "query_only_pass_count": sum(row["query_only_pass"] for row in items),
        "history_aware_pass_count": sum(row["history_aware_pass"] for row in items),
        "query_only_error_counts": dict(Counter(error for row in items for error in row["query_only_errors"])),
        "history_aware_error_counts": dict(Counter(error for row in items for error in row["history_aware_errors"])),
        "items": items,
    }
    write_json(args.dataset / "reports" / args.report_name, report)
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
