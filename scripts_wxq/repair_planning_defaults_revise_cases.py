#!/usr/bin/env python
"""Repair model-flagged planning_defaults review rows.

This script only writes candidate artifacts. It does not overwrite the primary
public/private dataset files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


VALID_PROBE_TYPES = {"direct_use", "boundary", "exception", "explicit_retrieval"}
VALID_DECISIONS = {"repaired", "needs_manual"}
BANNED_PUBLIC_WORDS = {"habit", "benchmark", "gold", "label", "evidence", "dataset", "annotation"}


SYSTEM_PROMPT = """You repair a small number of HABIT-Bench planning_defaults probes.

The benchmark tests whether an agent can infer a scoped travel-planning default
from long user history and apply it only when appropriate. You will receive one
flagged probe, its hidden habit graph, the model-review complaint, and candidate
history sessions for the same user.

Repair only the public probe question and choices, plus the correct choice id
and evidence session ids if needed. Do not invent a new hidden habit and do not
change the probe type.

Quality bar:
- The correct answer must be unique and more compelling than the distractors.
- Distractors must be plausible travel-planning tradeoffs, not absurd answers.
- The query should not make the answer obvious without history, except
  explicit_retrieval may ask for the repeated preference.
- direct_use should apply default_action in an in-scope request.
- boundary should avoid applying default_action outside condition.
- exception should follow the explicit current override.
- public query/choices must not contain: habit, benchmark, gold, label,
  evidence, dataset, annotation.
- If the provided sessions cannot support a usable repair, return
  repair_decision=needs_manual.

Return strict JSON only."""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def message_preview(messages: Sequence[Dict[str, str]], limit: int = 1100) -> str:
    parts = []
    for message in messages[:10]:
        parts.append(f"{message.get('role')}: {compact(message.get('content', ''), 220)}")
    return compact(" | ".join(parts), limit)


def relevant_signal_types(probe_type: str) -> set[str]:
    if probe_type in {"direct_use", "explicit_retrieval"}:
        return {"support"}
    if probe_type == "boundary":
        return {"boundary_counterexample"}
    if probe_type == "exception":
        return {"exception"}
    return {"support", "boundary_counterexample", "exception"}


def classify_issue(notes: str) -> str:
    lower = notes.lower()
    if "model labeling failed" in lower or "runtimeerror" in lower:
        return "api_review_failure"
    if "evidence" in lower or "support" in lower or "substantiate" in lower:
        return "weak_or_misaligned_evidence"
    if "not uniquely" in lower or "ambiguous" in lower or "also" in lower:
        return "non_unique_gold"
    if "trivial" in lower or "obvious" in lower or "distractor" in lower:
        return "weak_distractors_or_leaky_wording"
    return "general_probe_quality"


def evidence_preview_for_review(sessions_by_id: Dict[str, Dict[str, Any]], evidence_ids: Sequence[str]) -> str:
    preview = []
    for sid in evidence_ids[:4]:
        session = sessions_by_id.get(sid)
        if not session:
            continue
        preview.append(
            {
                "session_id": sid,
                "session_index": session.get("session_index"),
                "signal_type": session.get("memory_annotations", {}).get("signal_type"),
                "source_domain": session.get("source_seed", {}).get("source_domain"),
                "messages_preview": message_preview(session.get("messages", []), 820),
            }
        )
    return json.dumps(preview, ensure_ascii=False)


def build_prompt(
    review_row: Dict[str, str],
    key_row: Dict[str, Any],
    sessions_by_user: Dict[str, List[Dict[str, Any]]],
) -> str:
    probe_type = review_row["probe_type"]
    graph = key_row.get("hidden_habit_graph") or {}
    wanted_signals = relevant_signal_types(probe_type)
    candidate_sessions = []
    current_ids = set(key_row.get("gold_evidence_session_ids") or [])
    for session in sessions_by_user.get(review_row["user_id"], []):
        ann = session.get("memory_annotations", {})
        signal = ann.get("signal_type")
        if signal not in wanted_signals and session.get("session_id") not in current_ids:
            continue
        candidate_sessions.append(
            {
                "session_id": session.get("session_id"),
                "session_index": session.get("session_index"),
                "signal_type": signal,
                "source_domain": session.get("source_seed", {}).get("source_domain"),
                "evidence_summary": ann.get("evidence_summary"),
                "messages_preview": message_preview(session.get("messages", [])),
            }
        )
    candidate_sessions.sort(key=lambda row: (row.get("signal_type") not in wanted_signals, row.get("session_index") or 0))
    payload = {
        "task": "Repair one flagged planning_defaults multiple-choice probe.",
        "issue_category": classify_issue(review_row.get("reviewer_notes", "")),
        "reviewer_complaint": review_row.get("reviewer_notes"),
        "probe_id": review_row.get("probe_id"),
        "probe_type": probe_type,
        "user_id": review_row.get("user_id"),
        "hidden_habit_graph": graph,
        "original_probe": {
            "query": review_row.get("query"),
            "choices": json.loads(review_row.get("choices_json") or "[]"),
            "correct_choice_id": review_row.get("proposed_gold_choice_id"),
            "gold_action": review_row.get("proposed_gold_action"),
            "gold_evidence_session_ids": key_row.get("gold_evidence_session_ids"),
        },
        "candidate_history_sessions": candidate_sessions[:8],
        "output_schema": {
            "probe_id": review_row.get("probe_id"),
            "repair_decision": "repaired|needs_manual",
            "repair_category": "api_review_failure|weak_distractors_or_leaky_wording|weak_or_misaligned_evidence|non_unique_gold|general_probe_quality",
            "query": "public query text",
            "choices": [
                {"choice_id": "A", "text": "choice text"},
                {"choice_id": "B", "text": "choice text"},
                {"choice_id": "C", "text": "choice text"},
                {"choice_id": "D", "text": "choice text"},
            ],
            "correct_choice_id": "A|B|C|D",
            "gold_action": review_row.get("proposed_gold_action"),
            "gold_evidence_session_ids": ["session ids from candidate_history_sessions"],
            "private_repair_notes": "what changed and why",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_curl_json(base_url: str, api_key: str, model: str, prompt: str, timeout: int, retries: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 2200,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    payload = json.dumps(body, ensure_ascii=False)
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "--http1.1",
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout),
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                url,
            ],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                if "error" not in data:
                    return data
                last = json.dumps(data["error"], ensure_ascii=False)
            except Exception as exc:
                last = f"bad json: {type(exc).__name__}: {proc.stdout[:500]}"
        else:
            last = f"curl exit {proc.returncode}: {proc.stderr[:500]} {proc.stdout[:300]}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown curl error")


def parse_repair(raw: Dict[str, Any], row: Dict[str, str], key: Dict[str, Any], sessions_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    content = raw["choices"][0]["message"]["content"]
    data = json.loads(content)
    decision = str(data.get("repair_decision", "")).strip()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"invalid repair_decision: {decision}")
    if data.get("probe_id") and data["probe_id"] != row["probe_id"]:
        raise ValueError("probe_id mismatch")
    choices = data.get("choices") or []
    if len(choices) != 4:
        raise ValueError("expected exactly 4 choices")
    labels = [str(c.get("choice_id", "")).strip() for c in choices]
    if labels != ["A", "B", "C", "D"]:
        raise ValueError(f"choices must be A/B/C/D in order, got {labels}")
    texts = [compact(c.get("text", ""), 900) for c in choices]
    if any(not text for text in texts) or len(set(texts)) != 4:
        raise ValueError("choice texts must be nonempty and unique")
    correct = str(data.get("correct_choice_id", "")).strip()
    if correct not in labels:
        raise ValueError("correct choice missing")
    query = compact(data.get("query", ""), 1200)
    if not query:
        raise ValueError("empty query")
    public_text = " ".join([query, *texts]).lower()
    hits = sorted(word for word in BANNED_PUBLIC_WORDS if word in public_text)
    if hits:
        raise ValueError(f"banned public words: {hits}")
    evidence_ids = [str(sid) for sid in data.get("gold_evidence_session_ids") or key.get("gold_evidence_session_ids") or []]
    missing = [sid for sid in evidence_ids if sid not in sessions_by_id]
    if missing:
        raise ValueError(f"unknown evidence ids: {missing}")
    return {
        "probe_id": row["probe_id"],
        "public_probe_id": row["public_probe_id"],
        "user_id": row["user_id"],
        "probe_type": row["probe_type"],
        "repair_decision": decision,
        "repair_category": data.get("repair_category") or classify_issue(row.get("reviewer_notes", "")),
        "query": query,
        "choices": [{"choice_id": labels[i], "text": texts[i]} for i in range(4)],
        "correct_choice_id": correct,
        "gold_action": data.get("gold_action") or row.get("proposed_gold_action"),
        "gold_evidence_session_ids": evidence_ids,
        "original_reviewer_notes": row.get("reviewer_notes", ""),
        "private_repair_notes": data.get("private_repair_notes", ""),
    }


def public_probe_id(private_id: str) -> str:
    import hashlib

    return "taskmaster_planning_v02_probe_" + hashlib.sha256(private_id.encode("utf-8")).hexdigest()[:16]


def patch_artifacts(args: argparse.Namespace, repairs: Sequence[Dict[str, Any]]) -> None:
    repair_by_probe = {row["probe_id"]: row for row in repairs if row.get("repair_decision") == "repaired"}
    probe_rows = read_jsonl(args.public_probes)
    key_rows = read_jsonl(args.private_probe_key)
    review_rows = read_csv(args.review_csv)
    sessions = read_jsonl(args.sessions_jsonl)
    sessions_by_id = {row["session_id"]: row for row in sessions}

    public_id_to_private = {row["public_probe_id"]: row["probe_id"] for row in key_rows}
    patched_public = []
    for probe in probe_rows:
        private_id = public_id_to_private.get(probe["probe_id"])
        repair = repair_by_probe.get(private_id)
        if repair:
            probe = dict(probe)
            probe["query"] = repair["query"]
            probe["choices"] = repair["choices"]
            probe["metadata"] = dict(probe.get("metadata", {}))
            probe["metadata"]["repair_status"] = "gpt55_xhigh_repair_candidate"
        patched_public.append(probe)

    patched_keys = []
    for key in key_rows:
        repair = repair_by_probe.get(key["probe_id"])
        if repair:
            key = dict(key)
            key["gold_choice_id"] = repair["correct_choice_id"]
            key["gold_action"] = repair["gold_action"]
            key["gold_evidence_session_ids"] = repair["gold_evidence_session_ids"]
            key["review_status"] = "taskmaster_planning_defaults_v02_gpt55_xhigh_repaired_needs_human_review"
            key["repair_notes"] = repair["private_repair_notes"]
        patched_keys.append(key)

    patched_review = []
    for row in review_rows:
        repair = repair_by_probe.get(row["probe_id"])
        if repair:
            row = dict(row)
            row["query"] = repair["query"]
            row["choices_json"] = json.dumps(repair["choices"], ensure_ascii=False)
            row["proposed_gold_choice_id"] = repair["correct_choice_id"]
            row["proposed_gold_action"] = repair["gold_action"]
            row["evidence_preview_json"] = evidence_preview_for_review(sessions_by_id, repair["gold_evidence_session_ids"])
            row["reviewer_decision"] = ""
            row["reviewer_notes"] = f"repair_candidate: {repair['private_repair_notes']}"
        patched_review.append(row)

    write_jsonl(args.output_public_probes, patched_public)
    write_jsonl(args.output_private_probe_key, patched_keys)
    write_csv(args.output_review_csv, patched_review)


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL") or args.base_url
    if not api_key:
        raise SystemExit("Missing HABITBENCH_API_KEY")

    review_rows = read_csv(args.review_csv)
    revise_rows = [row for row in review_rows if row.get("reviewer_decision") == "revise"]
    if args.limit is not None:
        revise_rows = revise_rows[: args.limit]
    key_by_probe = {row["probe_id"]: row for row in read_jsonl(args.private_probe_key)}
    sessions = read_jsonl(args.sessions_jsonl)
    sessions_by_id = {row["session_id"]: row for row in sessions}
    sessions_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[session["user_id"]].append(session)
    for rows in sessions_by_user.values():
        rows.sort(key=lambda row: row.get("session_index", 0))

    existing = {}
    if args.resume and args.output_repairs_jsonl.exists():
        for row in read_jsonl(args.output_repairs_jsonl):
            existing[row["probe_id"]] = row

    repairs: List[Dict[str, Any]] = []
    counts = Counter()
    started = time.time()
    for idx, row in enumerate(revise_rows, 1):
        if row["probe_id"] in existing:
            repair = existing[row["probe_id"]]
            repairs.append(repair)
            counts[repair.get("repair_decision", "unknown")] += 1
            continue
        key = key_by_probe[row["probe_id"]]
        try:
            prompt = build_prompt(row, key, sessions_by_user)
            response = call_curl_json(base_url, api_key, args.model, prompt, args.timeout_sec, args.max_retries)
            repair = parse_repair(response, row, key, sessions_by_id)
        except Exception as exc:
            repair = {
                "probe_id": row["probe_id"],
                "public_probe_id": row.get("public_probe_id"),
                "user_id": row.get("user_id"),
                "probe_type": row.get("probe_type"),
                "repair_decision": "needs_manual",
                "repair_category": classify_issue(row.get("reviewer_notes", "")),
                "original_reviewer_notes": row.get("reviewer_notes", ""),
                "private_repair_notes": f"repair failed: {type(exc).__name__}: {str(exc)[:300]}",
            }
        repairs.append(repair)
        counts[repair["repair_decision"]] += 1
        write_jsonl(args.output_repairs_jsonl, repairs)
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"processed": idx, "counts": dict(counts)}, ensure_ascii=False), flush=True)

    patch_artifacts(args, repairs)
    summary = {
        "rows": len(repairs),
        "counts": dict(counts),
        "model": args.model,
        "reasoning_effort": "xhigh",
        "elapsed_sec": round(time.time() - started, 3),
        "output_repairs_jsonl": str(args.output_repairs_jsonl),
        "output_public_probes": str(args.output_public_probes),
        "output_private_probe_key": str(args.output_private_probe_key),
        "output_review_csv": str(args.output_review_csv),
    }
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    base = Path("/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=base / "review/planning_defaults_review_queue_all_model_labeled_gpt55_xhigh.csv")
    parser.add_argument("--public-probes", type=Path, default=base / "public/probes.jsonl")
    parser.add_argument("--private-probe-key", type=Path, default=base / "private/probe_key.jsonl")
    parser.add_argument("--sessions-jsonl", type=Path, default=base / "private/sessions_with_annotations.jsonl")
    parser.add_argument("--output-repairs-jsonl", type=Path, default=base / "review/planning_defaults_repair_19_gpt55_xhigh.jsonl")
    parser.add_argument("--output-summary-json", type=Path, default=base / "reports/repair_19_gpt55_xhigh_summary.json")
    parser.add_argument("--output-public-probes", type=Path, default=base / "public/probes_repaired19_candidate.jsonl")
    parser.add_argument("--output-private-probe-key", type=Path, default=base / "private/probe_key_repaired19_candidate.jsonl")
    parser.add_argument("--output-review-csv", type=Path, default=base / "review/planning_defaults_review_queue_all_repaired19_candidate.csv")
    parser.add_argument("--base-url", default="https://queqiao.online/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
