#!/usr/bin/env python
"""Model-label HABIT-Bench planning_defaults review queues.

This script calls an OpenAI-compatible `/chat/completions` API and writes a new
review CSV with `reviewer_decision` and `reviewer_notes` filled in. It never
stores API credentials; pass them through environment variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


VALID_DECISIONS = {"accept", "revise", "reject"}


SYSTEM_PROMPT = """You are a strict senior reviewer for HABIT-Bench.

Review a candidate planning_defaults benchmark row. The family tests whether an
agent can infer a scoped planning default from long user history and apply it
only when the current request matches the hidden habit graph.

You will receive:
- the public probe query and four choices;
- the proposed gold choice/action;
- a private evidence preview from the user's history;
- when available, the hidden habit graph with condition, default_action,
  boundary_condition, and exception_condition.

Review against the hidden graph, not against any single fixed travel preference.

Decide:
- accept: sample is directly usable; seed/evidence is coherent, scope is clear,
  gold choice is unique, and wording does not make the answer trivial.
- revise: sample is broadly usable but needs wording/gold/choice/evidence cleanup.
- reject: sample is fundamentally unsuitable; wrong domain, unsupported habit,
  non-unique/no correct gold, severe leakage, or incoherent evidence.

Specific checks:
- direct_use should require applying the scoped default_action in an in-scope
  current request.
- boundary should reward not applying the default outside the condition.
- exception should reward following the explicit current-trip override.
- explicit_retrieval may ask for the remembered planning preference, but the
  answer should still be supported by history.
- Distractors should be plausible travel-planning tradeoffs, not absurd
  nonanswers or obviously wrong meta-statements.
- The public query and choices should not expose private terms such as habit,
  benchmark, gold, label, evidence, annotation, or dataset.

Return strict JSON only:
{"reviewer_decision":"accept|revise|reject","reviewer_notes":"<decision>: concise reason"}

Notes must be one sentence, start with the decision label and colon, and mention
the main reason."""


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_json_field(value: str, max_chars: int) -> str:
    try:
        parsed = json.loads(value)
        text = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        text = value or ""
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def row_prompt(row: Dict[str, str], key_by_probe: Dict[str, Dict[str, Any]] | None = None) -> str:
    key = (key_by_probe or {}).get(row.get("probe_id", ""))
    payload = {
        "review_id": row.get("review_id"),
        "probe_type": row.get("probe_type"),
        "habit_family": row.get("habit_family"),
        "stress_variant": row.get("stress_variant"),
        "query": row.get("query"),
        "choices": json.loads(row.get("choices_json") or "[]"),
        "proposed_gold_choice_id": row.get("proposed_gold_choice_id"),
        "proposed_gold_action": row.get("proposed_gold_action"),
        "evidence_preview": json.loads(row.get("evidence_preview_json") or "[]"),
    }
    if key:
        payload["hidden_habit_graph"] = key.get("hidden_habit_graph")
        payload["gold_evidence_session_ids"] = key.get("gold_evidence_session_ids")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > 9000:
        payload["evidence_preview"] = compact_json_field(row.get("evidence_preview_json", ""), 4200)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    return text


def parse_json_response(content: str) -> Tuple[str, str, Dict[str, Any]]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    data = json.loads(raw)
    decision = str(data.get("reviewer_decision", "")).strip().lower()
    notes = str(data.get("reviewer_notes", "")).strip()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"invalid reviewer_decision: {decision!r}")
    if not notes:
        notes = f"{decision}: model returned no reason"
    if not notes.lower().startswith(f"{decision}:"):
        notes = f"{decision}: {notes}"
    return decision, notes, data


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTPError {exc.code}: {body_text}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error or "unknown API error")


def call_chat_completion_curl(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    payload = json.dumps(body, ensure_ascii=False)
    last_error = None
    for attempt in range(max_retries + 1):
        cmd = [
            "curl",
            "-sS",
            "--http1.1",
            "--connect-timeout",
            str(min(timeout, 30)),
            "--max-time",
            str(timeout),
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ]
        proc = subprocess.run(
            cmd,
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
                last_error = json.dumps(data["error"], ensure_ascii=False)[:500]
            except Exception as exc:
                last_error = f"bad JSON response: {type(exc).__name__}: {proc.stdout[:500]}"
        else:
            last_error = f"curl exit {proc.returncode}: {proc.stderr[:500]} {proc.stdout[:300]}"
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error or "unknown curl API error")


def heuristic_label(row: Dict[str, str]) -> Tuple[str, str]:
    probe_type = row.get("probe_type", "")
    query = (row.get("query") or "").lower()
    choices = row.get("choices_json") or ""
    evidence = row.get("evidence_preview_json") or ""
    lower = f"{query} {choices} {evidence}".lower()

    if "business-travel timing rule" in lower or "established planning preference" in query:
        if probe_type == "direct_use":
            return (
                "revise",
                "revise: evidence supports the business-travel buffer, but query/choices reveal the preference too directly",
            )
        if probe_type in {"boundary", "exception"}:
            return (
                "revise",
                "revise: scope is clear and gold is plausible, but wording names the business-travel rule too explicitly",
            )

    if probe_type == "explicit_retrieval":
        if "early arrivals" in lower and "90-minute buffer" in lower:
            return (
                "accept",
                "accept: explicit retrieval control has repeated support evidence and a unique gold preference",
            )
        return (
            "revise",
            "revise: explicit retrieval intent is valid, but evidence/gold wording should be checked manually",
        )

    if probe_type == "direct_use" and "90-minute buffer" in lower and "meeting" in lower:
        return (
            "accept",
            "accept: support evidence establishes the business-travel buffer and the direct-use gold is unique",
        )
    if probe_type == "boundary" and ("leisure" in lower or "weekend" in lower):
        return (
            "accept",
            "accept: boundary context is outside business travel and the non-personalized gold is unique",
        )
    if probe_type == "exception" and ("flexible" in lower or "personal" in lower):
        return (
            "accept",
            "accept: exception context is explicit and the gold correctly avoids applying the usual business default",
        )
    return (
        "revise",
        "revise: sample appears usable but needs manual review for evidence support and gold uniqueness",
    )


def infer_model(base_url: str, api_key: str, timeout: int) -> str:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError("/models returned no models; pass --model explicitly")
    return str(models[0].get("id") or models[0].get("name"))


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL") or args.base_url
    model = args.model or os.environ.get("HABITBENCH_LABEL_MODEL")
    if not args.offline_heuristic:
        if not api_key:
            raise SystemExit("Missing HABITBENCH_API_KEY environment variable")
        if not base_url:
            raise SystemExit("Missing HABITBENCH_BASE_URL or --base-url")
        if not model:
            model = infer_model(base_url, api_key, args.timeout_sec)
    else:
        model = "offline_heuristic"

    rows = read_csv(args.input_csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    key_by_probe = {row["probe_id"]: row for row in read_jsonl(args.private_key_jsonl)} if args.private_key_jsonl else {}
    existing_by_review_id: Dict[str, Dict[str, str]] = {}
    if args.resume and args.output_csv.exists():
        for row in read_csv(args.output_csv):
            if row.get("review_id") and row.get("reviewer_decision"):
                existing_by_review_id[row["review_id"]] = row

    labeled_rows: List[Dict[str, str]] = []
    raw_rows: List[Dict[str, Any]] = []
    counts = Counter()
    started = time.time()

    for idx, row in enumerate(rows, start=1):
        if args.resume and row.get("review_id") in existing_by_review_id:
            existing = existing_by_review_id[row["review_id"]]
            labeled_rows.append(existing)
            counts[existing.get("reviewer_decision", "unknown")] += 1
            continue
        prompt = row_prompt(row, key_by_probe)
        try:
            if args.offline_heuristic:
                decision, notes = heuristic_label(row)
                parsed = {"mode": "offline_heuristic"}
            else:
                caller = call_chat_completion_curl if args.transport == "curl" else call_chat_completion
                response = caller(
                    base_url=base_url,
                    api_key=api_key or "",
                    model=model or "",
                    user_prompt=prompt,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout_sec,
                    max_retries=args.max_retries,
                )
                content = response["choices"][0]["message"]["content"]
                decision, notes, parsed = parse_json_response(content)
            row = dict(row)
            row["reviewer_decision"] = decision
            row["reviewer_notes"] = notes
            counts[decision] += 1
            raw_rows.append(
                {
                    "review_id": row.get("review_id"),
                    "decision": decision,
                    "notes": notes,
                    "model": model,
                    "raw_response": parsed,
                }
            )
        except Exception as exc:
            row = dict(row)
            row["reviewer_decision"] = "revise"
            row["reviewer_notes"] = f"revise: model labeling failed; manually review ({type(exc).__name__})"
            counts["revise"] += 1
            counts["api_or_parse_error"] += 1
            raw_rows.append(
                {
                    "review_id": row.get("review_id"),
                    "decision": row["reviewer_decision"],
                    "notes": row["reviewer_notes"],
                    "model": model,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )
        labeled_rows.append(row)
        if args.incremental_write:
            write_csv(args.output_csv, labeled_rows)
            if args.raw_output_jsonl:
                args.raw_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.raw_output_jsonl.open("w", encoding="utf-8", newline="\n") as f:
                    for raw in raw_rows:
                        f.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"labeled": idx, "counts": dict(counts)}, ensure_ascii=False), flush=True)

    write_csv(args.output_csv, labeled_rows)
    if args.raw_output_jsonl:
        args.raw_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_output_jsonl.open("w", encoding="utf-8", newline="\n") as f:
            for raw in raw_rows:
                f.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "raw_output_jsonl": str(args.raw_output_jsonl) if args.raw_output_jsonl else None,
        "model": model,
        "rows": len(labeled_rows),
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_base = Path("/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_base / "review" / "planning_defaults_review_queue_all.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_base / "review" / "planning_defaults_review_queue_all_model_labeled.csv",
    )
    parser.add_argument(
        "--raw-output-jsonl",
        type=Path,
        default=default_base / "review" / "planning_defaults_review_queue_all_model_labeled_raw.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=default_base / "reports" / "model_label_summary.json",
    )
    parser.add_argument("--private-key-jsonl", type=Path, default=None)
    parser.add_argument("--base-url", default="https://queqiao.online/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--incremental-write", action="store_true", default=True)
    parser.add_argument(
        "--offline-heuristic",
        action="store_true",
        help="Do not call an API; produce clearly marked heuristic prelabels for manual review.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
