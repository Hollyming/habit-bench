#!/usr/bin/env python3
"""Incrementally repair stale option-letter prose after final choice balancing.

Full-corpus structural checks are deterministic and free. GPT-5.5 xhigh is
called only for probes whose private prose explicitly refers to A/B/C/D and
may therefore describe the pre-balancing layout. Each successful probe is
cached independently; one failed probe never invalidates or repeats the rest.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
REVISION = "v04_incremental_final_choice_prose_repair_r2"
CJK = re.compile(r"[\u3400-\u9fff]")
STALE_LETTER = re.compile(
    r"(?:\b(?i:option|choice)\s+[A-D]\b|"
    r"\b[A-D]\s+(?:and|or|is|was|offers|has|gives|costs|saves|adds|hits|turns|uses|"
    r"would|remains|wins|meets|preserves|fits|trades|matches|clears)\b)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def prose_values(key: dict[str, Any]) -> list[str]:
    values = [
        str(key.get("label_rationale") or ""),
        str(key.get("generator_difficulty_rationale") or ""),
        str(key.get("independent_gold_judge", {}).get("rationale") or ""),
        str(key.get("independent_gold_judge", {}).get("closest_distractor_rationale") or ""),
        str(key.get("query_only_judge", {}).get("rationale") or ""),
        str(key.get("query_only_judge", {}).get("closest_distractor_rationale") or ""),
    ]
    values.extend(str(item) for item in key.get("query_only_judge", {}).get("leakage_signals", []))
    return values


def needs_repair(key: dict[str, Any]) -> bool:
    return any(STALE_LETTER.search(text) for text in prose_values(key))


def user_evidence(session: dict[str, Any], habit_id: str | None) -> dict[str, Any]:
    signals = [
        signal for signal in session.get("memory_annotations", {}).get("verified_habit_signals", [])
        if not habit_id or signal.get("habit_id") == habit_id
    ]
    return {
        "session_id": session["session_id"],
        "session_index": session["session_index"],
        "episode_id": session.get("memory_annotations", {}).get("episode_id"),
        "verified_quotes": [signal.get("evidence_quote") for signal in signals if signal.get("evidence_quote")],
        "user_turns": [
            message["content"] for message in session["messages"]
            if message.get("role") == "user"
        ],
    }


def validate_repair(item: dict[str, Any], row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("probe_id") != item["probe_id"]:
        raise ValueError("wrong probe id")
    fields = [
        "label_rationale", "generator_difficulty_rationale",
        "independent_gold_rationale", "independent_closest_distractor_rationale",
        "query_only_rationale", "query_only_closest_distractor_rationale",
    ]
    for field in fields:
        text = str(row.get(field) or "").strip()
        if not text:
            raise ValueError(f"missing {field}")
        if CJK.search(text):
            raise ValueError(f"{field} is not English-only")
        if STALE_LETTER.search(text):
            raise ValueError(f"{field} still refers to an option letter")
        row[field] = text
    signals = row.get("query_only_leakage_signals")
    if not isinstance(signals, list):
        raise ValueError("query_only_leakage_signals must be a list")
    for signal in signals:
        if CJK.search(str(signal)) or STALE_LETTER.search(str(signal)):
            raise ValueError("invalid query-only leakage signal")
    return row


def call_batch(batch: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    pending = []
    completed: dict[str, dict[str, Any]] = {}
    cache_dir = args.dataset / "work" / "final_choice_prose_repairs"
    for item in batch:
        fingerprint = hashlib.sha256(json.dumps({
            "revision": REVISION, "model": args.model,
            "reasoning_effort": args.reasoning_effort, "item": item,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        item["cache_path"] = str(cache_dir / f"{item['probe_id']}_{fingerprint}.json")
        path = Path(item["cache_path"])
        if path.exists():
            try:
                completed[item["probe_id"]] = validate_repair(item, json.loads(path.read_text(encoding="utf-8"))["repair"])
                continue
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        pending.append(item)
    if not pending:
        return completed, {}

    payload_items = [{k: v for k, v in item.items() if k != "cache_path"} for item in pending]
    payload = {
        "task": "Repair private benchmark explanations so they describe the final public choice layout.",
        "rules": [
            "Do not change the query, option texts, gold choice, closest-distractor IDs, evidence, or scientific judgment.",
            "Rewrite prose only. Identify alternatives by semantic details such as hotel name, carrier, departure time, route, price, or amenity.",
            "Never identify an alternative using a bare A/B/C/D letter or phrases such as option A.",
            "Use English only. Preserve the distinction between query-only ambiguity and history-aware personalization.",
            "Return {repairs:[{probe_id,label_rationale,generator_difficulty_rationale,independent_gold_rationale,independent_closest_distractor_rationale,query_only_rationale,query_only_closest_distractor_rationale,query_only_leakage_signals}]}",
        ],
        "probes": payload_items,
    }
    failures: dict[str, str] = {}
    last_error = None
    for attempt in range(2):
        try:
            raw = post_chat(
                base_url=args.base_url, api_key=args.api_key, model=args.model,
                messages=[
                    {"role": "system", "content": "You are a precise benchmark copy editor. Return strict JSON only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=9000, timeout=args.timeout, retries=args.retries,
                transport=args.transport, reasoning_effort=args.reasoning_effort,
            )["json"]
            rows = raw.get("repairs") if isinstance(raw, dict) else None
            if not isinstance(rows, list):
                raise ValueError("response lacks repairs list")
            by_id = {row.get("probe_id"): row for row in rows if isinstance(row, dict)}
            for item in pending:
                try:
                    repaired = validate_repair(item, by_id.get(item["probe_id"]))
                    completed[item["probe_id"]] = repaired
                    write_json(Path(item["cache_path"]), {"revision": REVISION, "repair": repaired})
                except ValueError as exc:
                    failures[item["probe_id"]] = str(exc)
            if not failures:
                return completed, {}
            pending = [item for item in pending if item["probe_id"] in failures]
            payload["probes"] = [{k: v for k, v in item.items() if k != "cache_path"} for item in pending]
            payload["retry_feedback"] = failures
            failures = {}
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
            payload["retry_feedback"] = last_error
    for item in pending:
        failures[item["probe_id"]] = last_error or "repair remained invalid after two attempts"
    return completed, failures


def backup_once(path: Path, backup: Path) -> None:
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def apply_repair(key: dict[str, Any], repair: dict[str, Any]) -> None:
    key["label_rationale"] = repair["label_rationale"]
    key["generator_difficulty_rationale"] = repair["generator_difficulty_rationale"]
    key.setdefault("independent_gold_judge", {})["rationale"] = repair["independent_gold_rationale"]
    key["independent_gold_judge"]["closest_distractor_rationale"] = repair["independent_closest_distractor_rationale"]
    key.setdefault("query_only_judge", {})["rationale"] = repair["query_only_rationale"]
    key["query_only_judge"]["closest_distractor_rationale"] = repair["query_only_closest_distractor_rationale"]
    key["query_only_judge"]["leakage_signals"] = repair["query_only_leakage_signals"]
    key["choice_layout_prose_repair"] = REVISION
    key["review_status"] = "gpt55_xhigh_choice_layout_prose_repaired"


def repair_manifest(dataset: Path) -> None:
    dossiers = read_jsonl(dataset / "private" / "user_dossiers.jsonl")
    path = dataset / "reports" / "e2e_generation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["target_users"] = len(dossiers)
    manifest["target_sessions_by_user"] = {
        row["user_id"]: row["longitudinal_plan"]["target_sessions"] for row in dossiers
    }
    manifest["metadata_repaired_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    args = parser.parse_args()

    release = args.dataset / "release_single_user_pilot"
    public_path = release / "public" / "probes.jsonl"
    key_path = release / "private" / "probe_key.jsonl"
    public = read_jsonl(public_path)
    keys = read_jsonl(key_path)
    sessions = {
        row["session_id"]: row
        for row in read_jsonl(release / "private" / "sessions_with_annotations.jsonl")
    }
    public_by_id = {row["probe_id"]: row for row in public}
    key_by_id = {row["probe_id"]: row for row in keys}
    if len(public) != len(keys) or set(public_by_id) != set(key_by_id):
        raise SystemExit("public/private probe mismatch")
    if any(CJK.search(json.dumps(row, ensure_ascii=False)) for row in public + keys):
        raise SystemExit("probe files are not English-only")
    for key in keys:
        probe = public_by_id[key["probe_id"]]
        valid = {choice["choice_id"] for choice in probe["choices"]}
        if key.get("gold_choice_id") not in valid:
            raise SystemExit(f"{key['probe_id']}: invalid gold id")
        if any(session_id not in sessions for session_id in key.get("gold_evidence_session_ids", [])):
            raise SystemExit(f"{key['probe_id']}: missing evidence session")

    suspicious = [key for key in keys if needs_repair(key)]
    repair_manifest(args.dataset)
    if not suspicious:
        print(json.dumps({"status": "pass", "repair_candidates": 0, "message": "no stale option-letter prose"}))
        return
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY for suspicious-probe repair")

    items = []
    for key in suspicious:
        probe = public_by_id[key["probe_id"]]
        evidence_ids = key.get("gold_evidence_session_ids", [])
        gold_text = next(choice["text"] for choice in probe["choices"] if choice["choice_id"] == key["gold_choice_id"])
        generator_closest = key.get("generator_closest_distractor_choice_id")
        judge_closest = key.get("independent_gold_judge", {}).get("closest_distractor_choice_id")
        choice_text = {choice["choice_id"]: choice["text"] for choice in probe["choices"]}
        items.append({
            "probe_id": key["probe_id"], "query": probe["query"], "choices": probe["choices"],
            "gold_choice_id": key["gold_choice_id"], "gold_choice_text": gold_text,
            "generator_closest_distractor_text": choice_text.get(generator_closest),
            "independent_closest_distractor_text": choice_text.get(judge_closest),
            "gold_action": key.get("gold_action"), "probe_type": key.get("probe_type"),
            "hidden_habit_graph": key.get("hidden_habit_graph"),
            "existing_prose": {
                "label_rationale": key.get("label_rationale"),
                "generator_difficulty_rationale": key.get("generator_difficulty_rationale"),
                "independent_gold_rationale": key.get("independent_gold_judge", {}).get("rationale"),
                "independent_closest_distractor_rationale": key.get("independent_gold_judge", {}).get("closest_distractor_rationale"),
                "query_only_rationale": key.get("query_only_judge", {}).get("rationale"),
                "query_only_closest_distractor_rationale": key.get("query_only_judge", {}).get("closest_distractor_rationale"),
                "query_only_leakage_signals": key.get("query_only_judge", {}).get("leakage_signals", []),
            },
            "evidence": [user_evidence(sessions[session_id], key.get("habit_id")) for session_id in evidence_ids],
        })
    batches = [items[index:index + args.batch_size] for index in range(0, len(items), args.batch_size)]
    repaired: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(call_batch, batch, args): batch for batch in batches}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            good, bad = future.result()
            repaired.update(good); failures.update(bad)
            print(f"choice_prose_repair_progress {completed}/{len(batches)} repaired={len(repaired)} pending={len(failures)}", flush=True)

    archive = release / "reports" / "pre_choice_layout_prose_repair"
    backup_once(public_path, archive / "probes.jsonl")
    backup_once(key_path, archive / "probe_key.jsonl")
    for key in keys:
        if key["probe_id"] in repaired:
            apply_repair(key, repaired[key["probe_id"]])
    for probe in public:
        if probe["probe_id"] in repaired:
            probe.setdefault("metadata", {})["choice_layout_prose_repair"] = REVISION
    write_jsonl(public_path, public)
    write_jsonl(key_path, keys)

    # Synchronize repaired semantic fields back to the active main tree while
    # preserving release-only packaging metadata.
    main_public_path = args.dataset / "public" / "probes.jsonl"
    main_key_path = args.dataset / "private" / "probe_key.jsonl"
    main_public = read_jsonl(main_public_path)
    main_keys = read_jsonl(main_key_path)
    backup_once(main_public_path, args.dataset / "reports" / "pre_choice_layout_prose_repair" / "probes.jsonl")
    backup_once(main_key_path, args.dataset / "reports" / "pre_choice_layout_prose_repair" / "probe_key.jsonl")
    release_keys = {row["probe_id"]: row for row in keys}
    for row in main_public:
        if row["probe_id"] in repaired:
            row.setdefault("metadata", {})["choice_layout_prose_repair"] = REVISION
    for row in main_keys:
        if row["probe_id"] in repaired:
            source = release_keys[row["probe_id"]]
            for field in [
                "label_rationale", "generator_difficulty_rationale",
                "independent_gold_judge", "query_only_judge",
                "choice_layout_prose_repair", "review_status",
            ]:
                row[field] = copy.deepcopy(source[field])
    write_jsonl(main_public_path, main_public)
    write_jsonl(main_key_path, main_keys)

    remaining = [row["probe_id"] for row in keys if needs_repair(row)]
    summary = {
        "status": "pass" if not remaining else "needs_retry",
        "revision": REVISION, "probe_count": len(keys),
        "english_only": True, "initial_repair_candidates": len(suspicious),
        "repaired_this_run": len(repaired), "remaining_probe_ids": remaining,
        "api_failures": failures,
        "gold_ids_changed": 0, "public_queries_or_choices_changed": 0,
        "difficulty_counts": dict(Counter(
            row.get("independent_gold_judge", {}).get("difficulty") for row in keys
        )),
    }
    write_json(release / "reports" / "choice_layout_prose_repair.json", summary)
    write_json(args.dataset / "reports" / "choice_layout_prose_repair.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if remaining:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
