#!/usr/bin/env python3
"""Generate users in isolated session shards and merge them atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_client import write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
DEFAULT_USERS = ",".join(f"tm_pd_v04_user_{index:03d}" for index in range(1, 6))
GENERATOR = Path(__file__).with_name("04_generate_benchmark.py")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def backup_once(path: Path, backup: Path) -> None:
    if path.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def validate_shard(shard: Path, user_id: str, profile: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sessions = read_jsonl(shard / "private" / "sessions_with_annotations.jsonl")
    target = int(profile["longitudinal_plan"]["target_sessions"])
    if len(sessions) != target:
        raise ValueError(f"{user_id}: shard has {len(sessions)} sessions, expected {target}")
    if {row.get("user_id") for row in sessions} != {user_id}:
        raise ValueError(f"{user_id}: shard contains another user")
    if sorted(int(row["session_index"]) for row in sessions) != list(range(target)):
        raise ValueError(f"{user_id}: non-contiguous session indices")
    if len({row["session_id"] for row in sessions}) != target:
        raise ValueError(f"{user_id}: duplicate session ids")
    evidence = json.loads((shard / "reports" / f"{user_id}_verified_history_evidence.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "pass":
        raise ValueError(f"{user_id}: post-hoc evidence gate did not pass")
    lengths = json.loads((shard / "reports" / "history_length_qwen_tokens.json").read_text(encoding="utf-8"))
    if user_id not in lengths or int(lengths[user_id].get("qwen_tokens", 0)) <= 0:
        raise ValueError(f"{user_id}: missing exact Qwen history length")
    return sessions, lengths[user_id]


def merge_shards(dataset: Path, user_ids: list[str]) -> dict[str, Any]:
    profiles = {row["user_id"]: row for row in read_jsonl(dataset / "private" / "user_dossiers.jsonl")}
    unknown = sorted(set(user_ids) - set(profiles))
    if unknown:
        raise ValueError(f"unknown users: {unknown}")
    selected = set(user_ids)
    merged = [
        row for row in read_jsonl(dataset / "private" / "sessions_with_annotations.jsonl")
        if row.get("user_id") not in selected
    ]
    history_path = dataset / "reports" / "history_length_qwen_tokens.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    for user_id in user_ids:
        shard = dataset / "work" / "session_shards" / user_id
        sessions, length = validate_shard(shard, user_id, profiles[user_id])
        merged.extend(sessions)
        history[user_id] = length
    merged.sort(key=lambda row: (row["user_id"], int(row["session_index"])))
    if len({row["session_id"] for row in merged}) != len(merged):
        raise ValueError("merged output contains duplicate session ids")
    transcript_hashes: dict[str, str] = {}
    for row in merged:
        normalized = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in transcript_hashes:
            raise ValueError(f"exact duplicate transcripts: {transcript_hashes[digest]} and {row['session_id']}")
        transcript_hashes[digest] = row["session_id"]

    private_path = dataset / "private" / "sessions_with_annotations.jsonl"
    public_path = dataset / "public" / "lifelines.jsonl"
    archive = dataset / "reports" / "pre_parallel_session_merge"
    backup_once(private_path, archive / "sessions_with_annotations.jsonl")
    backup_once(public_path, archive / "lifelines.jsonl")
    backup_once(history_path, archive / "history_length_qwen_tokens.json")
    write_jsonl(private_path, merged)
    public = [
        {key: row[key] for key in ["user_id", "session_id", "session_index", "timestamp", "domain", "messages"]}
        for row in merged
    ]
    write_jsonl(public_path, public)
    write_json(history_path, history)

    manifest_path = dataset / "reports" / "e2e_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "stage": "sessions_parallel_merged",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_users": len(profiles),
        "target_sessions_by_user": {
            user_id: profile["longitudinal_plan"]["target_sessions"]
            for user_id, profile in profiles.items()
        },
        "parallel_session_shards": user_ids,
    })
    write_json(manifest_path, manifest)
    return {
        "status": "pass", "merged_users": user_ids, "total_sessions": len(merged),
        "session_counts": {
            user_id: sum(row["user_id"] == user_id for row in merged)
            for user_id in sorted(profiles)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--users", default=DEFAULT_USERS, help="Comma-separated user ids")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--transport", choices=["curl", "curl_stream", "urllib"], default="curl_stream")
    args = parser.parse_args()
    user_ids = [item.strip() for item in args.users.split(",") if item.strip()]
    if not user_ids or len(set(user_ids)) != len(user_ids):
        raise SystemExit("--users must contain unique user ids")
    if not 1 <= args.max_parallel <= len(user_ids):
        raise SystemExit("--max-parallel is outside the requested user count")
    if not (os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL")):
        raise SystemExit("Set HABITBENCH_BASE_URL")
    if not (os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("Set HABITBENCH_API_KEY")

    log_dir = args.dataset / "reports" / "parallel_sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(user_ids)
    running: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    exit_codes: dict[str, int] = {}
    while pending or running:
        while pending and len(running) < args.max_parallel:
            user_id = pending.pop(0)
            shard = args.dataset / "work" / "session_shards" / user_id
            command = [
                sys.executable, str(GENERATOR), "sessions",
                "--dataset", str(args.dataset), "--users", "6", "--only-users", user_id,
                "--session-shard-dir", str(shard), "--timeout", str(args.timeout),
                "--transport", args.transport,
            ]
            handle = (log_dir / f"{user_id}.log").open("a", encoding="utf-8", buffering=1)
            handle.write(f"\nSTART {datetime.now(timezone.utc).isoformat()}\n")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            running[user_id] = (process, handle)
            print(f"parallel_session_started {user_id} pid={process.pid}", flush=True)
        time.sleep(2)
        for user_id, (process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.write(f"END exit_code={code} {datetime.now(timezone.utc).isoformat()}\n")
            handle.close()
            exit_codes[user_id] = code
            del running[user_id]
            print(f"parallel_session_finished {user_id} exit_code={code}", flush=True)
        write_json(log_dir / "status.json", {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pending": pending, "running": sorted(running), "exit_codes": exit_codes,
        })
    failed = {user_id: code for user_id, code in exit_codes.items() if code != 0}
    if failed:
        write_json(log_dir / "merge_status.json", {"status": "not_merged", "failed": failed})
        raise SystemExit(f"session shards failed and were not merged: {failed}")
    result = merge_shards(args.dataset, user_ids)
    write_json(log_dir / "merge_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
