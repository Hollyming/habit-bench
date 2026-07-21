#!/usr/bin/env python
"""Memory-context adapter for the official A-MEM repository.

The adapter uses A-MEM's official `AgenticMemorySystem.add_note` and
`search_agentic` path with LLM metadata extraction, linking, and memory
evolution enabled. It outputs retrieved memories and never selects a choice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "user",
    "with",
    "without",
}


def tokenize(text: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    ]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_of_messages(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def session_text(session: Dict[str, Any]) -> str:
    return f"[SESSION_ID={session['session_id']}]\n" + text_of_messages(session["messages"])


def extract_session_ids(text: str) -> List[str]:
    seen = []
    for sid in re.findall(r"\[SESSION_ID=([^\]]+)\]", text):
        if sid not in seen:
            seen.append(sid)
    return seen


def visible_sessions(probe: Dict[str, Any], sessions_by_user: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    max_idx = probe["visible_history_scope"]["max_session_index"]
    return [
        session
        for session in sessions_by_user[probe["user_id"]]
        if session["session_index"] <= max_idx
    ]


def default_repo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "official-baselines" / "repos" / "a-mem"


def import_amem(repo_path: Path):
    if not repo_path.exists():
        raise FileNotFoundError(f"A-MEM repository not found: {repo_path}")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    sys.path.insert(0, str(repo_path))
    from agentic_memory.memory_system import AgenticMemorySystem

    return AgenticMemorySystem


def make_memory_system(AgenticMemorySystem, args: argparse.Namespace):
    return AgenticMemorySystem(
        model_name=args.model_name,
        llm_backend="openai",
        llm_model=args.llm_model,
        evo_threshold=args.evo_threshold,
        api_key=args.openai_api_key,
    )


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = Path(os.environ.get("AMEM_REPO", "")) if os.environ.get("AMEM_REPO") else args.repo_path
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    config_record = {
        "adapter": "amem_official_code",
        "repo_path": str(repo_path),
        "embedding_model": args.model_name,
        "memory_llm": {
            "model": args.llm_model,
            "base_url": args.openai_base_url,
            "api_key": "<redacted>" if args.openai_api_key else None,
            "evo_threshold": args.evo_threshold,
            "process_memory_enabled": True,
        },
        "topk": args.topk,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "amem_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.dry_run_config:
        write_jsonl(
            args.output,
            [
                {
                    "probe_id": probe["probe_id"],
                    "memory_context": "",
                    "evidence_session_ids": [],
                    "debug": {"dry_run_config": True, **config_record},
                    "cost": {},
                }
                for probe in payload["probes"]
            ],
        )
        return

    AgenticMemorySystem = import_amem(repo_path)
    probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)
    for probes in probes_by_user.values():
        probes.sort(key=lambda p: (p["visible_history_scope"]["max_session_index"], p["probe_id"]))

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    started = time.time()
    sessions_added = 0
    for user_id, probes in sorted(probes_by_user.items()):
        sessions = payload["sessions_by_user"][user_id]
        memory_system = make_memory_system(AgenticMemorySystem, args)
        next_session_pos = 0
        added_until_session_index = -1

        for probe in probes:
            max_idx = probe["visible_history_scope"]["max_session_index"]
            while next_session_pos < len(sessions) and sessions[next_session_pos]["session_index"] <= max_idx:
                session = sessions[next_session_pos]
                memory_system.add_note(
                    session_text(session),
                    time=str(session["session_index"]),
                    category=session.get("domain", "unknown"),
                    tags=[session.get("domain", "unknown"), f"session_{session['session_index']}"],
                )
                added_until_session_index = session["session_index"]
                next_session_pos += 1
                sessions_added += 1
                if args.progress_every and sessions_added % args.progress_every == 0:
                    print(
                        f"amem_add_progress sessions={sessions_added} elapsed_sec={time.time() - started:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )

            results = memory_system.search_agentic(probe["query"], k=args.topk)
            retrieved_text = "\n\n".join(result.get("content", "") for result in results)
            evidence_session_ids = extract_session_ids(retrieved_text)
            sessions_visible = visible_sessions(probe, payload["sessions_by_user"])
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "memory_context": retrieved_text,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "amem_official_code",
                    "official_repo_path": str(repo_path),
                    "retrieved_scores": [result.get("score") for result in results],
                    "retrieved_text_preview": retrieved_text[:500],
                    "added_until_session_index": added_until_session_index,
                },
                "cost": {
                    "visible_history_sessions": len(sessions_visible),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in sessions_visible
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": len(tokenize(retrieved_text)),
                    "stored_items_est": len(memory_system.memories),
                },
            }

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)
    (args.output.parent / "amem_runtime.json").write_text(
        json.dumps(
            {
                "elapsed_sec": round(time.time() - started, 3),
                "sessions_added": sessions_added,
                "total_predictions": len(predictions),
                "process_memory_enabled": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=default_repo_path())
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model-name", default=os.getenv("HABITBENCH_EMBED_MODEL", "/home/jmzhang/models/e5-base-v2"))
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_SERVED_MODEL", "habitbench-qwen3-8b"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--evo-threshold", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
