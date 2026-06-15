#!/usr/bin/env python
"""HABIT-Bench adapter for Mem0's LLM-backed official memory path.

Unlike `official_mem0_adapter.py`, this adapter calls `Memory.add(...,
infer=True)` so Mem0 performs its official fact extraction/update logic through
the configured LLM. It is intended for the Lumia full official subset with a
local OpenAI-compatible model endpoint.

The answer head remains the shared HABIT-Bench lexical multiple-choice scorer
over Mem0 retrieval results, so the full-path claim is about Mem0's write/update
and retrieval interface, not an end-to-end agent response policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
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


def token_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / max(na * nb, 1e-9)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_of_messages(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def mem0_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    """Preserve role structure while adding an auditable session marker."""
    rows = [{"role": "system", "content": f"SESSION_ID={session['session_id']}"}]
    rows.extend({"role": m["role"], "content": m["content"]} for m in session["messages"])
    return rows


def visible_sessions(probe: Dict[str, Any], sessions_by_user: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    max_idx = probe["visible_history_scope"]["max_session_index"]
    return [
        session
        for session in sessions_by_user[probe["user_id"]]
        if session["session_index"] <= max_idx
    ]


def extract_session_ids(text: str) -> List[str]:
    seen = []
    for sid in re.findall(r"(?:SESSION_ID=|\[SESSION_ID=)([a-zA-Z0-9_\-]+)", text):
        if sid not in seen:
            seen.append(sid)
    return seen


def score_choices(probe: Dict[str, Any], retrieved_text: str) -> Dict[str, float]:
    context_vec = token_counter(retrieved_text + "\n" + probe["query"])
    return {
        choice["choice_id"]: cosine_counter(context_vec, token_counter(choice["text"]))
        for choice in probe["choices"]
    }


def pick_choice(probe: Dict[str, Any], scores: Dict[str, float]) -> str:
    return max(
        probe["choices"],
        key=lambda choice: (scores.get(choice["choice_id"], 0.0), -ord(choice["choice_id"][0])),
    )["choice_id"]


def build_mem0_full_config(args: argparse.Namespace, base_dir: Path, collection_name: str) -> Dict[str, Any]:
    llm_config = {
        "model": args.llm_model,
        "temperature": args.temperature,
        "api_key": args.openai_api_key,
        "openai_base_url": args.openai_base_url,
        "max_tokens": args.max_tokens,
    }
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(base_dir / "qdrant"),
                "collection_name": collection_name,
                "embedding_model_dims": args.embedding_dims,
                "on_disk": False,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": args.embedding_model_name,
                "embedding_dims": args.embedding_dims,
                "model_kwargs": {"device": args.embedding_device},
            },
        },
        "llm": {
            "provider": "openai",
            "config": llm_config,
        },
        "history_db_path": str(base_dir / "history.db"),
        "custom_instructions": (
            "Extract durable user preferences, habits, exceptions, boundaries, "
            "temporal changes, and consent constraints. Preserve session ids when "
            "useful for later evidence."
        ),
    }


def add_sessions_full(memory, sessions_by_user: Dict[str, List[Dict[str, Any]]], args: argparse.Namespace) -> Dict[str, Any]:
    stored = 0
    failures = []
    started = time.time()
    attempted = 0
    total = sum(len(sessions) for sessions in sessions_by_user.values())
    for user_id, sessions in sorted(sessions_by_user.items()):
        for session in sessions:
            attempted += 1
            try:
                memory.add(
                    mem0_messages(session),
                    user_id=user_id,
                    metadata={
                        "session_id": session["session_id"],
                        "session_index": session["session_index"],
                        "domain": session.get("domain", "unknown"),
                    },
                    infer=True,
                )
                stored += 1
                if args.progress_every and (stored == 1 or stored % args.progress_every == 0 or attempted == total):
                    elapsed = time.time() - started
                    print(
                        f"mem0_add_progress stored={stored} attempted={attempted} total={total} elapsed_sec={elapsed:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "user_id": user_id,
                        "session_id": session["session_id"],
                        "error": repr(exc),
                    }
                )
                if not args.continue_on_add_error:
                    raise
    return {
        "stored_sessions_attempted": sum(len(sessions) for sessions in sessions_by_user.values()),
        "stored_sessions_succeeded": stored,
        "add_failures": failures[:20],
        "add_failure_count": len(failures),
        "add_elapsed_sec": round(time.time() - started, 3),
    }


def search_visible_memories(memory, probe: Dict[str, Any], topk: int, threshold: float, rerank: bool) -> List[Dict[str, Any]]:
    return memory.search(
        probe["query"],
        filters={
            "user_id": probe["user_id"],
            "session_index": {"lte": probe["visible_history_scope"]["max_session_index"]},
        },
        top_k=topk,
        threshold=threshold,
        rerank=rerank,
    ).get("results", [])


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    store_dir = args.output.parent / "mem0_full_llm_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    config = build_mem0_full_config(args, store_dir, args.collection_name)
    (args.output.parent / "mem0_full_llm_config.json").write_text(
        json.dumps(
            {
                **config,
                "llm": {
                    "provider": config["llm"]["provider"],
                    "config": {
                        **config["llm"]["config"],
                        "api_key": "<redacted>" if config["llm"]["config"].get("api_key") else None,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.dry_run_config:
        write_jsonl(
            args.output,
            [
                {
                    "probe_id": probe["probe_id"],
                    "choice_id": probe["choices"][0]["choice_id"],
                    "scores": {},
                    "evidence_session_ids": [],
                    "debug": {
                        "adapter": "official_mem0_full_llm_openai",
                        "dry_run_config": True,
                        "llm_model": args.llm_model,
                        "embedding_dims": args.embedding_dims,
                        "embedding_model_name": args.embedding_model_name,
                        "openai_base_url": args.openai_base_url,
                    },
                    "cost": {},
                }
                for probe in payload["probes"]
            ],
        )
        (args.output.parent / "mem0_full_llm_runtime.json").write_text(
            json.dumps(
                {
                    "dry_run_config": True,
                    "memory_add_infer": True,
                    "total_predictions": len(payload["probes"]),
                    "note": "Config generated without creating Memory or calling the LLM endpoint.",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return

    from mem0 import Memory

    memory = Memory.from_config(config)
    add_stats = add_sessions_full(memory, payload["sessions_by_user"], args)

    predictions = []
    search_started = time.time()
    for probe in payload["probes"]:
        results = search_visible_memories(memory, probe, args.topk, args.threshold, args.rerank)
        retrieved_text = "\n\n".join(result.get("memory", "") for result in results)
        scores = score_choices(probe, retrieved_text)
        evidence_session_ids = [
            result.get("metadata", {}).get("session_id")
            for result in results
            if result.get("metadata", {}).get("session_id")
        ]
        if not evidence_session_ids:
            evidence_session_ids = extract_session_ids(retrieved_text)
        sessions = visible_sessions(probe, payload["sessions_by_user"])
        predictions.append(
            {
                "probe_id": probe["probe_id"],
                "choice_id": pick_choice(probe, scores),
                "scores": scores,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "official_mem0_full_llm_openai",
                    "llm_model": args.llm_model,
                    "openai_base_url": args.openai_base_url,
                    "retrieved_scores": [result.get("score") for result in results],
                    "retrieved_text_preview": retrieved_text[:500],
                    "add_stats": add_stats,
                },
                "cost": {
                    "visible_history_sessions": len(sessions),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in sessions
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": len(tokenize(retrieved_text)),
                    "stored_items_est": add_stats["stored_sessions_succeeded"],
                },
            }
        )
    (args.output.parent / "mem0_full_llm_runtime.json").write_text(
        json.dumps(
                {
                    "add_stats": add_stats,
                    "memory_add_infer": True,
                    "search_elapsed_sec": round(time.time() - search_started, 3),
                    "total_predictions": len(predictions),
                },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    try:
        memory.vector_store.client.close()
    except Exception:
        pass
    try:
        memory._telemetry_vector_store.client.close()
    except Exception:
        pass
    write_jsonl(args.output, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--collection-name", default="habit_bench_mem0_full_llm")
    parser.add_argument("--embedding-model-name", default=os.getenv("HABITBENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("HABITBENCH_EMBED_DIMS", "384")))
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cpu"))
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_SERVED_MODEL", os.getenv("HABITBENCH_LLM_MODEL", "habitbench-open-llm")))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HABITBENCH_MEMORY_LLM_MAX_TOKENS", "256")))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--continue-on-add-error", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
