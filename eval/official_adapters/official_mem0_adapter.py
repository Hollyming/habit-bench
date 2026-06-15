#!/usr/bin/env python
"""HABIT-Bench adapter for the official Mem0 Python package.

This adapter uses the official Mem0 OSS API for memory storage and retrieval:
`Memory.add(..., infer=False)` and `Memory.search(...)`. LLM fact extraction is
disabled to keep the run deterministic and affordable; session text is stored
as raw memory with local HuggingFace embeddings and local Qdrant. The answer
head is the same simple HABIT-Bench lexical choice scorer used by other
retrieval adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
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


def session_text(session: Dict[str, Any]) -> str:
    return f"[SESSION_ID={session['session_id']}]\n" + text_of_messages(session["messages"])


def visible_sessions(probe: Dict[str, Any], sessions_by_user: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    max_idx = probe["visible_history_scope"]["max_session_index"]
    return [
        session
        for session in sessions_by_user[probe["user_id"]]
        if session["session_index"] <= max_idx
    ]


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


def build_mem0_config(args: argparse.Namespace, base_dir: Path, collection_name: str) -> Dict[str, Any]:
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
            "provider": "ollama",
            "config": {"model": "llama3.2"},
        },
        "history_db_path": str(base_dir / "history.db"),
    }


def add_sessions(memory, sessions_by_user: Dict[str, List[Dict[str, Any]]]) -> int:
    stored = 0
    for user_id, sessions in sorted(sessions_by_user.items()):
        for session in sessions:
            memory.add(
                session_text(session),
                user_id=user_id,
                metadata={
                    "session_id": session["session_id"],
                    "session_index": session["session_index"],
                    "domain": session.get("domain", "unknown"),
                },
                infer=False,
            )
            stored += 1
    return stored


def search_visible_memories(memory, probe: Dict[str, Any], topk: int, threshold: float) -> List[Dict[str, Any]]:
    return memory.search(
        probe["query"],
        filters={
            "user_id": probe["user_id"],
            "session_index": {"lte": probe["visible_history_scope"]["max_session_index"]},
        },
        top_k=topk,
        threshold=threshold,
    ).get("results", [])


def run(args: argparse.Namespace) -> None:
    from mem0 import Memory

    payload = read_json(args.input)
    store_dir = args.output.parent / "mem0_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory.from_config(build_mem0_config(args, store_dir, args.collection_name))
    total_stored = add_sessions(memory, payload["sessions_by_user"])

    predictions = []
    for probe in payload["probes"]:
        results = search_visible_memories(memory, probe, args.topk, args.threshold)
        retrieved_text = "\n\n".join(result.get("memory", "") for result in results)
        scores = score_choices(probe, retrieved_text)
        evidence_session_ids = [
            result.get("metadata", {}).get("session_id")
            for result in results
            if result.get("metadata", {}).get("session_id")
        ]
        sessions = visible_sessions(probe, payload["sessions_by_user"])
        predictions.append(
            {
                "probe_id": probe["probe_id"],
                "choice_id": pick_choice(probe, scores),
                "scores": scores,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "official_mem0_infer_false_hf_qdrant",
                    "retrieved_scores": [result.get("score") for result in results],
                    "retrieved_text_preview": retrieved_text[:500],
                    "total_stored_sessions": total_stored,
                },
                "cost": {
                    "visible_history_sessions": len(sessions),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in sessions
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": len(tokenize(retrieved_text)),
                    "stored_items_est": len(sessions),
                },
            }
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
    parser.add_argument("--collection-name", default="habit_bench_mem0")
    parser.add_argument("--embedding-model-name", default=os.getenv("HABITBENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("HABITBENCH_EMBED_DIMS", "384")))
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cpu"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
