#!/usr/bin/env python
"""HABIT-Bench adapter for the official O-Mem repository.

This adapter uses O-Mem's official `MemoryChain`, `MemoryManager`, and
`retrieve_from_memory_soft_segmentation` retrieval path. It bypasses the
LLM-based message-understanding and persona-update stages by injecting
HABIT-Bench session text into the official working/episodic/persona memory
structures. This is an official-code retrieval adapter, not a full O-Mem
active-profiling reproduction.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import shutil
import sys
import types
from collections import Counter, defaultdict
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


def extract_session_ids(text: str) -> List[str]:
    seen = []
    for sid in re.findall(r"\[SESSION_ID=([^\]]+)\]", text):
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


def default_repo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "official-baselines" / "repos" / "O-Mem"


def import_omem(repo_path: Path):
    if not repo_path.exists():
        raise FileNotFoundError(f"O-Mem repository not found: {repo_path}")

    # O-Mem imports FlagEmbedding unconditionally, but the retrieval path used
    # here relies on the supplied sentence-transformers embedding model.
    if "FlagEmbedding" not in sys.modules:
        flag = types.ModuleType("FlagEmbedding")

        class FlagAutoModel:
            pass

        flag.FlagAutoModel = FlagAutoModel
        sys.modules["FlagEmbedding"] = flag

    sys.path.insert(0, str(repo_path))
    from example_usage import SimpleMemory

    return SimpleMemory


def compact_topic(session: Dict[str, Any]) -> str:
    text = text_of_messages(session["messages"])
    toks = tokenize(text)
    return " ".join(toks[:16]) or f"session {session['session_index']}"


def inject_visible_sessions(memory, sessions: List[Dict[str, Any]]) -> None:
    wm = memory.memory_system.user_working_memory
    wm.working_memory_queue.queue.clear()
    em = memory.memory_system.user_episodic_memory
    pm = memory.memory_system.user_persona_memory

    em.fact_episodic_memory_dict = {}
    em.episodic_memory_cache_list = []
    memory.memory_system.user_topic_message_dict = {}
    memory.memory_system.user_detail_dict.clear()

    attr_map = {}
    for pos, session in enumerate(sessions, start=1):
        topic = compact_topic(session)
        raw = session_text(session)
        fact = topic
        attr = session.get("domain", "general")
        wm.add_message_to_working_memory(
            raw_message=raw,
            message=f"[{pos}]: {topic}",
            topics=topic,
            emotions="preference evidence",
            reason="visible HABIT-Bench session",
            index=session["session_index"],
            timestamp=str(session["session_index"]),
            fact=fact,
            attribute=[attr],
        )
        em.fact_episodic_memory_dict[fact] = f"The {session['session_index']} round fact: {fact}; "
        attr_map.setdefault(attr, f"Evidence from {attr} sessions")

    if not attr_map:
        attr_map["general"] = "No visible sessions"

    pm.attr_persona = [{"User Attributes": attr_map}]
    pm.aspect_attribute_dict = dict(attr_map)
    memory._sync_memory_mappings()


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = args.repo_path
    SimpleMemory = import_omem(repo_path)

    probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    for user_id, probes in sorted(probes_by_user.items()):
        store_dir = args.output.parent / "omem_store" / user_id
        shutil.rmtree(store_dir, ignore_errors=True)
        store_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            memory = SimpleMemory(
                user_name="user",
                agent_name="assistant",
                llm_model="gpt-4o-mini",
                api_key="dummy",
                base_url="http://localhost:1/v1",
                embedding_model_name=args.embedding_model_name,
                memory_dir=str(store_dir),
            )

        for probe in probes:
            sessions = visible_sessions(probe, payload["sessions_by_user"])
            with contextlib.redirect_stdout(io.StringIO()):
                inject_visible_sessions(memory, sessions)
                result, _, _, peak_memory, peak_increase = (
                    memory.memory_manager.retrieve_from_memory_soft_segmentation(
                        question=f"user {probe['query']}",
                        topn=args.topn,
                        drop_threshold=args.drop_threshold,
                    )
                )
            retrieved_messages = result.get("retrieved context messages", [])
            retrieved_text = "\n\n".join(
                item[0] if isinstance(item, list) and item else str(item)
                for item in retrieved_messages
            )
            scores = score_choices(probe, retrieved_text)
            evidence_session_ids = extract_session_ids(retrieved_text)
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "choice_id": pick_choice(probe, scores),
                "scores": scores,
                "evidence_session_ids": evidence_session_ids[: args.topn],
                "debug": {
                    "adapter": "official_omem_retrieval_injected_memory",
                    "official_repo_path": str(repo_path),
                    "peak_memory": peak_memory,
                    "peak_memory_increase": peak_increase,
                    "retrieved_text_preview": retrieved_text[:500],
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

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=default_repo_path())
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--topn", type=int, default=12)
    parser.add_argument("--drop-threshold", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
