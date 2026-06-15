#!/usr/bin/env python
"""HABIT-Bench adapter for the official A-MEM repository.

The adapter uses A-MEM's official `AgenticMemorySystem.add_note` and
`search_agentic` retrieval path. LLM-based memory evolution is disabled by
replacing `process_memory` with a no-evolution function, because the official
code otherwise calls a live LLM backend during every add after the first. This
run is therefore an official-code retrieval adapter, not a full reproduction of
A-MEM's agentic evolution mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
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
    return Path(__file__).resolve().parents[2] / "third_party" / "official-baselines" / "repos" / "a-mem"


def import_amem(repo_path: Path):
    if not repo_path.exists():
        raise FileNotFoundError(f"A-MEM repository not found: {repo_path}")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    sys.path.insert(0, str(repo_path))
    from agentic_memory.memory_system import AgenticMemorySystem

    return AgenticMemorySystem


def make_memory_system(AgenticMemorySystem, args: argparse.Namespace):
    memory_system = AgenticMemorySystem(
        model_name=args.model_name,
        llm_backend="ollama",
        llm_model=args.llm_model,
        evo_threshold=10**9,
    )
    memory_system.process_memory = lambda note: (False, note)
    return memory_system


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = Path(os.environ.get("AMEM_REPO", "")) if os.environ.get("AMEM_REPO") else args.repo_path
    AgenticMemorySystem = import_amem(repo_path)

    probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)
    for probes in probes_by_user.values():
        probes.sort(key=lambda p: (p["visible_history_scope"]["max_session_index"], p["probe_id"]))

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
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

            results = memory_system.search_agentic(probe["query"], k=args.topk)
            retrieved_text = "\n\n".join(result.get("content", "") for result in results)
            scores = score_choices(probe, retrieved_text)
            evidence_session_ids = extract_session_ids(retrieved_text)
            sessions_visible = visible_sessions(probe, payload["sessions_by_user"])
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "choice_id": pick_choice(probe, scores),
                "scores": scores,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "official_amem_search_agentic_no_evolution",
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
                    "stored_items_est": len(sessions_visible),
                },
            }

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=default_repo_path())
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="llama3.2")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
