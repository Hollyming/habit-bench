#!/usr/bin/env python
"""HABIT-Bench adapter for the official SeCom repository.

This adapter uses official SeCom code for memory retrieval
(`SeCom.retrieve_external_memory`) and keeps the answer head deliberately
simple: lexical matching between the retrieved memory text and HABIT-Bench
multiple-choice options. It is therefore an official-code retrieval adapter,
not a full paper-reproduction of SeCom's LLM segmentation/compression stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import types
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


def session_memory_unit(session: Dict[str, Any]) -> List[str]:
    marker = f"[SESSION_ID={session['session_id']}]"
    return [marker + "\n" + text_of_messages(session["messages"])]


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
    return Path(__file__).resolve().parents[2] / "third_party" / "official-baselines" / "repos" / "SeCom"


def install_unused_vllm_stub() -> bool:
    """Let SeCom import on platforms without vLLM when LocalLLM is unused."""
    if "vllm" in sys.modules:
        return False
    module = types.ModuleType("vllm")

    class _UnavailableVLLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM is unavailable; this adapter does not use SeCom LocalLLM.")

    module.LLM = _UnavailableVLLM
    module.SamplingParams = _UnavailableVLLM
    sys.modules["vllm"] = module
    return True


def import_secom(repo_path: Path):
    if not repo_path.exists():
        raise FileNotFoundError(f"SeCom repository not found: {repo_path}")
    used_vllm_stub = install_unused_vllm_stub()
    sys.path.insert(0, str(repo_path))
    from secom.secom import SeCom

    return SeCom, used_vllm_stub


def write_retrieval_only_config(path: Path) -> None:
    path.write_text("retriever:\n  storage: BM25Retriever\n", encoding="utf-8")


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


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = Path(os.environ.get("SECOM_REPO", "")) if os.environ.get("SECOM_REPO") else args.repo_path
    SeCom, used_vllm_stub = import_secom(repo_path)

    config_path = args.output.parent / "secom_retrieval_only_bm25.yaml"
    write_retrieval_only_config(config_path)
    memory_manager = SeCom(granularity="session", config_path=str(config_path))

    predictions = []
    for probe in payload["probes"]:
        sessions = visible_sessions(probe, payload["sessions_by_user"])
        memory_units = [session_memory_unit(session) for session in sessions]
        if memory_units:
            retrieved_texts, _, retrieved_tokens = memory_manager.retrieve_external_memory(
                [probe["query"]],
                memory_units,
                retrieve_topk=args.topk,
            )
            retrieved_text = retrieved_texts[0]
        else:
            retrieved_text = ""
            retrieved_tokens = 0
        scores = score_choices(probe, retrieved_text)
        evidence_session_ids = extract_session_ids(retrieved_text)
        predictions.append(
            {
                "probe_id": probe["probe_id"],
                "choice_id": pick_choice(probe, scores),
                "scores": scores,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "official_secom_retrieval_only_bm25_session",
                    "official_repo_path": str(repo_path),
                    "unused_vllm_import_stub": used_vllm_stub,
                    "retrieved_text_preview": retrieved_text[:500],
                },
                "cost": {
                    "visible_history_sessions": len(sessions),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in sessions
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": int(retrieved_tokens),
                    "stored_items_est": len(memory_units),
                },
            }
        )

    write_jsonl(args.output, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=default_repo_path())
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
