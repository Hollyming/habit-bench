#!/usr/bin/env python
"""HABIT-Bench adapter for the official Graphiti package.

This adapter uses Graphiti's official Kuzu graph driver, EntityNode/EntityEdge
storage objects, and advanced search path. It bypasses LLM episode extraction
and stores each visible HABIT-Bench session as an EntityEdge fact with a local
sentence-transformers embedding. This is an official-code graph storage/search
adapter, not a full Zep/Graphiti paper reproduction.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import re
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    SearchConfig,
)


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


class DummyLLM(LLMClient):
    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=1024,
        model_size=None,
    ) -> Dict[str, Any]:
        return {}


class SentenceTransformerEmbedder(EmbedderClient):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def _embed_one(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    async def create(self, input_data) -> List[float]:
        if isinstance(input_data, str):
            return self._embed_one(input_data)
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            return self._embed_one("\n".join(input_data))
        return self._embed_one(str(input_data))

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        vecs = self.model.encode(input_data_list, normalize_embeddings=True)
        return [[float(x) for x in vec] for vec in vecs]


class NoopCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
        return [(passage, 1.0 / (idx + 1)) for idx, passage in enumerate(passages)]


def graphiti_search_config(topk: int) -> SearchConfig:
    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
            sim_min_score=-1.0,
        ),
        limit=topk,
    )


async def insert_session(graph: Graphiti, driver: KuzuDriver, user_id: str, session: Dict[str, Any]) -> None:
    user_node = EntityNode(
        uuid=f"user::{user_id}",
        name=user_id,
        group_id=user_id,
        attributes={"kind": "user"},
    )
    session_node = EntityNode(
        uuid=f"session::{session['session_id']}",
        name=f"{session['session_id']} {session.get('domain', 'unknown')}",
        group_id=user_id,
        attributes={
            "kind": "session",
            "session_id": session["session_id"],
            "session_index": session["session_index"],
        },
    )
    await user_node.save(driver)
    await session_node.save(driver)

    edge = EntityEdge(
        uuid=f"edge::{session['session_id']}",
        group_id=user_id,
        source_node_uuid=user_node.uuid,
        target_node_uuid=session_node.uuid,
        created_at=datetime.now(timezone.utc),
        name="HAS_SESSION",
        fact=session_text(session),
        attributes={
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "domain": session.get("domain", "unknown"),
        },
    )
    await edge.generate_embedding(graph.embedder)
    await edge.save(driver)


async def async_run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    store_dir = args.output.parent / "graphiti_kuzu_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        driver = KuzuDriver(str(store_dir / "db"))
    graph = Graphiti(
        graph_driver=driver,
        llm_client=DummyLLM(None),
        embedder=SentenceTransformerEmbedder(args.embedding_model_name),
        cross_encoder=NoopCrossEncoder(),
    )

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        await graph.build_indices_and_constraints(delete_existing=True)

    probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)

    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    try:
        for user_id, probes in sorted(probes_by_user.items()):
            sessions = sorted(payload["sessions_by_user"][user_id], key=lambda row: row["session_index"])
            probes_sorted = sorted(
                probes,
                key=lambda row: row["visible_history_scope"]["max_session_index"],
            )
            next_session = 0
            for probe in probes_sorted:
                max_idx = probe["visible_history_scope"]["max_session_index"]
                while next_session < len(sessions) and sessions[next_session]["session_index"] <= max_idx:
                    await insert_session(graph, driver, user_id, sessions[next_session])
                    next_session += 1

                results = await graph.search_(
                    probe["query"],
                    config=graphiti_search_config(args.topk),
                    group_ids=[user_id],
                    driver=driver,
                )
                edges = results.edges
                retrieved_text = "\n\n".join(edge.fact for edge in edges)
                evidence_session_ids = extract_session_ids(retrieved_text)
                scores = score_choices(probe, retrieved_text)
                visible = visible_sessions(probe, payload["sessions_by_user"])
                predictions_by_probe[probe["probe_id"]] = {
                    "probe_id": probe["probe_id"],
                    "choice_id": pick_choice(probe, scores),
                    "scores": scores,
                    "evidence_session_ids": evidence_session_ids[: args.topk],
                    "debug": {
                        "adapter": "official_graphiti_kuzu_edge_cosine",
                        "search_config": "edge_cosine_rrf_no_bm25",
                        "backend": "Kuzu",
                        "retrieved_text_preview": retrieved_text[:500],
                    },
                    "cost": {
                        "visible_history_sessions": len(visible),
                        "visible_history_tokens_est": sum(
                            len(tokenize(text_of_messages(session["messages"]))) for session in visible
                        ),
                        "retrieved_sessions": len(evidence_session_ids),
                        "retrieved_tokens_est": len(tokenize(retrieved_text)),
                        "stored_items_est": len(visible),
                    },
                }
    finally:
        await graph.close()

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_run(parse_args()))
