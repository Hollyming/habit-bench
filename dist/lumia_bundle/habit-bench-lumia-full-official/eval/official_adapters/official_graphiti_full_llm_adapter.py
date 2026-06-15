#!/usr/bin/env python
"""HABIT-Bench adapter for Graphiti's LLM-backed `add_episode` path.

This full-path scaffold uses Graphiti's official episode ingestion pipeline:
`Graphiti.add_episode(...)` performs LLM-based entity/edge extraction, KG
resolution, embedding generation, and graph writes. Retrieval uses Graphiti's
official `search_` API with edge-cosine search to avoid the local Kuzu BM25
index limitation observed in the lightweight adapter.

The answer head remains the shared HABIT-Bench lexical multiple-choice scorer
over retrieved graph facts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


def apply_user_shard(payload: Dict[str, Any], shard_index: int, shard_count: int) -> Dict[str, Any]:
    if shard_count <= 1:
        return payload
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}")
    user_ids = sorted(payload["sessions_by_user"])
    selected_users = {
        user_id
        for idx, user_id in enumerate(user_ids)
        if idx % shard_count == shard_index
    }
    return {
        **payload,
        "sessions_by_user": {
            user_id: sessions
            for user_id, sessions in payload["sessions_by_user"].items()
            if user_id in selected_users
        },
        "probes": [
            probe for probe in payload["probes"] if probe["user_id"] in selected_users
        ],
        "shard": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "selected_users": sorted(selected_users),
        },
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_of_messages(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def session_body(session: Dict[str, Any]) -> str:
    return (
        f"SESSION_ID={session['session_id']}\n"
        f"SESSION_INDEX={session['session_index']}\n"
        f"DOMAIN={session.get('domain', 'unknown')}\n"
        + text_of_messages(session["messages"])
    )


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


def make_sentence_transformer_embedder_class():
    from graphiti_core.embedder.client import EmbedderClient

    class SentenceTransformerEmbedder(EmbedderClient):
        def __init__(self, model_name: str, device: str) -> None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)

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

    return SentenceTransformerEmbedder


def make_noop_cross_encoder_class():
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class NoopCrossEncoder(CrossEncoderClient):
        async def rank(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
            return [(passage, 1.0 / (idx + 1)) for idx, passage in enumerate(passages)]

    return NoopCrossEncoder


def graphiti_search_config(topk: int) -> SearchConfig:
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )

    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
            sim_min_score=-1.0,
        ),
        limit=topk,
    )


def build_config_record(args: argparse.Namespace, store_dir: Path) -> Dict[str, Any]:
    return {
        "backend": "Kuzu",
        "db_path": str(store_dir / "db"),
        "llm": {
            "client": "graphiti_core.llm_client.openai_generic_client.OpenAIGenericClient",
            "model": args.llm_model,
            "small_model": args.small_model,
            "base_url": args.openai_base_url,
            "api_key": "<redacted>" if args.openai_api_key else None,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "structured_output_mode": args.structured_output_mode,
        },
        "embedder": {
            "client": "local_sentence_transformers",
            "model": args.embedding_model_name,
            "device": args.embedding_device,
        },
        "search": "Graphiti.search_ edge cosine RRF, BM25 disabled for Kuzu local index compatibility",
    }


def prepare_kuzu_group_driver(graph: Any, group_id: str) -> None:
    """Keep Graphiti's Kuzu backend on one DB while preserving group_id filters."""

    driver = getattr(graph, "driver", None)
    if driver is None or driver.__class__.__name__ != "KuzuDriver":
        return

    # graphiti-core's add_episode currently expects graph drivers to expose
    # _database and clones the driver when group_id differs. KuzuDriver keeps a
    # single local database file, so set the active database marker to the group
    # being written and let Graphiti still store/query by group_id.
    setattr(driver, "_database", group_id)
    clients = getattr(graph, "clients", None)
    if clients is not None:
        setattr(clients, "driver", driver)


def cap_llm_generate_response(llm_client: Any, max_tokens: int) -> None:
    original_generate_response = llm_client.generate_response

    async def capped_generate_response(
        messages,
        response_model=None,
        max_tokens=None,
        *args,
        **kwargs,
    ):
        if max_tokens is None or max_tokens > max_tokens_cap:
            max_tokens = max_tokens_cap
        return await original_generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            *args,
            **kwargs,
        )

    max_tokens_cap = max_tokens
    llm_client.max_tokens = max_tokens_cap
    llm_client.generate_response = capped_generate_response


def patch_graphiti_kuzu_search_defaults() -> None:
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )
    from graphiti_core.utils.maintenance import edge_operations

    edge_operations.EDGE_HYBRID_SEARCH_RRF = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
            sim_min_score=-1.0,
        ),
        limit=10,
    )


async def add_visible_episodes(
    graph: Graphiti,
    sessions_by_user: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    started = time.time()
    added = 0
    attempted = 0
    failures = []
    total = sum(len(sessions) for sessions in sessions_by_user.values())
    from graphiti_core.nodes import EpisodeType

    for user_id, sessions in sorted(sessions_by_user.items()):
        previous_episode_uuid = None
        for session in sorted(sessions, key=lambda row: row["session_index"]):
            attempted += 1
            try:
                prepare_kuzu_group_driver(graph, user_id)
                result = await graph.add_episode(
                    name=session["session_id"],
                    episode_body=session_body(session),
                    source_description="HABIT-Bench user-agent session",
                    reference_time=datetime.fromisoformat(session["timestamp"]),
                    source=EpisodeType.message,
                    group_id=user_id,
                    previous_episode_uuids=[previous_episode_uuid] if previous_episode_uuid else None,
                    custom_extraction_instructions=(
                        "Extract user habits, preferences, scoped boundaries, exceptions, "
                        "drift evidence, consent constraints, and the SESSION_ID when useful."
                    ),
                )
                episode = getattr(result, "episode", None)
                previous_episode_uuid = getattr(episode, "uuid", None)
                added += 1
                if args.progress_every and (added == 1 or added % args.progress_every == 0 or attempted == total):
                    elapsed = time.time() - started
                    print(
                        f"graphiti_add_progress added={added} attempted={attempted} total={total} elapsed_sec={elapsed:.1f}",
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
        "episodes_attempted": sum(len(sessions) for sessions in sessions_by_user.values()),
        "episodes_added": added,
        "add_failure_count": len(failures),
        "add_failures": failures[:20],
        "add_elapsed_sec": round(time.time() - started, 3),
    }


async def async_run(args: argparse.Namespace) -> None:
    payload = apply_user_shard(read_json(args.input), args.shard_index, args.shard_count)
    store_dir = args.output.parent / "graphiti_full_llm_kuzu_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    config_record = build_config_record(args, store_dir)
    (args.output.parent / "graphiti_full_llm_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2, sort_keys=True),
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
                        "adapter": "official_graphiti_full_llm_episode_kuzu",
                        "dry_run_config": True,
                        "llm_model": args.llm_model,
                        "openai_base_url": args.openai_base_url,
                    },
                    "cost": {},
                }
                for probe in payload["probes"]
            ],
        )
        (args.output.parent / "graphiti_full_llm_runtime.json").write_text(
            json.dumps(
                {
                    "dry_run_config": True,
                    "total_predictions": len(payload["probes"]),
                    "note": "Config generated without creating Graphiti or calling the LLM endpoint.",
                    "shard": payload.get("shard"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return

    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    patch_graphiti_kuzu_search_defaults()
    SentenceTransformerEmbedder = make_sentence_transformer_embedder_class()
    NoopCrossEncoder = make_noop_cross_encoder_class()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        driver = KuzuDriver(str(store_dir / "db"))

    llm_client = OpenAIGenericClient(
        LLMConfig(
            api_key=args.openai_api_key,
            model=args.llm_model,
            small_model=args.small_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        max_tokens=args.max_tokens,
        structured_output_mode=args.structured_output_mode,
    )
    cap_llm_generate_response(llm_client, args.max_tokens)
    graph = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=SentenceTransformerEmbedder(args.embedding_model_name, args.embedding_device),
        cross_encoder=NoopCrossEncoder(),
    )

    add_stats: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await graph.build_indices_and_constraints(delete_existing=True)

        add_stats = await add_visible_episodes(graph, payload["sessions_by_user"], args)

        predictions = []
        search_started = time.time()
        for probe in payload["probes"]:
            results = await graph.search_(
                probe["query"],
                config=graphiti_search_config(args.topk),
                group_ids=[probe["user_id"]],
                driver=driver,
            )
            edges = results.edges
            retrieved_text = "\n\n".join(edge.fact for edge in edges)
            evidence_session_ids = extract_session_ids(retrieved_text)
            scores = score_choices(probe, retrieved_text)
            visible = visible_sessions(probe, payload["sessions_by_user"])
            predictions.append(
                {
                    "probe_id": probe["probe_id"],
                    "choice_id": pick_choice(probe, scores),
                    "scores": scores,
                    "evidence_session_ids": evidence_session_ids[: args.topk],
                    "debug": {
                        "adapter": "official_graphiti_full_llm_episode_kuzu",
                        "llm_model": args.llm_model,
                        "openai_base_url": args.openai_base_url,
                        "search_config": "edge_cosine_rrf_no_bm25",
                        "backend": "Kuzu",
                        "retrieved_text_preview": retrieved_text[:500],
                        "add_stats": add_stats,
                    },
                    "cost": {
                        "visible_history_sessions": len(visible),
                        "visible_history_tokens_est": sum(
                            len(tokenize(text_of_messages(session["messages"]))) for session in visible
                        ),
                        "retrieved_sessions": len(evidence_session_ids),
                        "retrieved_tokens_est": len(tokenize(retrieved_text)),
                        "stored_items_est": add_stats.get("episodes_added", 0),
                    },
                }
            )
        (args.output.parent / "graphiti_full_llm_runtime.json").write_text(
            json.dumps(
                {
                    "add_stats": add_stats,
                    "search_elapsed_sec": round(time.time() - search_started, 3),
                    "total_predictions": len(predictions),
                    "shard": payload.get("shard"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_jsonl(args.output, predictions)
    finally:
        await graph.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--embedding-model-name", default=os.getenv("HABITBENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cpu"))
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_SERVED_MODEL", os.getenv("HABITBENCH_LLM_MODEL", "habitbench-open-llm")))
    parser.add_argument("--small-model", default=os.getenv("HABITBENCH_SMALL_MODEL", None))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HABITBENCH_MEMORY_LLM_MAX_TOKENS", "256")))
    parser.add_argument("--structured-output-mode", choices=["json_schema", "json_object"], default=os.getenv("HABITBENCH_STRUCTURED_OUTPUT_MODE", "json_schema"))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("HABITBENCH_SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("HABITBENCH_SHARD_COUNT", "1")))
    parser.add_argument("--continue-on-add-error", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_run(parse_args()))
