#!/usr/bin/env python
"""Memory-context adapter for Graphiti's official ``add_episode`` path.

This full-path scaffold uses Graphiti's official episode ingestion pipeline:
`Graphiti.add_episode(...)` performs LLM-based entity/edge extraction, KG
resolution, embedding generation, and graph writes. Retrieval uses Graphiti's
official `search_` API with edge-cosine search to avoid the local Kuzu BM25
index limitation observed in the lightweight adapter.

Retrieved graph facts are returned to the shared evaluator. This adapter never
selects an answer choice.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
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
            "schema_max_items": args.schema_max_items,
            "schema_max_string_chars": args.schema_max_string_chars,
            "request_timeout_sec": args.request_timeout_sec,
            "request_max_retries": args.request_max_retries,
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


def apply_json_schema_bounds(
    schema: Dict[str, Any],
    *,
    max_items: int,
    max_string_chars: int,
) -> Dict[str, Any]:
    """Bound otherwise-unbounded Graphiti extraction schemas for local decoding.

    graphiti-core's Pydantic response models intentionally leave collection and
    string sizes open. Qwen can repeat valid entity objects indefinitely under
    constrained decoding, then reach the completion limit with an unclosed JSON
    document. These bounds only restrict degenerate output size; the returned
    object is still validated and consumed by graphiti-core unchanged.
    """
    if max_items <= 0 or max_string_chars <= 0:
        raise ValueError("Graphiti schema bounds must be positive")
    bounded = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "array":
                node["maxItems"] = min(
                    int(node.get("maxItems", max_items)),
                    max_items,
                )
            elif node_type == "string":
                node["maxLength"] = min(
                    int(node.get("maxLength", max_string_chars)),
                    max_string_chars,
                )
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(bounded)
    return bounded


def bound_llm_response_schema(
    llm_client: Any,
    *,
    max_items: int,
    max_string_chars: int,
) -> None:
    original_build_response_format = llm_client._build_response_format

    def bounded_response_format(response_model):
        response_format = original_build_response_format(response_model)
        if response_format.get("type") != "json_schema":
            return response_format
        schema_record = response_format.get("json_schema")
        if not isinstance(schema_record, dict):
            return response_format
        schema = schema_record.get("schema")
        if isinstance(schema, dict):
            schema_record["schema"] = apply_json_schema_bounds(
                schema,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
        return response_format

    llm_client._build_response_format = bounded_response_format


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


async def add_session_episode(
    graph: Graphiti,
    user_id: str,
    session: Dict[str, Any],
    previous_episode_uuid: str | None,
) -> str | None:
    from graphiti_core.nodes import EpisodeType

    prepare_kuzu_group_driver(graph, user_id)
    result = await graph.add_episode(
        name=session["session_id"],
        episode_body=session_body(session),
        source_description="HABIT-Bench user-agent session",
        reference_time=datetime.fromisoformat(session["timestamp"]),
        source=EpisodeType.message,
        group_id=user_id,
        previous_episode_uuids=[previous_episode_uuid] if previous_episode_uuid else None,
    )
    episode = getattr(result, "episode", None)
    return getattr(episode, "uuid", None)


async def async_run(args: argparse.Namespace) -> None:
    payload = apply_user_shard(read_json(args.input), args.shard_index, args.shard_count)
    store_dir = args.output.parent / "graphiti_kuzu_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    config_record = build_config_record(args, store_dir)
    (args.output.parent / "graphiti_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.dry_run_config:
        write_jsonl(
            args.output,
            [
                {
                    "probe_id": probe["probe_id"],
                    "memory_context": "",
                    "evidence_session_ids": [],
                    "debug": {
                        "adapter": "graphiti_official_api_kuzu",
                        "dry_run_config": True,
                        "llm_model": args.llm_model,
                        "openai_base_url": args.openai_base_url,
                    },
                    "cost": {},
                }
                for probe in payload["probes"]
            ],
        )
        (args.output.parent / "graphiti_runtime.json").write_text(
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
    from openai import AsyncOpenAI

    patch_graphiti_kuzu_search_defaults()
    SentenceTransformerEmbedder = make_sentence_transformer_embedder_class()
    NoopCrossEncoder = make_noop_cross_encoder_class()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        driver = KuzuDriver(str(store_dir / "db"))

    openai_client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        timeout=args.request_timeout_sec,
        max_retries=args.request_max_retries,
    )
    llm_client = OpenAIGenericClient(
        LLMConfig(
            api_key=args.openai_api_key,
            model=args.llm_model,
            small_model=args.small_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        client=openai_client,
        max_tokens=args.max_tokens,
        structured_output_mode=args.structured_output_mode,
    )
    cap_llm_generate_response(llm_client, args.max_tokens)
    bound_llm_response_schema(
        llm_client,
        max_items=args.schema_max_items,
        max_string_chars=args.schema_max_string_chars,
    )
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

        probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for probe in payload["probes"]:
            probes_by_user[probe["user_id"]].append(probe)
        for probes in probes_by_user.values():
            probes.sort(
                key=lambda probe: (
                    probe["visible_history_scope"]["max_session_index"],
                    probe["probe_id"],
                )
            )

        predictions_by_probe: Dict[str, Dict[str, Any]] = {}
        add_started = time.time()
        add_elapsed_sec = 0.0
        search_elapsed_sec = 0.0
        episodes_attempted = 0
        episodes_added = 0
        add_failures: List[Dict[str, Any]] = []
        for user_id, probes in sorted(probes_by_user.items()):
            sessions = sorted(
                payload["sessions_by_user"][user_id], key=lambda row: row["session_index"]
            )
            next_session_pos = 0
            previous_episode_uuid = None
            for probe in probes:
                cutoff = probe["visible_history_scope"]["max_session_index"]
                while (
                    next_session_pos < len(sessions)
                    and sessions[next_session_pos]["session_index"] <= cutoff
                ):
                    session = sessions[next_session_pos]
                    episodes_attempted += 1
                    try:
                        episode_started = time.time()
                        previous_episode_uuid = await add_session_episode(
                            graph, user_id, session, previous_episode_uuid
                        )
                        add_elapsed_sec += time.time() - episode_started
                        episodes_added += 1
                    except Exception as exc:
                        add_failures.append(
                            {
                                "user_id": user_id,
                                "session_id": session["session_id"],
                                "error": repr(exc),
                            }
                        )
                        if not args.continue_on_add_error:
                            raise
                    next_session_pos += 1
                    if args.progress_every and episodes_added % args.progress_every == 0:
                        print(
                            f"graphiti_add_progress added={episodes_added} "
                            f"elapsed_sec={time.time() - add_started:.1f}",
                            file=sys.stderr,
                            flush=True,
                        )

                search_started = time.time()
                results = await graph.search_(
                    probe["query"],
                    config=graphiti_search_config(args.topk),
                    group_ids=[user_id],
                    driver=driver,
                )
                search_elapsed_sec += time.time() - search_started
                edges = results.edges
                retrieved_text = "\n\n".join(edge.fact for edge in edges)
                evidence_session_ids = extract_session_ids(retrieved_text)
                visible = visible_sessions(probe, payload["sessions_by_user"])
                predictions_by_probe[probe["probe_id"]] = {
                    "probe_id": probe["probe_id"],
                    "memory_context": retrieved_text,
                    "evidence_session_ids": evidence_session_ids[: args.topk],
                    "debug": {
                        "adapter": "graphiti_official_api_kuzu",
                        "llm_model": args.llm_model,
                        "openai_base_url": args.openai_base_url,
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
                        "stored_items_est": episodes_added,
                    },
                }
        add_stats = {
            "episodes_attempted": episodes_attempted,
            "episodes_added": episodes_added,
            "add_failure_count": len(add_failures),
            "add_failures": add_failures[:20],
            "add_elapsed_sec": round(add_elapsed_sec, 3),
            "wall_elapsed_sec": round(time.time() - add_started, 3),
        }
        predictions = [
            predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]
        ]
        (args.output.parent / "graphiti_runtime.json").write_text(
            json.dumps(
                {
                    "add_stats": add_stats,
                    "search_elapsed_sec": round(search_elapsed_sec, 3),
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
    parser.add_argument(
        "--embedding-model-name",
        default=os.getenv(
            "HABITBENCH_EMBED_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/bge-m3",
        ),
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("HABITBENCH_EMBED_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
    )
    parser.add_argument("--small-model", default=os.getenv("HABITBENCH_SMALL_MODEL", None))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(
            os.getenv(
                "HABITBENCH_GRAPHITI_LLM_MAX_TOKENS",
                "4096",
            )
        ),
    )
    parser.add_argument(
        "--schema-max-items",
        type=int,
        default=int(os.getenv("HABITBENCH_GRAPHITI_SCHEMA_MAX_ITEMS", "16")),
    )
    parser.add_argument(
        "--schema-max-string-chars",
        type=int,
        default=int(
            os.getenv("HABITBENCH_GRAPHITI_SCHEMA_MAX_STRING_CHARS", "512")
        ),
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=float(os.getenv("HABITBENCH_GRAPHITI_REQUEST_TIMEOUT_SEC", "300")),
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=int(os.getenv("HABITBENCH_GRAPHITI_REQUEST_MAX_RETRIES", "2")),
    )
    parser.add_argument("--structured-output-mode", choices=["json_schema", "json_object"], default=os.getenv("HABITBENCH_STRUCTURED_OUTPUT_MODE", "json_schema"))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("HABITBENCH_SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("HABITBENCH_SHARD_COUNT", "1")))
    parser.add_argument("--continue-on-add-error", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_run(parse_args()))
