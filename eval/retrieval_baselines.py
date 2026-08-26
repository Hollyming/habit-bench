#!/usr/bin/env python
"""Non-agentic session-retrieval baselines for HABIT-Bench.

Every retrieval unit is one complete public session.  The adapter never sees
private labels, answer choices, or sessions after a probe's public cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from dateutil import parser as date_parser
from rank_bm25 import BM25Okapi

from eval.controls import render_session, visible_sessions


BASELINE_MODES = (
    "recency_5",
    "recency_10",
    "bm25_rag",
    "dense_rag",
    "temporal_hybrid_rag",
)
SESSION_SEPARATOR = "\n\n"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
AS_OF_PATTERN = re.compile(
    r"\bas[-\s]?of\s+(?P<target>.*?\b(?:19|20)\d{2})",
    flags=re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def session_search_text(session: dict[str, Any]) -> str:
    """Return message text only; metadata is rendered after retrieval."""

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in session["messages"]
    )


def lexical_tokens(text: str) -> list[str]:
    """Deterministic Unicode word tokenization for the lexical baseline."""

    return TOKEN_PATTERN.findall(text.casefold())


def _stable_rank(
    scores: Sequence[float], sessions: Sequence[dict[str, Any]]
) -> list[int]:
    """Rank descending by score with a neutral, deterministic tie break."""

    return sorted(
        range(len(sessions)),
        key=lambda index: (
            -float(scores[index]),
            int(sessions[index]["session_index"]),
            str(sessions[index]["session_id"]),
        ),
    )


def _rank_map(order: Sequence[int]) -> dict[int, int]:
    return {session_index: rank for rank, session_index in enumerate(order, start=1)}


def bm25_scores(
    sessions: Sequence[dict[str, Any]], query: str
) -> np.ndarray:
    if not sessions:
        return np.zeros(0, dtype=np.float64)
    corpus = [lexical_tokens(session_search_text(session)) for session in sessions]
    query_tokens = lexical_tokens(query)
    if not query_tokens or not any(corpus):
        return np.zeros(len(sessions), dtype=np.float64)
    model = BM25Okapi(corpus)
    return np.asarray(model.get_scores(query_tokens), dtype=np.float64)


def _normalized_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def dense_scores(
    document_embeddings: np.ndarray, query_embedding: np.ndarray
) -> np.ndarray:
    documents = _normalized_rows(document_embeddings)
    query = _normalized_rows(query_embedding)[0]
    return documents @ query


def parse_timestamp(value: Any, *, fuzzy: bool = False) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = date_parser.parse(str(value), fuzzy=fuzzy)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def explicit_as_of_time(query: str) -> datetime | None:
    match = AS_OF_PATTERN.search(query)
    if match is None:
        return None
    return parse_timestamp(match.group("target"), fuzzy=True)


@dataclass(frozen=True)
class TemporalScore:
    value: float
    target: datetime | None
    target_source: str
    relation: str


def temporal_score(
    session: dict[str, Any],
    probe: dict[str, Any],
    visible: Sequence[dict[str, Any]],
    *,
    explicit_half_life_days: float,
    recency_half_life_days: float,
    recency_half_life_sessions: float,
    future_penalty: float,
) -> TemporalScore:
    """Score temporal proximity without unconditionally preferring recency."""

    explicit_target = explicit_as_of_time(str(probe.get("query", "")))
    if explicit_target is not None:
        target = explicit_target
        source = "query_as_of"
    else:
        target = parse_timestamp(probe.get("timestamp"))
        source = "probe_timestamp"
        if target is None:
            target = next(
                (
                    parsed
                    for item in reversed(visible)
                    if (parsed := parse_timestamp(item.get("timestamp"))) is not None
                ),
                None,
            )
            source = "latest_visible_session" if target is not None else "session_index"

    session_time = parse_timestamp(session.get("timestamp"))
    if target is not None and session_time is not None:
        signed_days = (target - session_time).total_seconds() / 86_400.0
        if explicit_target is not None:
            if signed_days >= 0:
                value = math.exp(
                    -math.log(2.0) * signed_days / explicit_half_life_days
                )
                relation = "at_or_before_explicit_target"
            else:
                value = future_penalty * math.exp(
                    -math.log(2.0) * abs(signed_days) / explicit_half_life_days
                )
                relation = "after_explicit_target"
        else:
            value = math.exp(
                -math.log(2.0) * max(0.0, signed_days) / recency_half_life_days
            )
            relation = "cutoff_recency"
        return TemporalScore(float(value), target, source, relation)

    latest_index = max(int(item["session_index"]) for item in visible)
    distance = max(0, latest_index - int(session["session_index"]))
    value = math.exp(
        -math.log(2.0) * distance / recency_half_life_sessions
    )
    return TemporalScore(float(value), target, "session_index", "cutoff_recency")


def reciprocal_rank_fusion(
    bm25_order: Sequence[int],
    dense_order: Sequence[int],
    *,
    rrf_constant: int,
) -> dict[int, float]:
    if rrf_constant < 1:
        raise ValueError("rrf_constant must be positive")
    bm25_ranks = _rank_map(bm25_order)
    dense_ranks = _rank_map(dense_order)
    indices = set(bm25_ranks) | set(dense_ranks)
    return {
        index: (
            1.0 / (rrf_constant + bm25_ranks[index])
            + 1.0 / (rrf_constant + dense_ranks[index])
        )
        for index in indices
    }


def _render_ranked(sessions: Sequence[dict[str, Any]]) -> str:
    return SESSION_SEPARATOR.join(
        f"[RETRIEVAL_RANK={rank}]\n{render_session(session)}"
        for rank, session in enumerate(sessions, start=1)
    )


def _top_indices(
    order: Sequence[int], topk: int
) -> list[int]:
    if topk < 1:
        raise ValueError("topk must be positive")
    return list(order[:topk])


class BaselineRetriever:
    def __init__(
        self,
        payload: dict[str, Any],
        config: dict[str, Any],
        *,
        dense_model: Any | None = None,
        embedding_batch_size: int = 16,
    ) -> None:
        self.payload = payload
        self.config = config
        self.mode = str(config.get("method_name"))
        if self.mode not in BASELINE_MODES:
            raise ValueError(f"Unknown retrieval baseline mode: {self.mode}")
        retrieval = config.get("retrieval") or {}
        self.topk = int(retrieval.get("topk", retrieval.get("session_k", 5)))
        if self.topk < 1:
            raise ValueError("retrieval topk/session_k must be positive")
        self.dense_model = dense_model
        self.embedding_batch_size = embedding_batch_size
        self._document_embeddings: dict[str, dict[str, np.ndarray]] = {}
        self.embedding_elapsed_sec = 0.0
        if self.mode in {"dense_rag", "temporal_hybrid_rag"}:
            if dense_model is None:
                raise ValueError(f"{self.mode} requires a dense model")

    def _dense_documents(
        self, user_id: str, sessions: Sequence[dict[str, Any]]
    ) -> np.ndarray:
        """Encode only sessions visible at this probe and cache them by ID."""

        user_cache = self._document_embeddings.setdefault(user_id, {})
        missing = [
            session
            for session in sessions
            if str(session["session_id"]) not in user_cache
        ]
        if missing:
            started = time.perf_counter()
            encoded = self.dense_model.encode(
                [session_search_text(session) for session in missing],
                batch_size=self.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            normalized = _normalized_rows(encoded)
            for session, vector in zip(missing, normalized, strict=True):
                user_cache[str(session["session_id"])] = vector
            self.embedding_elapsed_sec += time.perf_counter() - started
        return np.asarray(
            [user_cache[str(session["session_id"])] for session in sessions]
        )

    def _dense_query(self, query: str) -> np.ndarray:
        encoded = self.dense_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalized_rows(encoded)[0]

    def retrieve(self, probe: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        visible = sorted(
            visible_sessions(probe, self.payload["sessions_by_user"]),
            key=lambda row: (int(row["session_index"]), str(row["session_id"])),
        )
        query = str(probe.get("query", ""))
        component_rows: dict[int, dict[str, Any]] = {
            index: {} for index in range(len(visible))
        }
        if not visible:
            return {
                "probe_id": probe["probe_id"],
                "memory_context": "",
                "evidence_session_ids": [],
                "debug": {
                    "adapter": "non_agentic_session_retrieval_baseline",
                    "mode": self.mode,
                    "strategy": "empty_visible_history",
                    "retrieval_unit": "complete_session",
                    "query_source": "public_probe_query_only",
                    "topk": self.topk,
                    "cutoff_session_index": int(
                        probe["visible_history_scope"]["max_session_index"]
                    ),
                    "visible_sessions": 0,
                    "retrieved_sessions": 0,
                    "context_order": "retrieval_rank",
                    "ranked_results": [],
                },
                "cost": {
                    "retrieval_elapsed_sec": round(
                        time.perf_counter() - started, 6
                    ),
                    "sessions_added_for_user": 0,
                    "visible_history_sessions": 0,
                    "stored_items": 0,
                    "retrieved_sessions": 0,
                    "retrieved_tokens_est": 0,
                },
            }

        if self.mode in {"recency_5", "recency_10"}:
            selected_indices = list(range(len(visible) - 1, -1, -1))[: self.topk]
            for rank, index in enumerate(selected_indices, start=1):
                component_rows[index] = {"recency_rank": rank}
            strategy = "fixed_recent_complete_sessions"
            temporal_target = None
            temporal_source = None
        else:
            bm25 = bm25_scores(visible, query)
            bm25_order = _stable_rank(bm25, visible)
            bm25_ranks = _rank_map(bm25_order)
            for index in range(len(visible)):
                component_rows[index].update(
                    {"bm25_score": float(bm25[index]), "bm25_rank": bm25_ranks[index]}
                )

            if self.mode == "bm25_rag":
                selected_indices = _top_indices(bm25_order, self.topk)
                strategy = "bm25_topk_complete_sessions"
                temporal_target = None
                temporal_source = None
            else:
                visible_embeddings = self._dense_documents(
                    str(probe["user_id"]), visible
                )
                dense = dense_scores(visible_embeddings, self._dense_query(query))
                dense_order = _stable_rank(dense, visible)
                dense_ranks = _rank_map(dense_order)
                for index in range(len(visible)):
                    component_rows[index].update(
                        {
                            "dense_cosine": float(dense[index]),
                            "dense_rank": dense_ranks[index],
                        }
                    )

                if self.mode == "dense_rag":
                    selected_indices = _top_indices(dense_order, self.topk)
                    strategy = "bge_m3_cosine_topk_complete_sessions"
                    temporal_target = None
                    temporal_source = None
                else:
                    fusion = self.config.get("fusion") or {}
                    temporal = self.config.get("temporal") or {}
                    rrf_constant = int(fusion.get("rrf_constant", 60))
                    temporal_lambda = float(fusion.get("temporal_lambda", 0.02))
                    weak_scale = float(fusion.get("no_explicit_time_scale", 0.25))
                    rrf = reciprocal_rank_fusion(
                        bm25_order, dense_order, rrf_constant=rrf_constant
                    )
                    explicit_target = explicit_as_of_time(query)
                    final_scores: list[float] = []
                    temporal_target = None
                    temporal_source = None
                    for index, session in enumerate(visible):
                        time_component = temporal_score(
                            session,
                            probe,
                            visible,
                            explicit_half_life_days=float(
                                temporal.get("explicit_half_life_days", 90.0)
                            ),
                            recency_half_life_days=float(
                                temporal.get("recency_half_life_days", 180.0)
                            ),
                            recency_half_life_sessions=float(
                                temporal.get("recency_half_life_sessions", 20.0)
                            ),
                            future_penalty=float(
                                temporal.get("after_explicit_target_penalty", 0.05)
                            ),
                        )
                        scale = 1.0 if explicit_target is not None else weak_scale
                        final = rrf[index] + temporal_lambda * scale * time_component.value
                        final_scores.append(final)
                        temporal_target = time_component.target
                        temporal_source = time_component.target_source
                        component_rows[index].update(
                            {
                                "rrf_score": rrf[index],
                                "time_score": time_component.value,
                                "time_scale": scale,
                                "time_relation": time_component.relation,
                                "final_score": final,
                            }
                        )
                    final_order = _stable_rank(final_scores, visible)
                    selected_indices = _top_indices(final_order, self.topk)
                    strategy = "rrf_bm25_dense_plus_temporal_prior"

        selected = [visible[index] for index in selected_indices]
        elapsed = time.perf_counter() - started
        ranked_results = []
        for rank, index in enumerate(selected_indices, start=1):
            session = visible[index]
            ranked_results.append(
                {
                    "rank": rank,
                    "session_id": session["session_id"],
                    "session_index": session["session_index"],
                    "timestamp": session.get("timestamp"),
                    **component_rows[index],
                }
            )
        context = _render_ranked(selected)
        return {
            "probe_id": probe["probe_id"],
            "memory_context": context,
            "evidence_session_ids": [str(session["session_id"]) for session in selected],
            "debug": {
                "adapter": "non_agentic_session_retrieval_baseline",
                "mode": self.mode,
                "strategy": strategy,
                "retrieval_unit": "complete_session",
                "query_source": "public_probe_query_only",
                "topk": self.topk,
                "cutoff_session_index": int(
                    probe["visible_history_scope"]["max_session_index"]
                ),
                "visible_sessions": len(visible),
                "retrieved_sessions": len(selected),
                "context_order": "retrieval_rank",
                "explicit_as_of_target": (
                    explicit_as_of_time(query).isoformat()
                    if explicit_as_of_time(query) is not None
                    else None
                ),
                "temporal_target": (
                    temporal_target.isoformat() if temporal_target is not None else None
                ),
                "temporal_target_source": temporal_source,
                "ranked_results": ranked_results,
            },
            "cost": {
                "retrieval_elapsed_sec": round(elapsed, 6),
                "sessions_added_for_user": len(visible),
                "visible_history_sessions": len(visible),
                "stored_items": len(visible),
                "retrieved_sessions": len(selected),
                "retrieved_tokens_est": len(lexical_tokens(context)),
            },
        }


def load_dense_model(model_path: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_path,
        device=device,
        trust_remote_code=False,
        local_files_only=True,
    )


def load_method_config(path: Path, mode: str) -> dict[str, Any]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Method config must contain a YAML object: {path}")
    if parsed.get("method_name") != mode:
        raise ValueError(
            f"Method config mismatch: expected {mode}, got {parsed.get('method_name')}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=BASELINE_MODES, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument(
        "--embedding-model-path",
        default=os.getenv(
            "HABITBENCH_EMBED_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/bge-m3",
        ),
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("HABITBENCH_EMBED_DEVICE", "cpu"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "25")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    payload = read_json(args.input)
    config = load_method_config(args.method_config, args.mode)
    dense_model = None
    model_load_sec = 0.0
    if args.mode in {"dense_rag", "temporal_hybrid_rag"}:
        started = time.perf_counter()
        dense_model = load_dense_model(
            args.embedding_model_path, args.embedding_device
        )
        model_load_sec = time.perf_counter() - started
    retriever = BaselineRetriever(
        payload,
        config,
        dense_model=dense_model,
        embedding_batch_size=args.embedding_batch_size,
    )
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(payload["probes"], start=1):
        rows.append(retriever.retrieve(probe))
        if args.progress_every and (index == 1 or index % args.progress_every == 0):
            print(
                f"retrieval_baseline_progress mode={args.mode} "
                f"completed={index} total={len(payload['probes'])} "
                f"elapsed_sec={time.perf_counter() - started:.1f}",
                flush=True,
            )
    write_jsonl(args.output, rows)
    runtime = {
        "contract_version": "habitbench.retrieval_baseline_runtime.v1",
        "mode": args.mode,
        "retrieval_unit": "complete_session",
        "query_source": "public_probe_query_only",
        "method_config": str(args.method_config.resolve()),
        "embedding_model_path": (
            str(Path(args.embedding_model_path).expanduser().resolve())
            if dense_model is not None
            else None
        ),
        "embedding_device": args.embedding_device if dense_model is not None else None,
        "model_load_sec": round(model_load_sec, 3),
        "document_embedding_sec": round(retriever.embedding_elapsed_sec, 3),
        "retrieval_sec": round(time.perf_counter() - started, 3),
        "probes": len(rows),
    }
    (args.output.parent / "retrieval_baseline_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
