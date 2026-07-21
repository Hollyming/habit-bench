#!/usr/bin/env python
"""Mem0 official-API adapter for the HABIT-Bench memory-context contract.

The adapter performs chronological ``Memory.add(..., infer=True)`` updates and
native ``Memory.search`` retrieval. It never selects an answer choice; the
shared evaluator performs that step with the configured base model.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# Mem0's telemetry helper creates a new PostHog worker for every captured
# operation, even when telemetry is disabled after construction. The full
# benchmark performs thousands of operations, so suppress the helper before
# constructing Memory and close the one import-time client during cleanup.
os.environ.setdefault("MEM0_TELEMETRY", "False")


class FailOnMemoryError(logging.Handler):
    """Turn Mem0's otherwise silent extraction/update failures into run failures."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            raise RuntimeError(f"Mem0 internal error: {record.getMessage()}")


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
    }


def add_session_full(memory, user_id: str, session: Dict[str, Any]) -> None:
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


def search_visible_memories(memory, probe: Dict[str, Any], topk: int, threshold: float, rerank: bool) -> List[Dict[str, Any]]:
    return memory.search(
        probe["query"],
        user_id=probe["user_id"],
        limit=topk,
        threshold=threshold,
        rerank=rerank,
    ).get("results", [])


def configure_qwen_non_thinking(memory) -> None:
    """Pass Qwen3's non-thinking flag through Mem0's OpenAI client."""
    original = memory.llm.generate_response

    @functools.wraps(original)
    def generate_response(*args, **kwargs):
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        return original(*args, extra_body=extra_body, **kwargs)

    memory.llm.generate_response = generate_response


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    store_dir = args.output.parent / "mem0_store"
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    mem0_process_dir = store_dir / "mem0_home"
    mem0_process_dir.mkdir(parents=True, exist_ok=True)
    # Mem0 1.0.2 creates a telemetry migration vector store even when event
    # capture is disabled. Isolate that global path so concurrent user shards
    # never race through ~/.mem0/migrations_qdrant.
    os.environ["MEM0_DIR"] = str(mem0_process_dir)

    config = build_mem0_full_config(args, store_dir, args.collection_name)
    (args.output.parent / "mem0_config.json").write_text(
        json.dumps(
            {
                **config,
                "mem0_process_dir": str(mem0_process_dir),
                "qwen_enable_thinking": False,
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
                    "memory_context": "",
                    "evidence_session_ids": [],
                    "debug": {
                        "adapter": "mem0_official_api",
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
        (args.output.parent / "mem0_runtime.json").write_text(
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

    import mem0.memory.main as mem0_memory_main
    from mem0 import Memory

    mem0_memory_main.capture_event = lambda *args, **kwargs: None
    fail_on_error_handler = None
    if not args.allow_memory_log_errors:
        fail_on_error_handler = FailOnMemoryError()
        mem0_memory_main.logger.addHandler(fail_on_error_handler)

    memory = Memory.from_config(config)
    configure_qwen_non_thinking(memory)
    probes_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)
    for probes in probes_by_user.values():
        probes.sort(key=lambda row: (row["visible_history_scope"]["max_session_index"], row["probe_id"]))

    total_to_add = sum(
        sum(1 for session in payload["sessions_by_user"][user_id] if session["session_index"] <= max(
            probe["visible_history_scope"]["max_session_index"] for probe in probes
        ))
        for user_id, probes in probes_by_user.items()
    )
    add_started = time.time()
    attempted = 0
    stored = 0
    failures: List[Dict[str, Any]] = []
    predictions_by_probe: Dict[str, Dict[str, Any]] = {}
    search_elapsed_sec = 0.0

    for user_id, probes in sorted(probes_by_user.items()):
        sessions = payload["sessions_by_user"][user_id]
        next_session_pos = 0
        stored_for_user = 0
        for probe in probes:
            max_idx = probe["visible_history_scope"]["max_session_index"]
            while next_session_pos < len(sessions) and sessions[next_session_pos]["session_index"] <= max_idx:
                session = sessions[next_session_pos]
                attempted += 1
                try:
                    add_session_full(memory, user_id, session)
                    stored += 1
                    stored_for_user += 1
                    if args.progress_every and (stored == 1 or stored % args.progress_every == 0 or attempted == total_to_add):
                        print(
                            f"mem0_add_progress stored={stored} attempted={attempted} total={total_to_add} "
                            f"elapsed_sec={time.time() - add_started:.1f}",
                            file=sys.stderr,
                            flush=True,
                        )
                except Exception as exc:
                    failures.append(
                        {"user_id": user_id, "session_id": session["session_id"], "error": repr(exc)}
                    )
                    if not args.continue_on_add_error:
                        raise
                next_session_pos += 1

            search_started = time.time()
            results = search_visible_memories(memory, probe, args.topk, args.threshold, args.rerank)
            search_elapsed_sec += time.time() - search_started
            retrieved_text = "\n\n".join(result.get("memory", "") for result in results)
            evidence_session_ids = [
                result.get("metadata", {}).get("session_id")
                for result in results
                if result.get("metadata", {}).get("session_id")
            ]
            if not evidence_session_ids:
                evidence_session_ids = extract_session_ids(retrieved_text)
            evidence_session_ids = list(dict.fromkeys(evidence_session_ids))
            visible = visible_sessions(probe, payload["sessions_by_user"])
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "memory_context": retrieved_text,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "mem0_official_api",
                    "llm_model": args.llm_model,
                    "openai_base_url": args.openai_base_url,
                    "retrieved_scores": [result.get("score") for result in results],
                    "retrieved_text_preview": retrieved_text[:500],
                    "added_until_session_index": max_idx,
                },
                "cost": {
                    "visible_history_sessions": len(visible),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in visible
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": len(tokenize(retrieved_text)),
                    "stored_items_est": stored_for_user,
                },
            }

    add_stats = {
        "stored_sessions_attempted": attempted,
        "stored_sessions_succeeded": stored,
        "add_failures": failures[:20],
        "add_failure_count": len(failures),
        "add_elapsed_sec": round(time.time() - add_started - search_elapsed_sec, 3),
    }
    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    (args.output.parent / "mem0_runtime.json").write_text(
        json.dumps(
                {
                    "add_stats": add_stats,
                    "memory_add_infer": True,
                    "mem0_telemetry_disabled": True,
                    "search_elapsed_sec": round(search_elapsed_sec, 3),
                    "total_predictions": len(predictions),
                },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_jsonl(args.output, predictions)
    try:
        memory.vector_store.client.close()
    except Exception:
        pass
    try:
        memory._telemetry_vector_store.client.close()
    except Exception:
        pass
    try:
        from mem0.memory.telemetry import client_telemetry

        client_telemetry.close()
    except Exception:
        pass
    if fail_on_error_handler is not None:
        mem0_memory_main.logger.removeHandler(fail_on_error_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--collection-name", default="habit_bench_mem0")
    parser.add_argument("--embedding-model-name", default=os.getenv("HABITBENCH_EMBED_MODEL", "/home/jmzhang/models/e5-base-v2"))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("HABITBENCH_EMBED_DIMS", "768")))
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cpu"))
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_SERVED_MODEL", "habitbench-qwen3-8b"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HABITBENCH_MEMORY_LLM_MAX_TOKENS", "4096")))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--continue-on-add-error", action="store_true")
    parser.add_argument("--allow-memory-log-errors", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
