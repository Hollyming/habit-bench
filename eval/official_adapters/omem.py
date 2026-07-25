#!/usr/bin/env python
"""Memory-context adapter for the official O-Mem repository.

The adapter chronologically calls O-Mem's message-understanding, working and
episodic memory updates, active persona update, and soft-segmentation retrieval.
Retrieved memory is returned to the shared evaluator for choice selection.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time
import types
from collections import defaultdict
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


MESSAGE_UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tags": {
            "type": "object",
            "properties": {
                "topic": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "attitude": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "reason": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "facts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "attributes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topic", "attitude", "reason", "facts", "attributes"],
            "additionalProperties": False,
        },
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["text", "tags", "summary", "rationale"],
    "additionalProperties": False,
}

MEMORY_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "Action": {"type": "string", "enum": ["UPDATE", "ADD", "IGNORE"]},
        "Target": {"type": "string"},
    },
    "required": ["Action", "Target"],
    "additionalProperties": False,
}


def normalize_topic_merge_payload(
    payload: Dict[str, Any], prompt_text: str
) -> tuple[Dict[str, Any], bool, bool]:
    """Keep topic-merge references inside the input event-key set."""
    try:
        input_text = prompt_text.rsplit("Input:", 1)[1].split("Output:", 1)[0].strip()
        input_topics = ast.literal_eval(input_text)
    except (IndexError, SyntaxError, ValueError):
        return payload, False, True
    if not isinstance(input_topics, dict):
        return payload, False, True

    allowed = [str(topic) for topic in input_topics]
    allowed_set = set(allowed)
    grouped = payload.get("Grouped Topics")
    grouped = grouped if isinstance(grouped, dict) else {}
    normalized: Dict[str, List[str]] = {}
    used = set()
    for raw_group_name, raw_events in grouped.items():
        events = raw_events if isinstance(raw_events, list) else [raw_events]
        valid = [
            event
            for event in events
            if isinstance(event, str) and event in allowed_set and event not in used
        ]
        if valid:
            group_name = str(raw_group_name)
            normalized[group_name] = valid
            used.update(valid)

    for topic in allowed:
        if topic in used:
            continue
        group_name = topic
        suffix = 1
        while group_name in normalized:
            suffix += 1
            group_name = f"{topic} (unmerged {suffix})"
        normalized[group_name] = [topic]
        used.add(topic)

    repaired = grouped != normalized
    normalized_payload = dict(payload)
    normalized_payload["Grouped Topics"] = normalized
    normalized_payload["Grouping Rationale"] = str(
        payload.get("Grouping Rationale", "Retained all input topics without unsupported references.")
    )
    return normalized_payload, repaired, False


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


def default_repo_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "official-baselines"
        / "vendor"
        / "O-Mem"
    )


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


def install_omem_json_output_contract(args: argparse.Namespace) -> Dict[str, int]:
    """Make O-Mem's JSON-parsing calls explicit at the OpenAI API boundary."""
    from openai.resources.chat.completions.completions import AsyncCompletions

    original_create = AsyncCompletions.create
    stats = {
        "calls_attempted": 0,
        "calls_succeeded": 0,
        "invalid_json_responses": 0,
        "raw_invalid_json_responses": 0,
        "transport_failures": 0,
        "json_schema_calls": 0,
        "json_object_calls": 0,
        "schema_contract_failures": 0,
        "topic_merge_calls": 0,
        "topic_merge_repairs": 0,
        "topic_merge_parse_fallbacks": 0,
        "topic_merge_unresolved": 0,
    }

    async def create_json_completion(self, *call_args, **call_kwargs):
        stats["calls_attempted"] += 1
        messages = call_kwargs.get("messages") or []
        prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
        if "Perform topic tagging on this message" in prompt_text:
            contract_name = "message_understanding"
            schema = MESSAGE_UNDERSTANDING_SCHEMA
        elif "user profile updater" in prompt_text and '"Action"' in prompt_text:
            contract_name = "memory_router"
            schema = MEMORY_ROUTER_SCHEMA
        elif (
            "Given a group of topics extracted from users' messages" in prompt_text
            and '"Grouped Topics"' in prompt_text
        ):
            contract_name = "topic_merge"
            schema = None
        else:
            contract_name = None
            schema = None
        if schema is not None:
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"omem_{contract_name}",
                    "strict": True,
                    "schema": schema,
                },
            }
            stats["json_schema_calls"] += 1
        else:
            call_kwargs.setdefault("response_format", {"type": "json_object"})
            stats["json_object_calls"] += 1
        call_kwargs.setdefault("temperature", args.memory_llm_temperature)
        if contract_name == "topic_merge":
            call_kwargs["max_tokens"] = max(
                int(call_kwargs.get("max_tokens") or 0), args.topic_merge_max_tokens
            )
            stats["topic_merge_calls"] += 1
        else:
            call_kwargs.setdefault("max_tokens", args.memory_llm_max_tokens)
        call_kwargs.setdefault("seed", args.memory_llm_seed)
        try:
            response = await original_create(self, *call_args, **call_kwargs)
        except Exception:
            stats["transport_failures"] += 1
            raise
        stats["calls_succeeded"] += 1
        try:
            payload = json.loads(response.choices[0].message.content)
        except (IndexError, TypeError, json.JSONDecodeError):
            stats["raw_invalid_json_responses"] += 1
            if contract_name == "topic_merge":
                payload, _, unresolved = normalize_topic_merge_payload({}, prompt_text)
                if unresolved:
                    stats["invalid_json_responses"] += 1
                    stats["topic_merge_unresolved"] += 1
                else:
                    stats["topic_merge_parse_fallbacks"] += 1
                    response.choices[0].message.content = json.dumps(payload, ensure_ascii=False)
            else:
                stats["invalid_json_responses"] += 1
        else:
            if contract_name == "message_understanding":
                tags = payload.get("tags", {})
                required_lists = [tags.get(field) for field in ["topic", "attitude", "reason", "facts"]]
                if not all(isinstance(value, list) and value for value in required_lists):
                    stats["schema_contract_failures"] += 1
            elif contract_name == "memory_router":
                if payload.get("Action") not in {"UPDATE", "ADD", "IGNORE"} or not isinstance(
                    payload.get("Target"), str
                ):
                    stats["schema_contract_failures"] += 1
            elif contract_name == "topic_merge":
                payload, repaired, unresolved = normalize_topic_merge_payload(payload, prompt_text)
                if repaired:
                    stats["topic_merge_repairs"] += 1
                if unresolved:
                    stats["topic_merge_unresolved"] += 1
                response.choices[0].message.content = json.dumps(payload, ensure_ascii=False)
        return response

    AsyncCompletions.create = create_json_completion
    return stats


def serialize_retrieval(result: Dict[str, Any]) -> str:
    fields = {
        key: result.get(key)
        for key in [
            "persona attributes",
            "persona facts",
            "working memory facts",
            "retrieved context messages",
        ]
        if key in result
    }
    return json.dumps(fields, ensure_ascii=False, default=str, indent=2)


def stored_item_count(memory) -> int:
    system = memory.memory_system
    return sum(
        [
            len(system.user_working_memory.working_memory_queue.queue),
            len(system.agent_working_memory.working_memory_queue.queue),
            len(system.user_episodic_memory.episodic_memory_cache_list),
            len(system.agent_episodic_memory.episodic_memory_cache_list),
            len(system.user_episodic_memory.fact_episodic_memory_dict),
            len(system.user_persona_memory.preference_persona),
            len(system.user_persona_memory.attr_persona),
        ]
    )


async def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = (
        Path(os.environ["HABITBENCH_OMEM_REPO"])
        if os.environ.get("HABITBENCH_OMEM_REPO")
        else args.repo_path
    )
    config_record = {
        "adapter": "omem_official_code_compatibility",
        "repo_path": str(repo_path),
        "embedding_model": args.embedding_model_name,
        "memory_llm": {
            "model": args.llm_model,
            "base_url": args.openai_base_url,
            "api_key": "<redacted>" if args.openai_api_key else None,
            "message_understanding_enabled": True,
            "persona_update_enabled": True,
            "response_format": "json_object",
            "temperature_default": args.memory_llm_temperature,
            "max_tokens_default": args.memory_llm_max_tokens,
            "topic_merge_max_tokens": args.topic_merge_max_tokens,
            "seed_default": args.memory_llm_seed,
        },
        "topn": args.topn,
        "drop_threshold": args.drop_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "omem_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.dry_run_config:
        write_jsonl(
            args.output,
            [
                {
                    "probe_id": probe["probe_id"],
                    "memory_context": "",
                    "evidence_session_ids": [],
                    "debug": {"dry_run_config": True, **config_record},
                    "cost": {},
                }
                for probe in payload["probes"]
            ],
        )
        return

    SimpleMemory = import_omem(repo_path)
    memory_llm_stats = install_omem_json_output_contract(args)

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
    started = time.time()
    messages_added = 0
    for user_id, probes in sorted(probes_by_user.items()):
        store_dir = args.output.parent / "omem_store" / user_id
        shutil.rmtree(store_dir, ignore_errors=True)
        store_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            memory = SimpleMemory(
                user_name="user",
                agent_name="assistant",
                llm_model=args.llm_model,
                api_key=args.openai_api_key,
                base_url=args.openai_base_url,
                embedding_model_name=args.embedding_model_name,
                memory_dir=str(store_dir),
            )

        sessions = sorted(payload["sessions_by_user"][user_id], key=lambda row: row["session_index"])
        next_session_pos = 0
        persona_synced_pos = -1
        for probe in probes:
            cutoff = probe["visible_history_scope"]["max_session_index"]
            while (
                next_session_pos < len(sessions)
                and sessions[next_session_pos]["session_index"] <= cutoff
            ):
                session = sessions[next_session_pos]
                for message in session["messages"]:
                    marked_message = f"[SESSION_ID={session['session_id']}] {message['content']}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        await asyncio.wait_for(
                            memory.add_message(
                                marked_message,
                                is_user=message["role"] == "user",
                                timestamp=session["timestamp"],
                            ),
                            timeout=args.message_timeout_sec,
                        )
                    messages_added += 1
                    if args.progress_every and messages_added % args.progress_every == 0:
                        print(
                            f"omem_add_progress messages={messages_added} "
                            f"elapsed_sec={time.time() - started:.1f}",
                            file=sys.stderr,
                            flush=True,
                        )
                next_session_pos += 1
            if persona_synced_pos != next_session_pos:
                with contextlib.redirect_stdout(io.StringIO()):
                    memory._sync_memory_mappings()
                    await asyncio.wait_for(
                        memory.update_persona(), timeout=args.persona_timeout_sec
                    )
                persona_synced_pos = next_session_pos

            visible = visible_sessions(probe, payload["sessions_by_user"])
            with contextlib.redirect_stdout(io.StringIO()):
                result, _, _, peak_memory, peak_increase = (
                    memory.memory_manager.retrieve_from_memory_soft_segmentation(
                        question=f"user {probe['query']}",
                        topn=args.topn,
                        drop_threshold=args.drop_threshold,
                    )
                )
            retrieved_text = serialize_retrieval(result)
            evidence_session_ids = extract_session_ids(retrieved_text)
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "memory_context": retrieved_text,
                "evidence_session_ids": evidence_session_ids[: args.topn],
                "debug": {
                    "adapter": "omem_official_code_compatibility",
                    "official_repo_path": str(repo_path),
                    "peak_memory": peak_memory,
                    "peak_memory_increase": peak_increase,
                    "retrieved_text_preview": retrieved_text[:500],
                },
                "cost": {
                    "visible_history_sessions": len(visible),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in visible
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": len(tokenize(retrieved_text)),
                    "stored_items_est": stored_item_count(memory),
                },
            }

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)
    (args.output.parent / "omem_runtime.json").write_text(
        json.dumps(
            {
                "elapsed_sec": round(time.time() - started, 3),
                "messages_added": messages_added,
                "total_predictions": len(predictions),
                "message_understanding_enabled": True,
                "persona_update_enabled": True,
                "memory_llm_json_output_contract": True,
                "memory_llm_stats": memory_llm_stats,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=default_repo_path())
    parser.add_argument(
        "--embedding-model-name",
        default=os.getenv(
            "HABITBENCH_EMBED_MODEL",
            "/plm-shared/zhangjunming/Workspace/models/bge-m3",
        ),
    )
    parser.add_argument("--topn", type=int, default=12)
    parser.add_argument("--drop-threshold", type=float, default=0.0)
    parser.add_argument(
        "--llm-model",
        default=os.getenv("HABITBENCH_SERVED_MODEL", "Qwen3-8B"),
    )
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--message-timeout-sec", type=float, default=300.0)
    parser.add_argument("--persona-timeout-sec", type=float, default=1800.0)
    parser.add_argument(
        "--memory-llm-temperature",
        type=float,
        default=float(os.getenv("HABITBENCH_MEMORY_LLM_TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--memory-llm-max-tokens",
        type=int,
        default=int(os.getenv("HABITBENCH_MEMORY_LLM_MAX_TOKENS", "1024")),
    )
    parser.add_argument(
        "--memory-llm-seed",
        type=int,
        default=int(os.getenv("HABITBENCH_MEMORY_LLM_SEED", "42")),
    )
    parser.add_argument(
        "--topic-merge-max-tokens",
        type=int,
        default=int(os.getenv("HABITBENCH_OMEM_TOPIC_MERGE_MAX_TOKENS", "4096")),
    )
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
