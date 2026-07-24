#!/usr/bin/env python
"""Memory-context adapter for the official SeCom repository.

Each session is ingested chronologically through SeCom's segment construction
and LLMLingua compression path. The adapter returns native retrieval context;
choice selection is handled by the shared evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def write_full_config(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "segmentor": {
            "segment_model": args.llm_model,
            "prompt_path": "instructions/segment_with_exchange_number.md",
            "incremental_prompt_path": "instructions/segment_incremental.md",
        },
        "compressor": {"compress_model": args.compressor_model},
        "retriever": {
            "storage": "FAISS",
            "embedding_model": args.embedding_model,
            "device_map": args.embedding_device,
        },
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def ingest_session(memory_manager, session: Dict[str, Any], compress_rate: float) -> None:
    """Append one chronological session using SeCom's native build stages."""
    existing = list(memory_manager.memory_bank)
    memory_manager.build_memory([session_memory_unit(session)], compress_rate=compress_rate)
    memory_manager.memory_bank = existing + list(memory_manager.memory_bank)


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    repo_path = Path(os.environ.get("SECOM_REPO", "")) if os.environ.get("SECOM_REPO") else args.repo_path
    os.environ["OPENAI_API_BASE"] = args.openai_base_url
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.parent / "secom_method_config.json"
    method_config = write_full_config(config_path, args)
    config_record = {
        "adapter": "secom_official_code_online",
        "repo_path": str(repo_path),
        "method_config": method_config,
        "compress_rate": args.compress_rate,
        "topk": args.topk,
        "memory_llm": {
            "model": args.llm_model,
            "base_url": args.openai_base_url,
            "api_key": "<redacted>" if args.openai_api_key else None,
        },
    }
    (args.output.parent / "secom_config.json").write_text(
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

    SeCom, used_vllm_stub = import_secom(repo_path)
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
    sessions_added = 0
    for user_id, probes in sorted(probes_by_user.items()):
        memory_manager = SeCom(granularity="segment", config_path=str(config_path))
        sessions = sorted(payload["sessions_by_user"][user_id], key=lambda row: row["session_index"])
        next_session_pos = 0
        retriever_session_pos = -1
        for probe in probes:
            cutoff = probe["visible_history_scope"]["max_session_index"]
            while (
                next_session_pos < len(sessions)
                and sessions[next_session_pos]["session_index"] <= cutoff
            ):
                ingest_session(
                    memory_manager, sessions[next_session_pos], args.compress_rate
                )
                next_session_pos += 1
                sessions_added += 1
                if args.progress_every and sessions_added % args.progress_every == 0:
                    print(
                        f"secom_add_progress sessions={sessions_added} "
                        f"elapsed_sec={time.time() - started:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
            if retriever_session_pos != next_session_pos:
                memory_manager.init_retriever(args.topk, **memory_manager.config.retriever)
                retriever_session_pos = next_session_pos
            retrieved_texts, _, retrieved_tokens = memory_manager.retrieve([probe["query"]])
            retrieved_text = retrieved_texts[0]
            evidence_session_ids = extract_session_ids(retrieved_text)
            sessions_visible = visible_sessions(probe, payload["sessions_by_user"])
            predictions_by_probe[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "memory_context": retrieved_text,
                "evidence_session_ids": evidence_session_ids[: args.topk],
                "debug": {
                    "adapter": "secom_official_code_online",
                    "official_repo_path": str(repo_path),
                    "unused_vllm_import_stub": used_vllm_stub,
                    "retrieved_text_preview": retrieved_text[:500],
                },
                "cost": {
                    "visible_history_sessions": len(sessions_visible),
                    "visible_history_tokens_est": sum(
                        len(tokenize(text_of_messages(session["messages"]))) for session in sessions_visible
                    ),
                    "retrieved_sessions": len(evidence_session_ids),
                    "retrieved_tokens_est": int(retrieved_tokens),
                    "stored_items_est": len(memory_manager.memory_bank),
                },
            }

    predictions = [predictions_by_probe[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, predictions)
    (args.output.parent / "secom_runtime.json").write_text(
        json.dumps(
            {
                "elapsed_sec": round(time.time() - started, 3),
                "sessions_added": sessions_added,
                "total_predictions": len(predictions),
                "segmentation_enabled": True,
                "compression_enabled": args.compress_rate < 1.0,
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
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_SERVED_MODEL", "habitbench-qwen3-8b"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument(
        "--compressor-model",
        default=os.getenv(
            "HABITBENCH_SECOM_COMPRESSOR",
            "/home/jmzhang/models/llmlingua-2-xlm-roberta-large-meetingbank",
        ),
    )
    parser.add_argument("--compress-rate", type=float, default=0.9)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv(
            "HABITBENCH_SECOM_EMBED_MODEL",
            os.getenv("HABITBENCH_EMBED_MODEL", "/home/jmzhang/models/e5-base-v2"),
        ),
    )
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cuda"))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("HABITBENCH_PROGRESS_EVERY", "100")))
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
