#!/usr/bin/env python3
"""Run vendored A-MEM's native per-turn LoCoMo ingestion with matched QA protocol.

This is a validation runner, not the main benchmark entry point.  It keeps the
official A-MEM runner's one-note-per-turn construction while matching the
MedMemoryBench adapter on dataset, embedding model, direct-question retrieval,
top-k, reader prompt/model, and official LoCoMo metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMEM_ROOT = PROJECT_ROOT / "methods" / "amem" / "A-mem"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# A-MEM uses top-level imports such as ``memory_layer``.  Keep its directory
# available without shadowing MedMemoryBench's own top-level ``utils`` package.
if str(AMEM_ROOT) not in sys.path:
    sys.path.append(str(AMEM_ROOT))

from benchmarks.locomo.dataset import LoCoMoDataset  # noqa: E402
from metrics import MetricsAggregator, MetricsCalculator  # noqa: E402
from utils.llm_client import create_llm_client, format_messages, get_usage_tracker  # noqa: E402
from utils.templates import get_prompt_manager, get_template_manager  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _note_to_dict(note: Any) -> dict[str, Any]:
    return {
        "id": note.id,
        "content": note.content,
        "timestamp": note.timestamp,
        "context": note.context,
        "keywords": list(note.keywords),
        "tags": list(note.tags),
        "links": list(note.links),
        "evolution_history": list(note.evolution_history),
    }


def _save_memory_cache(system: Any, output_dir: Path) -> None:
    with (output_dir / "native_memories.pkl").open("wb") as handle:
        pickle.dump(system.memories, handle)
    system.retriever.save(
        str(output_dir / "native_retriever.pkl"),
        str(output_dir / "native_retriever_embeddings.npy"),
    )
    (output_dir / "native_memories.json").write_text(
        json.dumps(
            [_note_to_dict(note) for note in system.memories.values()],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_memory_cache(system: Any, output_dir: Path) -> bool:
    memory_path = output_dir / "native_memories.pkl"
    retriever_path = output_dir / "native_retriever.pkl"
    embeddings_path = output_dir / "native_retriever_embeddings.npy"
    if not all(path.exists() for path in (memory_path, retriever_path, embeddings_path)):
        return False
    with memory_path.open("rb") as handle:
        system.memories = pickle.load(handle)
    system.retriever = system.retriever.load(str(retriever_path), str(embeddings_path))
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset.resolve()

    dataset = LoCoMoDataset(
        data_dir=dataset_path.parent,
        config={
            "data_file": dataset_path.name,
            "sample_ids": [args.sample_id],
            "include_images": True,
        },
    )
    dataset.load()
    units = list(dataset.get_evaluation_units())
    if len(units) != 1:
        raise ValueError(f"Expected exactly one sample {args.sample_id!r}, found {len(units)}")
    unit = units[0]

    robust_module = importlib.import_module("memory_layer_robust")
    memory_class = getattr(robust_module, "RobustAgenticMemorySystem")
    system = memory_class(
        model_name=str(args.embedding_model.resolve()),
        llm_backend="openai",
        llm_model=args.model,
        evo_threshold=args.evo_threshold,
        api_key=args.api_key,
        api_base=args.base_url,
        max_tokens=args.max_tokens,
        max_context_chars=args.max_context_chars,
        check_connection=False,
        usage_tracker=get_usage_tracker(),
    )
    reader = create_llm_client(
        provider="openai",
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    prompt_manager = get_prompt_manager(dataset="locomo", method="amem")
    system_message = get_template_manager("locomo").get_system_message()
    metric_calculator = MetricsCalculator(dataset="locomo")
    aggregator = MetricsAggregator()

    get_usage_tracker().reset()
    get_usage_tracker().set_phase("memorize")
    build_started = time.time()
    loaded_cache = args.resume and _load_memory_cache(system, output_dir)
    turns_added = 0
    if not loaded_cache:
        for session in unit.sessions_to_inject:
            for turn in session.dialogues:
                # Exact formatting used by the vendored official native runner.
                content = f"Speaker {turn['speaker']}says : {turn['text']}"
                system.add_note(content, time=session.date_time)
                turns_added += 1
                if turns_added % 25 == 0:
                    print(f"native_memory_progress turns={turns_added}", flush=True)
        _save_memory_cache(system, output_dir)
    build_seconds = time.time() - build_started

    all_notes = list(system.memories.values())
    result_rows: list[dict[str, Any]] = []
    queries = unit.queries_to_evaluate[: args.max_queries or None]
    get_usage_tracker().set_phase("query")
    for index, query in enumerate(queries, start=1):
        query_started = time.time()
        memory_context, indices = system.find_related_memories(query.question, k=args.top_k)
        indices_list = [int(value) for value in np.asarray(indices).tolist()]
        retrieved = [_note_to_dict(all_notes[value]) for value in indices_list]
        formatted_question = prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )
        full_question = (
            f"[Retrieved A-Mem Notes]\n{memory_context}\n\n{formatted_question}"
            if memory_context.strip()
            else formatted_question
        )
        response = reader.chat(format_messages(full_question, system_message))
        query_seconds = time.time() - query_started
        metric = metric_calculator.compute(
            query_id=query.query_id,
            query_type=query.query_type,
            model_output=response.content,
            expected_answers=query.get_correct_answers(),
            question=query.question,
            category=query.category,
            evidence=query.evidence,
            adversarial_answer=query.adversarial_answer,
            metadata=query.metadata,
        )
        metric.query_time = query_seconds
        metric.memory_construction_time = build_seconds / len(queries) if queries else 0.0
        metric.retrieved_count = len(retrieved)
        metric.retrieved_memories = retrieved
        aggregator.add_result(metric)
        result_rows.append(
            {
                **metric.to_dict(),
                "context_id": args.sample_id,
                "question": query.question,
                "expected_answers": query.get_correct_answers(),
                "memory_context": memory_context,
                "retrieved_indices": indices_list,
                "retrieved_memories": retrieved,
            }
        )
        if index % 25 == 0 or index == len(queries):
            print(f"native_query_progress queries={index}/{len(queries)}", flush=True)
            (output_dir / "native_query_answer.partial.json").write_text(
                json.dumps(result_rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    payload = {
        "contract": "amem.native_matched_locomo.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "memory_source": str(AMEM_ROOT),
            "ingestion": "vendored native one note per dialogue turn",
            "retrieval": "direct raw question",
            "reader_protocol": "MedMemoryBench adapter-matched",
            "metric": "official LoCoMo F1",
        },
        "config": {
            "sample_id": args.sample_id,
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "model": args.model,
            "base_url": args.base_url,
            "embedding_model": str(args.embedding_model.resolve()),
            "top_k": args.top_k,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "evo_threshold": args.evo_threshold,
            "max_context_chars": args.max_context_chars,
            "max_queries": args.max_queries,
        },
        "coverage": {
            "sessions": len(unit.sessions_to_inject),
            "turns": sum(len(session.dialogues) for session in unit.sessions_to_inject),
            "turns_added": turns_added,
            "loaded_memory_cache": loaded_cache,
            "memories": len(system.memories),
            "queries": len(result_rows),
        },
        "timing": {"memory_construction_seconds": build_seconds},
        "summary": aggregator.get_summary(),
        "llm_usage": get_usage_tracker().get_stats(),
        "queries": result_rows,
    }
    (output_dir / "native_query_answer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--evo-threshold", type=int, default=100)
    parser.add_argument("--max-context-chars", type=int, default=49152)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0 or args.max_tokens <= 0 or args.evo_threshold <= 0:
        parser.error("top-k, max-tokens, and evo-threshold must be positive")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("max-queries must be positive")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"coverage": result["coverage"], "summary": result["summary"]}, indent=2))
