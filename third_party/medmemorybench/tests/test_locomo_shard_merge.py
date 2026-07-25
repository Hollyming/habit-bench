import json
from pathlib import Path

import pytest

from scripts.merge_locomo_shards import merge


def _write_shard(root: Path, context: str, query_id: str) -> None:
    root.mkdir(parents=True)
    identity = {"method_name": "bm25_rag", "model_name": "Qwen3-8B", "dataset_name": "locomo"}
    result = {
        **identity,
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T00:01:00",
        "duration_seconds": 60,
        "config": {"dataset_config": {"evaluation": {}}},
        "llm_usage": {},
    }
    query = {
        **identity,
        "by_context": {context: {"total": 1, "correct": 1, "query_ids": [query_id]}},
        "queries": [{
            "query_id": query_id,
            "context_id": context,
            "query_type": "single_hop",
            "score": 0.75,
            "is_correct": True,
            "query_time": 1.0,
            "retrieved_count": 5,
        }],
    }
    memory = {
        **identity,
        "memory_chunk_size": 10240,
        "units": [{
            "context_id": context,
            "session_count": 2,
            "chunk_count": 1,
            "total_time": 3.0,
            "total_entries": 2,
            "total_stored_chunks": 2,
        }],
    }
    for suffix, payload in (("result", result), ("query_answer", query), ("memory_build", memory)):
        (root / f"locomo_{suffix}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_locomo_merge_accepts_string_context_ids(tmp_path):
    shard_a, shard_b = tmp_path / "a", tmp_path / "b"
    _write_shard(shard_a, "conv-26", "conv-26_q0")
    _write_shard(shard_b, "conv-30", "conv-30_q0")
    manifest = merge([shard_a, shard_b], tmp_path / "merged", expected_queries=2)
    assert manifest["sample_ids"] == ["conv-26", "conv-30"]
    merged = json.loads((tmp_path / "merged/locomo_merged_result.json").read_text())
    assert merged["summary"]["total_queries"] == 2
    assert merged["summary"]["overall_avg_score"] == 0.75


def test_locomo_merge_rejects_duplicate_sample(tmp_path):
    shard_a, shard_b = tmp_path / "a", tmp_path / "b"
    _write_shard(shard_a, "conv-26", "conv-26_q0")
    _write_shard(shard_b, "conv-26", "conv-26_q1")
    with pytest.raises(ValueError, match="Duplicate LoCoMo samples"):
        merge([shard_a, shard_b], tmp_path / "merged", expected_queries=2)
