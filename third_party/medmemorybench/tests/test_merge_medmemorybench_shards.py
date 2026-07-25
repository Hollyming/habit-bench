import json
from pathlib import Path

import pytest

from metrics import MetricResult
from scripts.merge_medmemorybench_shards import merge
from src.result import EvaluationReport, ResultCollector


def _write_shard(root: Path, persona: int, query_id: str, correct: bool) -> None:
    root.mkdir(parents=True)
    identity = {"method_name": "mem0", "model_name": "Qwen3-8B", "dataset_name": "medmemorybench"}
    result = {
        **identity,
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T00:01:00",
        "duration_seconds": 60,
        "summary": {},
        "efficiency": {},
        "memory_build_summary": {"by_method": {}},
        "llm_usage": {},
        "config": {"total_personas": 1, "dataset_config": {"evaluation": {"persona_ids": [persona]}}},
    }
    query = {
        **identity,
        "by_context": {str(persona): {"total": 1, "correct": int(correct), "query_ids": [query_id]}},
        "queries": [{
            "query_id": query_id,
            "query_type": "entity_exact_match",
            "score": float(correct),
            "is_correct": correct,
            "query_time": 2.0,
            "retrieved_count": 1,
            "evaluation_details": {},
        }],
    }
    memory = {
        **identity,
        "summary": {},
        "units": [{"context_id": persona, "unit_id": f"u{persona}", "session_count": 1, "total_time": 3.0}],
    }
    (root / "x_result.json").write_text(json.dumps(result))
    (root / "x_query_answer.json").write_text(json.dumps(query))
    (root / "x_memory_build.json").write_text(json.dumps(memory))


def test_merge_independent_shards(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_shard(first, 1, "q1", True)
    _write_shard(second, 2, "q2", False)
    output = tmp_path / "merged"
    manifest = merge([first, second], output, expected_queries=2)
    result = json.loads((output / "medmemorybench_merged_result.json").read_text())
    assert manifest["persona_ids"] == [1, 2]
    assert result["summary"]["total_queries"] == 2
    assert result["summary"]["overall_accuracy"] == 0.5
    assert result["efficiency"]["total_memory_construction_time"] == 6.0


def test_merge_allows_persona_local_query_ids(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_shard(first, 1, "same", True)
    _write_shard(second, 2, "same", False)
    output = tmp_path / "merged"
    merge([first, second], output, expected_queries=2)
    payload = json.loads((output / "medmemorybench_merged_query_answer.json").read_text())
    assert [(row["context_id"], row["query_id"]) for row in payload["queries"]] == [
        (1, "same"),
        (2, "same"),
    ]


def test_merge_rejects_non_persona_config_drift(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_shard(first, 1, "q1", True)
    _write_shard(second, 2, "q2", False)
    result_path = second / "x_result.json"
    result = json.loads(result_path.read_text())
    result["config"]["method_config"] = {"retrieval": {"top_k": 10}}
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="Shard config mismatch"):
        merge([first, second], tmp_path / "merged", expected_queries=2)


def test_result_collector_persists_context_id(tmp_path: Path) -> None:
    metric = MetricResult("local-id", "entity_exact_match", 1.0, True, "a", "a")
    collector = ResultCollector()
    collector.add_result(metric, context_id=7)
    report = EvaluationReport(
        method_name="mem0",
        model_name="Qwen3-8B",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 1, "correct": 1, "overall_accuracy": 1.0},
        detailed_results=[metric.to_dict()],
    )
    _, _, query_path = collector.save_reports(report, tmp_path, [])
    payload = json.loads(query_path.read_text())
    assert payload["queries"][0]["context_id"] == 7
