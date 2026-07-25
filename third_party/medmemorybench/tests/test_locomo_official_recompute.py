import json

import pytest

from metrics.locomo_metrics import LoCoMoF1Metric
from scripts.recompute_locomo_official import official_score, recompute


def _row(category, prediction, answer, query_type="single_hop"):
    return {
        "query_id": f"q{category}",
        "context_id": "conv-26",
        "query_type": query_type,
        "model_output": prediction,
        "expected_answer": answer,
        "score": 0.5,
        "is_correct": True,
        "evaluation_details": {"category": category, "enhanced": True},
    }


def test_metric_defaults_to_official_without_semantic_boost():
    result = LoCoMoF1Metric().compute(
        query_id="q",
        query_type="single_hop",
        model_output="The answer is Chapel Hill and this is deliberately verbose",
        expected_answers=["Chapel Hill"],
        category=4,
    )
    assert result.details["enhanced"] is False
    assert result.score < 0.5


def test_official_category_rules_are_exact():
    assert official_score(_row(3, "Chapel Hill", "Chapel Hill; NC")) == 1.0
    assert official_score(_row(1, "painting, hiking", "hiking, painting", "multi_hop")) == 1.0
    assert official_score(_row(5, "It is unknown", "", "adversarial")) == 0.0
    assert official_score(
        _row(5, "No information available in the dialogue", "", "adversarial")
    ) == 1.0


def test_recompute_preserves_original_and_writes_official_summary(tmp_path):
    source = tmp_path / "query_answer.json"
    target = tmp_path / "official.json"
    source.write_text(
        json.dumps(
            {
                "method_name": "bm25_rag",
                "model_name": "Qwen3-8B",
                "dataset_name": "locomo",
                "summary": {"correct_count": 99},
                "queries": [
                    _row(4, "Chapel Hill", "Chapel Hill"),
                    _row(5, "Unknown", "", "adversarial"),
                ],
            }
        ),
        encoding="utf-8",
    )

    metadata = recompute(source, target)
    output = json.loads(target.read_text(encoding="utf-8"))
    assert output["summary"] == {"correct_count": 99}
    assert [row["official_score"] for row in output["queries"]] == [1.0, 0.0]
    assert metadata["headline_metric"] == "mean_official_score"
    assert metadata["summary"]["mean_official_score"] == pytest.approx(0.5)
