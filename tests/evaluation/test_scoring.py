from __future__ import annotations

import unittest
from pathlib import Path

from eval.core.dataset import DatasetBundle, DatasetContractError
from eval.core.scoring import score_predictions


def bundle() -> DatasetBundle:
    probes = [
        {
            "probe_id": "p1",
            "user_id": "u1",
            "domain": "food",
            "query": "q1",
            "choices": [
                {"choice_id": "A", "text": "a"},
                {"choice_id": "B", "text": "b"},
            ],
            "visible_history_scope": {"user_id": "u1", "max_session_index": 0},
            "split": "test",
        },
        {
            "probe_id": "p2",
            "user_id": "u1",
            "domain": "food",
            "query": "q2",
            "choices": [
                {"choice_id": "A", "text": "a"},
                {"choice_id": "B", "text": "b"},
            ],
            "visible_history_scope": {"user_id": "u1", "max_session_index": 0},
            "split": "test",
        },
    ]
    keys = {
        "p1": {"gold_choice_id": "A", "probe_type": "direct_use"},
        "p2": {"gold_choice_id": "B", "probe_type": "boundary"},
    }
    return DatasetBundle(Path("."), {"u1": []}, probes, keys, {"name": "fixture"})


def evidence_bundle(*, finance: bool = False) -> DatasetBundle:
    session_ids = (
        ["d1", "d2", "t1", "n1", "other"]
        if finance
        else ["s1", "s2", "s3", "s4", "s5", "other"]
    )
    sessions = [
        {"session_id": session_id, "session_index": index}
        for index, session_id in enumerate(session_ids)
    ]
    probe = {
        "probe_id": "p1",
        "user_id": "u1",
        "domain": "finance" if finance else "food",
        "query": "q",
        "choices": [
            {"choice_id": "A", "text": "a"},
            {"choice_id": "B", "text": "b"},
        ],
        "visible_history_scope": {
            "user_id": "u1",
            "max_session_index": len(sessions) - 1,
        },
        "split": "test",
    }
    if finance:
        key = {
            "probe_id": "p1",
            "gold_choice_id": "A",
            "probe_type": "surface_decoy_pair",
            "capability_group": "surface_decoy_multi_habit_retrieval",
            "decision_evidence_session_ids": ["d1", "d2"],
            "gold_evidence_session_ids": ["d1", "d2", "t1", "n1"],
            "temporal_context_session_ids": ["t1"],
            "nonbinding_evidence_session_ids": ["n1"],
            "required_component_groups": [["d1", "d2"]],
            "decision_unit_ids": ["du1"],
            "decision_bundle_id": "db1",
        }
    else:
        key = {
            "probe_id": "p1",
            "gold_choice_id": "A",
            "probe_type": "direct_use",
            "capability_group": "habit_direct_use",
            "gold_evidence_session_ids": ["s1", "s2", "s3", "s4", "s5"],
        }
    return DatasetBundle(
        Path("."),
        {"u1": sessions},
        [probe],
        {"p1": key},
        {"name": "finance-fixture" if finance else "food-fixture"},
    )


class ScoringTest(unittest.TestCase):
    def test_exact_choice_accuracy(self) -> None:
        _, metrics, _ = score_predictions(
            [
                {"probe_id": "p1", "choice_id": "A"},
                {"probe_id": "p2", "choice_id": "A"},
            ],
            bundle(),
            "method",
        )
        self.assertEqual(metrics["overall"]["accuracy"], 0.5)
        self.assertEqual(metrics["overall"]["correct"], 1)

    def test_coverage_is_strict(self) -> None:
        with self.assertRaises(DatasetContractError):
            score_predictions(
                [{"probe_id": "p1", "choice_id": "A"}], bundle(), "method"
            )

    def test_food_ranked_retrieval_metrics(self) -> None:
        detailed, metrics, _ = score_predictions(
            [
                {
                    "probe_id": "p1",
                    "choice_id": "A",
                    "evidence_session_ids": ["s1", "other", "s2", "s3", "s4"],
                    "memory_debug": {"retrieved_count": 5},
                }
            ],
            evidence_bundle(),
            "amem",
        )
        retrieval = detailed[0]["retrieval"]
        self.assertEqual(retrieval["evidence_recall_at_5"], 0.8)
        self.assertEqual(retrieval["evidence_precision_at_5"], 0.8)
        self.assertEqual(retrieval["evidence_mrr_at_5"], 1.0)
        self.assertEqual(metrics["overall"]["evidence_recall_at_5_macro"], 0.8)
        self.assertEqual(
            metrics["overall"]["joint_answer_evidence_hit_rate_at_5"], 1.0
        )

    def test_finance_scores_decisive_chain_and_penalizes_nonbinding(self) -> None:
        detailed, metrics, _ = score_predictions(
            [
                {
                    "probe_id": "p1",
                    "choice_id": "A",
                    "evidence_session_ids": ["n1", "d1", "t1", "other", "d2"],
                    "memory_debug": {"retrieved_count": 5},
                }
            ],
            evidence_bundle(finance=True),
            "amem",
        )
        retrieval = detailed[0]["retrieval"]
        self.assertEqual(retrieval["evidence_recall_at_5"], 1.0)
        self.assertEqual(retrieval["nonbinding_intrusion_rate_at_5"], 0.2)
        self.assertEqual(retrieval["temporal_context_recall_at_5"], 1.0)
        self.assertEqual(retrieval["component_complete_coverage_at_5"], 1.0)
        self.assertFalse(retrieval["clean_grounded_answer_at_5"])
        self.assertEqual(metrics["overall"]["decision_unit_count"], 1)
        self.assertEqual(
            metrics["overall"]["decision_unit_macro_evidence_recall_at_5"], 1.0
        )

    def test_controls_do_not_masquerade_as_ranked_retrieval(self) -> None:
        _, no_memory_metrics, _ = score_predictions(
            [{"probe_id": "p1", "choice_id": "A", "evidence_session_ids": []}],
            evidence_bundle(finance=True),
            "no_memory",
        )
        self.assertEqual(no_memory_metrics["overall"]["retrieval_mode"], "none")
        self.assertNotIn(
            "evidence_recall_at_5_macro", no_memory_metrics["overall"]
        )
        self.assertEqual(
            no_memory_metrics["overall"]["decision_unit_macro_accuracy"], 1.0
        )
        self.assertNotIn(
            "decision_unit_macro_evidence_recall_at_5",
            no_memory_metrics["overall"],
        )

        _, full_metrics, _ = score_predictions(
            [
                {
                    "probe_id": "p1",
                    "choice_id": "A",
                    "evidence_session_ids": ["d1", "d2", "t1", "n1", "other"],
                }
            ],
            evidence_bundle(finance=True),
            "full_memory",
        )
        self.assertEqual(full_metrics["overall"]["retrieval_mode"], "context")
        self.assertEqual(full_metrics["overall"]["context_evidence_recall_macro"], 1.0)
        self.assertNotIn("evidence_recall_at_5_macro", full_metrics["overall"])


if __name__ == "__main__":
    unittest.main()
