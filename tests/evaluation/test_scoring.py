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


if __name__ == "__main__":
    unittest.main()
