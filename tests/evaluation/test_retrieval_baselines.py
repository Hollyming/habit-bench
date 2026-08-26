from __future__ import annotations

import unittest

import numpy as np

from eval.retrieval_baselines import (
    BaselineRetriever,
    explicit_as_of_time,
    reciprocal_rank_fusion,
    temporal_score,
)


def make_session(index: int, timestamp: str, text: str) -> dict:
    return {
        "session_id": f"s{index}",
        "user_id": "u1",
        "session_index": index,
        "timestamp": timestamp,
        "domain": "test",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"recorded {text}"},
        ],
    }


SESSIONS = [
    make_session(0, "2025-01-01T00:00:00Z", "needle alpha"),
    make_session(1, "2025-02-01T00:00:00Z", "semantic beta"),
    make_session(2, "2025-03-01T00:00:00Z", "recent gamma"),
    make_session(3, "2025-04-01T00:00:00Z", "future delta"),
]


def make_probe(query: str, cutoff: int = 2) -> dict:
    return {
        "probe_id": "p1",
        "user_id": "u1",
        "query": query,
        "timestamp": "2025-03-02T00:00:00Z",
        "choices": [
            {"choice_id": "A", "text": "SECRET_CHOICE_A"},
            {"choice_id": "B", "text": "SECRET_CHOICE_B"},
        ],
        "visible_history_scope": {"max_session_index": cutoff},
    }


def payload(probe: dict) -> dict:
    return {"sessions_by_user": {"u1": SESSIONS}, "probes": [probe]}


class FakeDenseModel:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, **kwargs):
        del kwargs
        self.encoded_texts.extend(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "beta" in lowered:
                vectors.append([0.0, 1.0])
            elif "needle" in lowered or "alpha" in lowered:
                vectors.append([1.0, 0.0])
            elif "gamma" in lowered:
                vectors.append([-1.0, 0.0])
            else:
                vectors.append([0.0, -1.0])
        return np.asarray(vectors, dtype=np.float32)


class RetrievalBaselineTest(unittest.TestCase):
    def test_recency_uses_fixed_complete_sessions_and_respects_cutoff(self) -> None:
        probe = make_probe("query", cutoff=2)
        retriever = BaselineRetriever(
            payload(probe),
            {
                "method_name": "recency_5",
                "retrieval": {"session_k": 5},
            },
        )
        row = retriever.retrieve(probe)
        self.assertEqual(row["evidence_session_ids"], ["s2", "s1", "s0"])
        self.assertNotIn("s3", row["memory_context"])
        self.assertIn("[TIMESTAMP=2025-03-01T00:00:00Z]", row["memory_context"])
        self.assertNotIn("SECRET_CHOICE", row["memory_context"])

    def test_bm25_returns_lexically_matching_complete_session(self) -> None:
        probe = make_probe("needle")
        retriever = BaselineRetriever(
            payload(probe),
            {
                "method_name": "bm25_rag",
                "retrieval": {"topk": 2},
            },
        )
        row = retriever.retrieve(probe)
        self.assertEqual(row["evidence_session_ids"][0], "s0")
        first = row["debug"]["ranked_results"][0]
        self.assertEqual(first["session_id"], "s0")
        self.assertGreater(first["bm25_score"], 0)

    def test_dense_rag_uses_cosine_and_excludes_future_session(self) -> None:
        probe = make_probe("beta")
        model = FakeDenseModel()
        retriever = BaselineRetriever(
            payload(probe),
            {
                "method_name": "dense_rag",
                "retrieval": {"topk": 2},
            },
            dense_model=model,
        )
        row = retriever.retrieve(probe)
        self.assertEqual(row["evidence_session_ids"][0], "s1")
        self.assertNotIn("s3", row["evidence_session_ids"])
        self.assertAlmostEqual(
            row["debug"]["ranked_results"][0]["dense_cosine"], 1.0
        )
        self.assertFalse(any("future delta" in text for text in model.encoded_texts))

    def test_as_of_parser_keeps_clock_colon_and_year(self) -> None:
        parsed = explicit_as_of_time(
            "Use the policy state as of 12:30 am on May 23, 2025: close it."
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2025-05-23T00:30:00")

    def test_explicit_time_targets_history_instead_of_latest_session(self) -> None:
        explicit_probe = make_probe(
            "Use the policy state as of 9:00 am on February 1, 2025: answer."
        )
        target = temporal_score(
            SESSIONS[1],
            explicit_probe,
            SESSIONS[:3],
            explicit_half_life_days=90,
            recency_half_life_days=180,
            recency_half_life_sessions=20,
            future_penalty=0.05,
        )
        future = temporal_score(
            SESSIONS[2],
            explicit_probe,
            SESSIONS[:3],
            explicit_half_life_days=90,
            recency_half_life_days=180,
            recency_half_life_sessions=20,
            future_penalty=0.05,
        )
        self.assertEqual(target.target_source, "query_as_of")
        self.assertGreater(target.value, future.value)
        self.assertEqual(future.relation, "after_explicit_target")

    def test_no_explicit_time_uses_weak_current_cutoff_recency(self) -> None:
        probe = make_probe("What is the current handling?")
        recent = temporal_score(
            SESSIONS[2],
            probe,
            SESSIONS[:3],
            explicit_half_life_days=90,
            recency_half_life_days=180,
            recency_half_life_sessions=20,
            future_penalty=0.05,
        )
        old = temporal_score(
            SESSIONS[0],
            probe,
            SESSIONS[:3],
            explicit_half_life_days=90,
            recency_half_life_days=180,
            recency_half_life_sessions=20,
            future_penalty=0.05,
        )
        self.assertEqual(recent.target_source, "probe_timestamp")
        self.assertGreater(recent.value, old.value)

    def test_rrf_is_one_indexed_and_symmetric(self) -> None:
        scores = reciprocal_rank_fusion([0, 1, 2], [1, 0, 2], rrf_constant=60)
        expected = 1 / 61 + 1 / 62
        self.assertAlmostEqual(scores[0], expected)
        self.assertAlmostEqual(scores[1], expected)
        self.assertAlmostEqual(scores[2], 2 / 63)

    def test_temporal_hybrid_records_every_score_component(self) -> None:
        probe = make_probe(
            "unknown tokens as of 9:00 am on February 1, 2025: answer."
        )
        retriever = BaselineRetriever(
            payload(probe),
            {
                "method_name": "temporal_hybrid_rag",
                "retrieval": {"topk": 3},
                "fusion": {
                    "rrf_constant": 60,
                    "temporal_lambda": 0.02,
                    "no_explicit_time_scale": 0.25,
                },
                "temporal": {
                    "explicit_half_life_days": 90,
                    "recency_half_life_days": 180,
                    "recency_half_life_sessions": 20,
                    "after_explicit_target_penalty": 0.05,
                },
            },
            dense_model=FakeDenseModel(),
        )
        row = retriever.retrieve(probe)
        first = row["debug"]["ranked_results"][0]
        for field in (
            "bm25_score",
            "bm25_rank",
            "dense_cosine",
            "dense_rank",
            "rrf_score",
            "time_score",
            "time_scale",
            "final_score",
        ):
            self.assertIn(field, first)
        self.assertEqual(row["debug"]["temporal_target_source"], "query_as_of")


if __name__ == "__main__":
    unittest.main()
