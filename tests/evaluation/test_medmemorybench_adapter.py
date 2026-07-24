import unittest
from types import SimpleNamespace

from eval.medmemorybench_adapters.structured_memory import (
    empty_contexts,
    extract_session_ids,
    require_successful_memory_build,
    render_session,
)


class MedMemoryBenchAdapterTest(unittest.TestCase):
    def test_session_marker_round_trip(self):
        rendered = render_session(
            {
                "session_id": "food-u1-s2",
                "session_index": 2,
                "timestamp": "2026-01-02",
                "domain": "food",
                "messages": [
                    {"role": "user", "content": "I prefer oats."},
                    {"role": "assistant", "content": "Noted."},
                ],
            }
        )
        self.assertEqual(extract_session_ids(rendered), ["food-u1-s2"])
        self.assertIn("user: I prefer oats.", rendered)

    def test_dry_run_respects_memory_context_contract(self):
        rows = empty_contexts(
            {"probes": [{"probe_id": "p1"}, {"probe_id": "p2"}]},
            "mem0_qwen3-8b_smoke",
        )
        self.assertEqual([row["probe_id"] for row in rows], ["p1", "p2"])
        self.assertTrue(all(row["memory_context"] == "" for row in rows))
        self.assertTrue(all("choice_id" not in row for row in rows))

    def test_memory_build_failure_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "food-u1-s2"):
            require_successful_memory_build(
                SimpleNamespace(
                    success=False,
                    method="mem0",
                    extra={"error": "upstream write failed"},
                ),
                session_id="food-u1-s2",
            )

    def test_invalid_memory_build_result_is_rejected(self):
        with self.assertRaises(TypeError):
            require_successful_memory_build(None, session_id="food-u1-s2")


if __name__ == "__main__":
    unittest.main()
