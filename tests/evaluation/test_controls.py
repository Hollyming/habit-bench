from __future__ import annotations

import unittest

from eval.compact_history import (
    CompactCall,
    CompactorLengthError,
    OpenAIHistoryCompactor,
    build_compact_rows,
    compaction_limits,
    render_bounded_compact_payload,
)
from eval.controls import render_session, select_recent_history


class CharacterTokenizer:
    """Small deterministic tokenizer used to test budget semantics."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(value) for value in ids)


def session(index: int) -> dict:
    return {
        "session_id": f"s{index}",
        "session_index": index,
        "timestamp": f"2026-01-{index:02d}T00:00:00+00:00",
        "messages": [
            {"role": "user", "content": f"user message {index}"},
            {"role": "assistant", "content": f"assistant reply {index}"},
        ],
    }


class FullHistoryControlTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.sessions = [session(1), session(2), session(3)]

    def test_full_history_is_kept_when_it_fits(self):
        selection = select_recent_history(
            self.sessions, self.tokenizer, max_history_tokens=10_000
        )
        self.assertFalse(selection.truncated)
        self.assertEqual(selection.evidence_session_ids, ["s1", "s2", "s3"])
        self.assertLess(
            selection.context.index("[SESSION_ID=s1]"),
            selection.context.index("[SESSION_ID=s3]"),
        )

    def test_overflow_keeps_recent_complete_sessions_in_chronological_order(self):
        recent_context = render_session(self.sessions[1]) + "\n\n" + render_session(
            self.sessions[2]
        )
        selection = select_recent_history(
            self.sessions,
            self.tokenizer,
            max_history_tokens=len(recent_context),
        )
        self.assertTrue(selection.truncated)
        self.assertEqual(selection.evidence_session_ids, ["s2", "s3"])
        self.assertNotIn("[SESSION_ID=s1]", selection.context)
        self.assertLess(
            selection.context.index("[SESSION_ID=s2]"),
            selection.context.index("[SESSION_ID=s3]"),
        )
        self.assertLessEqual(selection.context_tokens, len(recent_context))


class RecordingCompactor:
    def __init__(self):
        self.inputs = []

    def compact(self, previous_summary, sessions, *, user_id, cutoff):
        self.inputs.append(
            {
                "previous_summary": previous_summary,
                "sessions": sessions,
                "user_id": user_id,
                "cutoff": cutoff,
            }
        )
        identifiers = []
        if previous_summary:
            identifiers.append(previous_summary)
        identifiers.extend(
            f"[SESSION_ID={item['session_id']}]" for item in sessions
        )
        return CompactCall(
            summary="## stable_defaults\n- " + " ".join(identifiers),
            records=[
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            ],
        )


class CompactFullMemoryControlTest(unittest.TestCase):
    def test_recursive_split_recovers_from_length_overflow_without_changing_normal_calls(self):
        class SplitRecoveringCompactor(OpenAIHistoryCompactor):
            def __init__(self):
                self.tokenizer = CharacterTokenizer()
                self.input_token_budget = 100_000
                self.calls = []

            def _messages(self, previous_summary, sessions, *, attempt=1):
                del previous_summary, sessions, attempt
                return [{"role": "user", "content": "short"}]

            def _chat_tokens(self, messages):
                del messages
                return 1

            def _request(self, previous_summary, sessions, *, user_id, cutoff):
                del user_id, cutoff
                self.calls.append(len(sessions))
                if len(sessions) > 1:
                    raise CompactorLengthError("synthetic length overflow")
                session_id = sessions[0]["session_id"]
                return (
                    (previous_summary + f"\n[SESSION_ID={session_id}]").strip(),
                    {"source_session_ids": [session_id]},
                )

        compactor = SplitRecoveringCompactor()
        result = compactor.compact(
            "",
            [session(index) for index in range(1, 5)],
            user_id="u1",
            cutoff=4,
        )
        self.assertEqual(compactor.calls, [4, 2, 1, 3, 1, 2, 1, 1])
        self.assertEqual(
            [f"s{index}" for index in range(1, 5)],
            sorted(set(item.split("=")[1][:-1] for item in result.summary.splitlines())),
        )
        self.assertGreaterEqual(
            sum(int(record.get("length_overflow_splits", 0)) for record in result.records),
            3,
        )

    def test_bounded_fallback_rejects_invented_citations(self):
        payload = {section: [] for section in (
            "stable_defaults",
            "scoped_preferences",
            "exceptions_and_one_offs",
            "changes_and_reversals",
            "supporting_observations",
            "unresolved_conflicts",
        )}
        payload["stable_defaults"] = [
            {"fact": "Prefers aisle seats", "session_ids": ["s1"]}
        ]
        rendered = render_bounded_compact_payload(
            payload,
            allowed_session_ids={"s1"},
        )
        self.assertIn("Prefers aisle seats [SESSION_ID=s1]", rendered)
        payload["stable_defaults"][0]["session_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "invented session IDs"):
            render_bounded_compact_payload(
                payload,
                allowed_session_ids={"s1"},
            )

    def test_compactor_retries_have_progressively_stricter_hard_limits(self):
        first = compaction_limits(4096, 1)
        second = compaction_limits(4096, 2)
        final = compaction_limits(4096, 3)
        self.assertEqual(first["target_tokens"], 2048)
        self.assertEqual(second["target_tokens"], 1024)
        self.assertEqual(final["target_tokens"], 640)
        self.assertGreater(first["max_bullets"], second["max_bullets"])
        self.assertGreater(second["max_bullets"], final["max_bullets"])
        self.assertLess(final["target_tokens"], 4096)

        compactor = object.__new__(OpenAIHistoryCompactor)
        compactor.summary_max_tokens = 4096
        final_prompt = compactor._messages("", [], attempt=3)[1]["content"]
        self.assertIn("at most 640 tokenizer tokens", final_prompt)
        self.assertIn("at most 6 bullets", final_prompt)
        self.assertIn("never enumerate every session", final_prompt)

    def test_compactor_is_query_independent_and_retains_recent_raw_history(self):
        tokenizer = CharacterTokenizer()
        compactor = RecordingCompactor()
        sessions = [session(index) for index in range(1, 6)]
        payload = {
            "sessions_by_user": {"u1": sessions},
            "probes": [
                {
                    "probe_id": "early",
                    "user_id": "u1",
                    "query": "SECRET_EARLY_QUERY",
                    "choices": [{"choice_id": "A", "text": "SECRET_CHOICE"}],
                    "visible_history_scope": {"max_session_index": 1},
                },
                {
                    "probe_id": "late",
                    "user_id": "u1",
                    "query": "SECRET_LATE_QUERY",
                    "choices": [{"choice_id": "B", "text": "SECRET_CHOICE"}],
                    "visible_history_scope": {"max_session_index": 5},
                },
            ],
        }
        rows, records = build_compact_rows(
            payload,
            tokenizer=tokenizer,
            compactor=compactor,
            history_token_budget=500,
            summary_token_budget=150,
            recent_token_budget=250,
        )
        self.assertEqual([row["probe_id"] for row in rows], ["early", "late"])
        self.assertNotIn("[COMPACT_MEMORY_BEGIN]", rows[0]["memory_context"])
        self.assertIn("[COMPACT_MEMORY_BEGIN]", rows[1]["memory_context"])
        self.assertIn("[RECENT_RAW_SESSIONS_BEGIN]", rows[1]["memory_context"])
        exposed = repr(compactor.inputs)
        self.assertNotIn("SECRET_EARLY_QUERY", exposed)
        self.assertNotIn("SECRET_LATE_QUERY", exposed)
        self.assertNotIn("SECRET_CHOICE", exposed)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(
            rows[1]["debug"]["strategy"],
            "query_independent_online_compaction_plus_recent_raw",
        )
        self.assertLessEqual(rows[1]["debug"]["context_tokens"], 500)


if __name__ == "__main__":
    unittest.main()
