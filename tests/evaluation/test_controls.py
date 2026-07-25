from __future__ import annotations

import unittest

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


class FullMemoryControlTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
