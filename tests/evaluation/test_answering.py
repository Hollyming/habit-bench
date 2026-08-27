from __future__ import annotations

import unittest

from eval.core.answering import _message_text, build_user_prompt, parse_choice_id


PROBE = {
    "probe_id": "p1",
    "query": "What should the assistant say?",
    "choices": [
        {"choice_id": "A", "text": "first"},
        {"choice_id": "B", "text": "second"},
    ],
}


class AnsweringTest(unittest.TestCase):
    def test_prompt_contains_only_model_visible_inputs(self) -> None:
        prompt = build_user_prompt(PROBE, "remembered preference")
        self.assertIn("remembered preference", prompt)
        self.assertIn("What should the assistant say?", prompt)
        self.assertNotIn("gold_choice", prompt)

    def test_parse_qwen_json(self) -> None:
        self.assertEqual(parse_choice_id('{"choice_id":"B"}', ["A", "B"]), "B")
        self.assertEqual(parse_choice_id("```json\n{\"choice_id\": \"A\"}\n```", ["A", "B"]), "A")

    def test_reasoning_content_is_a_recovery_source_when_content_is_empty(self) -> None:
        class Message:
            content = ""
            reasoning_content = 'I will answer with {"choice_id":"A"}.'

        text, source = _message_text(Message())
        self.assertEqual(source, "reasoning_content_recovery")
        self.assertIn('"choice_id":"A"', text)


if __name__ == "__main__":
    unittest.main()
