from __future__ import annotations

import unittest

from eval.core.dataset import DatasetContractError
from eval.official_adapters.mem0 import configure_qwen_non_thinking
from eval.run import validate_memory_contexts


class ProtocolTest(unittest.TestCase):
    def test_mem0_qwen_requests_disable_thinking(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.kwargs = None

            def generate_response(self, *args, **kwargs):
                self.kwargs = kwargs
                return "{}"

        class FakeMemory:
            def __init__(self) -> None:
                self.llm = FakeLLM()

        memory = FakeMemory()
        configure_qwen_non_thinking(memory)
        memory.llm.generate_response(messages=[])
        self.assertFalse(
            memory.llm.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_memory_adapter_cannot_select_choices(self) -> None:
        with self.assertRaises(DatasetContractError):
            validate_memory_contexts(
                [{"probe_id": "p1", "memory_context": "x", "choice_id": "A"}],
                ["p1"],
            )

    def test_context_coverage(self) -> None:
        rows = [{"probe_id": "p1", "memory_context": "context"}]
        self.assertEqual(validate_memory_contexts(rows, ["p1"])["p1"]["memory_context"], "context")


if __name__ == "__main__":
    unittest.main()
