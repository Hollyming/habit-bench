from __future__ import annotations

import sys
import unittest
from pathlib import Path

from eval.official_adapters.graphiti import apply_json_schema_bounds
from eval.official_adapters.omem import normalize_topic_merge_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIRIX_ROOT = PROJECT_ROOT / "third_party/medmemorybench/methods/MIRIX"
if str(MIRIX_ROOT) not in sys.path:
    sys.path.insert(0, str(MIRIX_ROOT))

from mirix.llm_api.local_json_tool_bridge import (  # noqa: E402
    build_memory_json_tool_bridge,
)


def _tool(name: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


class OfficialAdapterCompatibilityTest(unittest.TestCase):
    def test_mirix_bridge_retains_native_finish_tool(self) -> None:
        response_format, metadata = build_memory_json_tool_bridge(
            [
                _tool(
                    "episodic_memory_insert",
                    {"items": {"type": "array", "items": {"type": "string"}}},
                ),
                _tool("finish_memory_update"),
            ]
        )
        self.assertEqual(
            set(metadata["allowed_names"]),
            {"episodic_memory_insert", "finish_memory_update"},
        )
        variants = response_format["json_schema"]["schema"]["anyOf"]
        names = {variant["properties"]["name"]["enum"][0] for variant in variants}
        self.assertEqual(names, {"episodic_memory_insert", "finish_memory_update"})

    def test_graphiti_bounds_nested_arrays_and_strings(self) -> None:
        bounded = apply_json_schema_bounds(
            {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    }
                },
            },
            max_items=16,
            max_string_chars=512,
        )
        entities = bounded["properties"]["entities"]
        self.assertEqual(entities["maxItems"], 16)
        self.assertEqual(entities["items"]["properties"]["name"]["maxLength"], 512)

    def test_omem_topic_fallback_is_lossless_and_non_merging(self) -> None:
        prompt = (
            "Given a group of topics extracted from users' messages\n"
            'Input:\n{"budgeting": "one", "spreadsheets": "two"}\nOutput:'
        )
        payload, repaired, unresolved = normalize_topic_merge_payload({}, prompt)
        self.assertTrue(repaired)
        self.assertFalse(unresolved)
        self.assertEqual(
            payload["Grouped Topics"],
            {"budgeting": ["budgeting"], "spreadsheets": ["spreadsheets"]},
        )


if __name__ == "__main__":
    unittest.main()
