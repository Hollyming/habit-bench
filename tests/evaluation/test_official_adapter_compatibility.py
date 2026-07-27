from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.official_adapters.graphiti import apply_json_schema_bounds
from eval.official_adapters.graphiti_parallel import run as run_graphiti_parallel
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

    def test_graphiti_parallel_merges_isolated_user_workers(self) -> None:
        class FakeProcess:
            def __init__(self, command: list[str]):
                self.command = command
                input_path = Path(command[command.index("--input") + 1])
                output_path = Path(command[command.index("--output") + 1])
                worker_index = int(command[command.index("--shard-index") + 1])
                worker_count = int(command[command.index("--shard-count") + 1])
                payload = json.loads(input_path.read_text(encoding="utf-8"))
                selected_users = {
                    user_id
                    for index, user_id in enumerate(sorted(payload["sessions_by_user"]))
                    if index % worker_count == worker_index
                }
                rows = [
                    {
                        "probe_id": probe["probe_id"],
                        "memory_context": probe["user_id"],
                        "evidence_session_ids": [],
                    }
                    for probe in payload["probes"]
                    if probe["user_id"] in selected_users
                ]
                output_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (output_path.parent / "graphiti_config.json").write_text(
                    json.dumps({"backend": "Kuzu", "db_path": str(output_path.parent)}),
                    encoding="utf-8",
                )
                session_count = sum(
                    len(payload["sessions_by_user"][user_id])
                    for user_id in selected_users
                )
                (output_path.parent / "graphiti_runtime.json").write_text(
                    json.dumps(
                        {
                            "add_stats": {
                                "episodes_attempted": session_count,
                                "episodes_added": session_count,
                                "add_failure_count": 0,
                                "add_elapsed_sec": 1.0,
                                "wall_elapsed_sec": 1.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            def wait(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            output_path = root / "memory_contexts.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "sessions_by_user": {
                            "u0": [{"session_id": "s0"}],
                            "u1": [{"session_id": "s1"}],
                            "u2": [{"session_id": "s2"}],
                        },
                        "probes": [
                            {"probe_id": "p0", "user_id": "u0"},
                            {"probe_id": "p1", "user_id": "u1"},
                            {"probe_id": "p2", "user_id": "u2"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input=input_path,
                output=output_path,
                topk=5,
                user_workers=2,
                progress_every=25,
                continue_on_add_error=False,
            )
            with patch(
                "eval.official_adapters.graphiti_parallel.subprocess.Popen",
                FakeProcess,
            ):
                run_graphiti_parallel(args)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["probe_id"] for row in rows], ["p0", "p1", "p2"])
            runtime = json.loads(
                (root / "graphiti_runtime.json").read_text(encoding="utf-8")
            )
            self.assertEqual(runtime["user_workers_effective"], 2)
            self.assertEqual(runtime["add_stats"]["episodes_attempted"], 3)
            self.assertEqual(runtime["add_stats"]["episodes_added"], 3)
            self.assertEqual(runtime["add_stats"]["add_failure_count"], 0)

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
