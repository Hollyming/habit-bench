from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from eval.core.answering import DEFAULT_SERVED_MODEL
from eval.core.dataset import DatasetContractError
from eval.official_adapters.graphiti import (
    apply_json_schema_bounds,
    parse_args as parse_graphiti_args,
)
from eval.run import validate_memory_contexts
from scripts.run_multigpu_plan import CUDA_REQUIRED_ADAPTER_METHODS


class ProtocolTest(unittest.TestCase):
    METHODS = ("mem0", "amem", "memos", "memrl", "lightmem", "letta", "mirix")
    OFFICIAL_METHODS = {
        "graphiti": "eval/official_adapters/graphiti.py",
        "secom": "eval/official_adapters/secom.py",
        "omem": "eval/official_adapters/omem.py",
    }

    def test_canonical_methods_use_medmemorybench_adapter(self) -> None:
        registry_path = Path(__file__).resolve().parents[2] / "eval" / "methods.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for method in self.METHODS:
            self.assertEqual(
                registry[method]["adapter"],
                "eval/medmemorybench_adapters/structured_memory.py",
            )

    def test_official_adapted_methods_are_registered(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        registry = json.loads(
            (project_root / "eval/methods.json").read_text(encoding="utf-8")
        )
        for method, adapter in self.OFFICIAL_METHODS.items():
            self.assertEqual(registry[method]["implementation_kind"], "official_adapted")
            self.assertEqual(registry[method]["adapter"], adapter)
            self.assertTrue((project_root / adapter).is_file())

    def test_full_memory_control_is_registered(self) -> None:
        registry_path = Path(__file__).resolve().parents[2] / "eval" / "methods.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertEqual(registry["full_memory"]["implementation_kind"], "control")
        self.assertEqual(registry["full_memory"]["revision"], "memory_context.v3")

    def test_official_source_snapshots_are_plain_directories(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        vendor_root = project_root / "third_party/official-baselines/vendor"
        expected = {
            "SeCom": "secom/secom.py",
            "O-Mem": "example_usage.py",
        }
        for source, marker in expected.items():
            source_root = vendor_root / source
            self.assertTrue((source_root / marker).is_file())
            self.assertFalse((source_root / ".git").exists())

    def test_active_method_configs_pin_bge_m3(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_root = (
            project_root / "third_party/medmemorybench/configs/method_config"
        )
        expected = {
            "provider": "local",
            "model": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "model_path": "/plm-shared/zhangjunming/Workspace/models/bge-m3",
            "dim": 1024,
        }
        for method in self.METHODS:
            path = config_root / f"{method}_qwen3-8b_adapted.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(config["embedding"], expected, method)
            self.assertEqual(config["model"]["name"], DEFAULT_SERVED_MODEL, method)
            self.assertNotIn("e5", path.read_text(encoding="utf-8").lower(), method)

    def test_memory_adapter_cannot_select_choices(self) -> None:
        with self.assertRaises(DatasetContractError):
            validate_memory_contexts(
                [{"probe_id": "p1", "memory_context": "x", "choice_id": "A"}],
                ["p1"],
            )

    def test_qwen3_server_template_defaults_to_non_thinking(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        template_path = (
            project_root / "configs/chat_templates/qwen3_no_thinking.jinja"
        )
        template = template_path.read_text(encoding="utf-8")
        self.assertIn(
            "set enable_thinking = enable_thinking | default(false)",
            template,
        )
        self.assertIn("<think>\\n\\n</think>", template)

    def test_context_coverage(self) -> None:
        rows = [{"probe_id": "p1", "memory_context": "context"}]
        self.assertEqual(validate_memory_contexts(rows, ["p1"])["p1"]["memory_context"], "context")

    def test_native_cuda_adapters_are_bound_to_their_worker_gpu(self) -> None:
        self.assertEqual(CUDA_REQUIRED_ADAPTER_METHODS, {"mirix", "secom"})

    def test_graphiti_uses_official_extraction_completion_budget(self) -> None:
        argv = ["graphiti", "--input", "input.json", "--output", "output.jsonl"]
        environment = dict(os.environ)
        environment.pop("HABITBENCH_GRAPHITI_LLM_MAX_TOKENS", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            sys, "argv", argv
        ):
            args = parse_graphiti_args()
            self.assertEqual(args.max_tokens, 16384)
            self.assertEqual(args.schema_max_items, 64)
            self.assertEqual(args.schema_max_string_chars, 1000)

    def test_graphiti_local_schema_bounds_are_recursive(self) -> None:
        schema = {
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
        }
        bounded = apply_json_schema_bounds(
            schema,
            max_items=64,
            max_string_chars=1000,
        )
        entities = bounded["properties"]["entities"]
        self.assertEqual(entities["maxItems"], 64)
        self.assertEqual(
            entities["items"]["properties"]["name"]["maxLength"],
            1000,
        )
        self.assertNotIn("maxItems", schema["properties"]["entities"])


if __name__ == "__main__":
    unittest.main()
