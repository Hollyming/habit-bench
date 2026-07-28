from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from eval.core.answering import DEFAULT_SERVED_MODEL
from eval.core.dataset import DatasetContractError
from eval.run import validate_memory_contexts
from scripts.run_multigpu_plan import (
    CUDA_REQUIRED_ADAPTER_METHODS,
    _validate_mirix_vllm_profile,
)
from scripts.create_v3_experiment_plans import SHARDS, _group_is_complete


class ProtocolTest(unittest.TestCase):
    METHODS = ("mem0", "amem", "memos", "memrl", "lightmem", "letta", "mirix")
    OFFICIAL_METHODS = {
        "secom": "eval/official_adapters/secom.py",
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
        expected = {"SeCom": "secom/secom.py"}
        for source, marker in expected.items():
            source_root = vendor_root / source
            self.assertTrue((source_root / marker).is_file())
            self.assertFalse((source_root / ".git").exists())

    def test_problematic_methods_are_explicitly_unsupported(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        registry = json.loads(
            (project_root / "eval/methods.json").read_text(encoding="utf-8")
        )
        unsupported = json.loads(
            (project_root / "eval/unsupported_methods.json").read_text(
                encoding="utf-8"
            )
        )
        for method in ("graphiti", "omem"):
            self.assertNotIn(method, registry)
            self.assertEqual(unsupported[method]["status"], "not_implemented")
            self.assertIn("reason", unsupported[method])

    def test_v3_resume_requires_all_shards_to_be_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_root = Path(temporary)
            method_root = suite_root / "food" / "mem0"
            for index in range(SHARDS - 1):
                shard = method_root / f"shard_{index:03d}_of_{SHARDS:03d}"
                shard.mkdir(parents=True)
                (shard / "metrics.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(_group_is_complete(suite_root, "mem0", "food"))
            last = method_root / f"shard_{SHARDS - 1:03d}_of_{SHARDS:03d}"
            last.mkdir(parents=True)
            (last / "metrics.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(_group_is_complete(suite_root, "mem0", "food"))

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

    def test_cluster_structured_output_disallows_unbounded_whitespace(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        environment_profile = (
            project_root / "scripts/cluster/env.example.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--structured-outputs-config", environment_profile)
        self.assertIn(
            '\\"backend\\":\\"xgrammar\\",'
            '\\"disable_any_whitespace\\":true',
            environment_profile,
        )

    def test_mirix_preflight_requires_compact_xgrammar(self) -> None:
        profile = _validate_mirix_vllm_profile(
            {
                "HABITBENCH_VLLM_EXTRA_ARGS": (
                    "--dtype bfloat16 --structured-outputs-config "
                    '\'{"backend":"xgrammar","disable_any_whitespace":true}\''
                )
            }
        )
        self.assertEqual(profile["backend"], "xgrammar")
        with self.assertRaisesRegex(ValueError, "MIRIX requires"):
            _validate_mirix_vllm_profile(
                {"HABITBENCH_VLLM_EXTRA_ARGS": "--dtype bfloat16"}
            )

    def test_context_coverage(self) -> None:
        rows = [{"probe_id": "p1", "memory_context": "context"}]
        self.assertEqual(validate_memory_contexts(rows, ["p1"])["p1"]["memory_context"], "context")

    def test_native_cuda_adapters_are_bound_to_their_worker_gpu(self) -> None:
        self.assertEqual(CUDA_REQUIRED_ADAPTER_METHODS, {"mirix", "secom"})

if __name__ == "__main__":
    unittest.main()
