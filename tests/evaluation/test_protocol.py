from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from eval.core.answering import DEFAULT_SERVED_MODEL
from eval.core.dataset import DatasetContractError
from eval.run import validate_memory_contexts
from scripts.run_multigpu_plan import (
    CUDA_REQUIRED_ADAPTER_METHODS,
    COMPLETION_FILES,
    _inspect_vllm_runtime,
    _prepare_task_output,
    _validate_mirix_vllm_profile,
)
from scripts.create_shard_plan import (
    DEFAULT_DATASETS,
    SUPPLEMENTARY_DIAGNOSTIC_METHODS,
    _effective_method_config,
)


class ProtocolTest(unittest.TestCase):

    def test_vllm_runtime_accepts_official_torch_cuda_local_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            python_stub = Path(temp_dir) / "python"
            python_stub.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '{\"flashinfer-python\":\"0.6.4\","
                "\"python\":\"3.10.20\",\"torch\":\"2.10.0+cu128\","
                "\"torch_cuda\":\"12.8\",\"transformers\":\"4.57.6\","
                "\"triton\":\"3.6.0\",\"vllm\":\"0.17.1\","
                "\"xgrammar\":\"0.1.29\"}'\n",
                encoding="utf-8",
            )
            python_stub.chmod(0o755)
            versions = _inspect_vllm_runtime(python_stub, os.environ.copy())
        self.assertEqual(versions["torch"], "2.10.0+cu128")

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
        self.assertEqual(registry["full_memory"]["revision"], "memory_context.v5")
        self.assertEqual(registry["full_memory"]["adapter"], "eval/compact_history.py")
        self.assertEqual(registry["full_history"]["revision"], "memory_context.v3")

    def test_non_agentic_retrieval_baselines_are_registered(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        registry = json.loads(
            (project_root / "eval/methods.json").read_text(encoding="utf-8")
        )
        expected = {
            "recency_5",
            "recency_10",
            "bm25_rag",
            "dense_rag",
            "temporal_hybrid_rag",
        }
        for method in expected:
            with self.subTest(method=method):
                self.assertEqual(
                    registry[method]["implementation_kind"],
                    "retrieval_baseline",
                )
                self.assertEqual(
                    registry[method]["adapter"],
                    "eval/retrieval_baselines.py",
                )
                self.assertTrue(
                    (project_root / f"configs/methods/{method}.yaml").is_file()
                )

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

    def test_supplementary_planner_is_current_four_domain(self) -> None:
        self.assertEqual(
            set(DEFAULT_DATASETS) - {"finance_software"},
            {"food", "finance", "software", "travel"},
        )
        self.assertIn("food_habit_lifelines_stress_v5", str(DEFAULT_DATASETS["food"]))
        self.assertIn("release_gated_v1_4", str(DEFAULT_DATASETS["finance"]))
        self.assertIn("release_candidate_v16", str(DEFAULT_DATASETS["travel"]))
        self.assertEqual(
            SUPPLEMENTARY_DIAGNOSTIC_METHODS,
            {"oracle_evidence", "oracle_habit_state"},
        )

    def test_qwen3_8b_supplementary_launcher_excludes_human_audit(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = (
            project_root / "scripts/cluster/submit_qwen3_8b_supplementary.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--replicas 2", launcher)
        self.assertIn("--gpus 8", launcher)
        self.assertIn("--shards 16", launcher)
        self.assertIn("no_memory,oracle_evidence,oracle_habit_state", launcher)
        self.assertIn("food,finance,software,travel", launcher)
        self.assertIn("--post-supplementary-analysis", launcher)
        self.assertNotIn("human_audit.py", launcher)

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
            '\\"disable_any_whitespace\\":true,'
            '\\"disable_fallback\\":true',
            environment_profile,
        )

    def test_h_cluster_launcher_keeps_rjob_classes_separate(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = (project_root / "scripts/submit_h_cluster.sh").read_text(
            encoding="utf-8"
        )
        worker = (project_root / "scripts/cluster/run_h_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--namespace ailab-llmarchitecture", launcher)
        self.assertIn("--charged-group llmarchitecture_gpu", launcher)
        self.assertIn("--private-machine group", launcher)
        self.assertIn("--priority 1", launcher)
        self.assertIn("--priority 5", launcher)
        cluster_entry = (
            "http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture."
            "svc.pjlab.local:11451"
        )
        self.assertIn(
            f'KUBEBRAIN_CLUSTER_ENTRY_REQUIRED="{cluster_entry}"', launcher
        )
        self.assertGreaterEqual(
            launcher.count(
                'export KUBEBRAIN_CLUSTER_ENTRY='
                '"$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"'
            ),
            2,
        )
        self.assertIn(
            '--metadata "kubebrain_cluster_entry=$KUBEBRAIN_CLUSTER_ENTRY"',
            launcher,
        )
        idle_branch = launcher.rsplit("  idle)\n", 1)[1].split("    ;;", 1)[0]
        self.assertIn("--task-type idle", idle_branch)
        self.assertIn("--restart-policy never", idle_branch)
        self.assertNotIn("--priority", idle_branch)
        self.assertNotIn("--charged-group", idle_branch)
        self.assertNotIn("--private-machine", idle_branch)
        self.assertIn('-P "$REPLICAS"', launcher)
        self.assertIn("--host-network=true", launcher)
        self.assertIn("-e DISTRIBUTED_JOB=true", launcher)
        self.assertIn('[[ "$GPUS" != "4" && "$GPUS" != "8" ]]', launcher)
        self.assertIn('[[ "$gpu_name" != *H200* ]]', worker)
        self.assertIn('VLLM_ENV_LIB="$(dirname', worker)
        self.assertIn('export LD_LIBRARY_PATH="$env_lib', worker)
        self.assertIn("-c 'import sqlite3'", worker)

    def test_h_external_api_launcher_is_globally_limited_and_h_compliant(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = (project_root / "scripts/submit_h_api_suite.sh").read_text(
            encoding="utf-8"
        )
        worker = (
            project_root / "scripts/cluster/run_h_api_suite.sh"
        ).read_text(encoding="utf-8")
        cluster_entry = (
            "http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture."
            "svc.pjlab.local:11451"
        )
        self.assertIn(
            f'KUBEBRAIN_CLUSTER_ENTRY_REQUIRED="{cluster_entry}"', launcher
        )
        self.assertIn("--namespace ailab-llmarchitecture", launcher)
        self.assertIn("--gpu 8", launcher)
        self.assertIn("-P 1", launcher)
        self.assertIn("--priority 1", launcher)
        self.assertIn("--priority 5", launcher)
        self.assertIn("--charged-group llmarchitecture_gpu", launcher)
        self.assertIn("--private-machine group", launcher)
        idle_branch = launcher.rsplit("  idle)\n", 1)[1].split("    ;;", 1)[0]
        self.assertIn("--task-type idle", idle_branch)
        self.assertNotIn("--priority", idle_branch)
        self.assertNotIn("--charged-group", idle_branch)
        self.assertNotIn("--private-machine", idle_branch)
        self.assertIn('CREDENTIAL_FILE="${HABITBENCH_API_CREDENTIAL_FILE:-}"', launcher)
        self.assertNotIn("OPENAI_API_KEY=sk-", launcher)
        self.assertIn('MODELS="deepseek-v4-pro-0813,glm-5.2,kimi-k3"', launcher)
        self.assertIn('GPU_ALLOCATIONS="3,3,2"', launcher)
        self.assertIn("--rpm \"$RPM\"", worker)
        self.assertIn("--tpm \"$TPM\"", worker)
        self.assertIn('--credential-slots "$API_KEY_COUNT"', launcher)
        self.assertIn(
            'HABITBENCH_ANSWER_MAX_TOKENS="${HABITBENCH_ANSWER_MAX_TOKENS:-4096}"',
            worker,
        )
        self.assertIn('flock -n 9', worker)
        self.assertEqual(worker.count("-m eval.api_gateway"), 1)
        self.assertIn('cd "$PROJECT_ROOT"', worker)
        self.assertIn(
            'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
            worker,
        )

    def test_h_launcher_default_job_names_use_zjm_prefix(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        paths = [
            project_root / "scripts/submit_h_cluster.sh",
            project_root / "scripts/submit_h_api_suite.sh",
            project_root / "scripts/cluster/submit_qwen3_4b_main.sh",
            project_root / "scripts/cluster/submit_qwen3_14b_main.sh",
            project_root / "scripts/cluster/submit_qwen3_32b_main.sh",
            project_root / "scripts/cluster/submit_qwen3_8b_supplementary.sh",
        ]
        for path in paths:
            assignments = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("JOB_NAME=")
                and ("JOB_NAME:-" in line or "_JOB_NAME:-" in line)
            ]
            self.assertEqual(len(assignments), 1, path)
            self.assertIn(":-zjm-", assignments[0], path)

    def test_h_cluster_profile_separates_shared_assets_and_clone_state(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        profile = project_root / "scripts/cluster/env.h.example.sh"
        clone_root = "/mnt/shared-storage-gpfs2/plm-gpfs/alice/workspace/habit-bench"
        command = (
            f'source "{profile}"\n'
            "printf '%s\\n' "
            '"$HABITBENCH_H_SHARED_ROOT" "$HABITBENCH_PROJECT_ROOT" '
            '"$HABITBENCH_LLM_MODEL" "$HF_HOME" "$XDG_CACHE_HOME" '
            '"$VLLM_CACHE_ROOT" "$TIKTOKEN_CACHE_DIR"\n'
        )
        completed = subprocess.run(
            ["bash", "-c", command],
            check=True,
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ["PATH"],
                "HABITBENCH_PROJECT_ROOT": clone_root,
            },
        )
        values = completed.stdout.splitlines()
        shared_root = "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang"
        clone_cache = f"{clone_root}/results/.cache/habitbench"
        self.assertEqual(
            values,
            [
                shared_root,
                clone_root,
                f"{shared_root}/models/habitbench/Qwen3-8B",
                f"{clone_cache}/huggingface",
                clone_cache,
                f"{clone_cache}/vllm",
                f"{shared_root}/.cache/tiktoken",
            ],
        )

    def test_h_cluster_launcher_supports_shared_and_user_mounts(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = (project_root / "scripts/submit_h_cluster.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MOUNT_CONFIGS=()", launcher)
        self.assertIn('MOUNT_CONFIGS+=("${2:?missing value for --mount}")', launcher)
        self.assertIn('gpfs://gpfs2/plm-gpfs/$owner_name:', launcher)
        self.assertIn('path_is_mounted "$mounted_path"', launcher)
        self.assertIn('--mount "${MOUNT_CONFIGS[@]}"', launcher)
        for cache_name in (
            "HF_HOME",
            "XDG_CACHE_HOME",
            "VLLM_CACHE_ROOT",
            "TORCH_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        ):
            self.assertIn(f'  "${cache_name}"', launcher)

    def test_qwen3_14b_scale_launcher_is_a_distinct_two_node_run(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = (
            project_root / "scripts/cluster/submit_qwen3_14b_main.sh"
        ).read_text(encoding="utf-8")
        environment = (
            project_root / "scripts/cluster/env.h.qwen3_14b.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--job-type reserved", launcher)
        self.assertIn("--gpus 8", launcher)
        self.assertIn("--replicas 2", launcher)
        self.assertIn("--shards 16", launcher)
        self.assertIn(
            'HABITBENCH_RESULTS_ROOT:-$PROJECT_ROOT/results', launcher
        )
        self.assertNotIn("/mnt/shared-storage-gpfs2", launcher)
        self.assertIn("habit-h200-main-qwen3-14b-v1", launcher)
        self.assertIn('HABITBENCH_SERVED_MODEL="Qwen3-14B"', environment)
        self.assertIn('HABITBENCH_LLM_MODEL_ID="Qwen/Qwen3-14B"', environment)

    def test_qwen3_4b_and_32b_scale_launchers_are_distinct_two_node_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        profiles = {
            "4": (
                "1cfa9a7208912126459214e8b04321603b3df60c",
                "habit-h200-main-qwen3-4b-v1",
            ),
            "32": (
                "9216db5781bf21249d130ec9da846c4624c16137",
                "habit-h200-main-qwen3-32b-v1",
            ),
        }
        for scale, (revision, output_name) in profiles.items():
            with self.subTest(scale=scale):
                launcher = (
                    project_root
                    / f"scripts/cluster/submit_qwen3_{scale}b_main.sh"
                ).read_text(encoding="utf-8")
                environment = (
                    project_root / f"scripts/cluster/env.h.qwen3_{scale}b.sh"
                ).read_text(encoding="utf-8")
                self.assertIn("--job-type reserved", launcher)
                self.assertIn("--gpus 8", launcher)
                self.assertIn("--replicas 2", launcher)
                self.assertIn("--shards 16", launcher)
                self.assertIn(
                    'HABITBENCH_RESULTS_ROOT:-$PROJECT_ROOT/results', launcher
                )
                self.assertNotIn("/mnt/shared-storage-gpfs2", launcher)
                self.assertIn(output_name, launcher)
                self.assertIn(
                    f'HABITBENCH_SERVED_MODEL="Qwen3-{scale}B"',
                    environment,
                )
                self.assertIn(
                    f'HABITBENCH_LLM_MODEL_ID="Qwen/Qwen3-{scale}B"',
                    environment,
                )
                self.assertIn(
                    f'HABITBENCH_LLM_MODEL_REVISION="{revision}"',
                    environment,
                )

    def test_plan_snapshot_can_record_h_cluster_embedding_path(self) -> None:
        effective, overrides = _effective_method_config(
            {
                "embedding": {
                    "provider": "local",
                    "model": "BAAI/bge-m3",
                    "model_path": "/plm-shared/legacy/bge-m3",
                },
                "history": {"tokenizer_path": "/plm-shared/legacy/qwen3"},
                "memory": {
                    "compressor_model_path": "/plm-shared/legacy/llmlingua2"
                },
                "agent_params": {
                    "embedding_model_path": "/plm-shared/legacy/bge-m3",
                    "topic_segmenter_model_path": "/plm-shared/legacy/llmlingua2",
                },
            },
            embedding_model_path=Path("/mnt/shared-storage-gpfs2/models/bge-m3"),
            llm_model_path=Path("/mnt/shared-storage-gpfs2/models/qwen3"),
            compressor_model_path=Path(
                "/mnt/shared-storage-gpfs2/models/llmlingua2"
            ),
        )
        self.assertEqual(
            effective["embedding"]["model_path"],
            "/mnt/shared-storage-gpfs2/models/bge-m3",
        )
        self.assertEqual(
            overrides["embedding.model_path"],
            "/mnt/shared-storage-gpfs2/models/bge-m3",
        )
        self.assertEqual(
            effective["agent_params"]["embedding_model_path"],
            "/mnt/shared-storage-gpfs2/models/bge-m3",
        )
        self.assertEqual(
            effective["history"]["tokenizer_path"],
            "/mnt/shared-storage-gpfs2/models/qwen3",
        )
        self.assertEqual(
            effective["memory"]["compressor_model_path"],
            "/mnt/shared-storage-gpfs2/models/llmlingua2",
        )
        self.assertEqual(
            effective["agent_params"]["topic_segmenter_model_path"],
            "/mnt/shared-storage-gpfs2/models/llmlingua2",
        )

    def test_plan_snapshot_records_scale_ablation_model_identity(self) -> None:
        effective, overrides = _effective_method_config(
            {
                "description": "HABIT profile: local Qwen3-8B",
                "model": {"name": "Qwen3-8B"},
                "answer_model": {"name": "Qwen3-8B"},
                "history": {"compactor": {"name": "Qwen3-8B"}},
                "agent_params": {
                    "amem_model": "Qwen3-8B",
                    "memos_model": "Qwen3-8B",
                },
            },
            embedding_model_path=Path("/mnt/shared-storage-gpfs2/models/bge-m3"),
            served_model_name="Qwen3-14B",
        )
        self.assertEqual(
            effective["description"],
            "HABIT profile: local Qwen3-14B",
        )
        self.assertEqual(
            overrides["description"],
            "HABIT profile: local Qwen3-14B",
        )
        self.assertEqual(effective["model"]["name"], "Qwen3-14B")
        self.assertEqual(effective["answer_model"]["name"], "Qwen3-14B")
        self.assertEqual(
            effective["history"]["compactor"]["name"],
            "Qwen3-14B",
        )
        self.assertEqual(
            effective["agent_params"]["amem_model"],
            "Qwen3-14B",
        )
        self.assertEqual(
            overrides["agent_params.memos_model"],
            "Qwen3-14B",
        )

    def test_interrupted_shard_state_is_removed_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "mem0" / "shard_000_of_004"
            output_dir.mkdir(parents=True)
            (output_dir / "partial-state.db").write_text(
                "incomplete",
                encoding="utf-8",
            )
            was_complete = _prepare_task_output(
                output_dir,
                force_rerun=False,
            )
            self.assertFalse(was_complete)
            self.assertFalse((output_dir / "partial-state.db").exists())
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

            (output_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
            was_complete = _prepare_task_output(
                output_dir,
                force_rerun=False,
            )
            self.assertFalse(was_complete)

            for name in COMPLETION_FILES:
                path = output_dir / name
                if name == "worker_runtime.json":
                    payload = {"status": "succeeded"}
                elif name == "run_manifest.json":
                    payload = {"execution": {"status": "succeeded"}}
                else:
                    payload = {}
                path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                _prepare_task_output(output_dir, force_rerun=False)
            )

    def test_mirix_preflight_requires_compact_xgrammar(self) -> None:
        profile = _validate_mirix_vllm_profile(
            {
                "HABITBENCH_VLLM_EXTRA_ARGS": (
                    "--dtype bfloat16 --structured-outputs-config "
                    '\'{"backend":"xgrammar","disable_any_whitespace":true,'
                    '"disable_fallback":true}\''
                )
            }
        )
        self.assertEqual(profile["backend"], "xgrammar")
        with self.assertRaisesRegex(ValueError, "MIRIX requires"):
            _validate_mirix_vllm_profile(
                {"HABITBENCH_VLLM_EXTRA_ARGS": "--dtype bfloat16"}
            )
        with self.assertRaisesRegex(ValueError, "must not enable"):
            _validate_mirix_vllm_profile(
                {
                    "HABITBENCH_VLLM_EXTRA_ARGS": (
                        "--reasoning-parser qwen3 "
                        "--structured-outputs-config "
                        "'{\"backend\":\"xgrammar\","
                        "\"disable_any_whitespace\":true,"
                        "\"disable_fallback\":true}'"
                    )
                }
            )

    def test_default_vllm_profile_has_no_reasoning_parser(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        profile_source = (
            project_root / "scripts/cluster/env.example.sh"
        ).read_text(encoding="utf-8")
        default_assignment = next(
            line
            for line in profile_source.splitlines()
            if line.startswith("export HABITBENCH_VLLM_EXTRA_ARGS=")
        )
        self.assertNotIn("--reasoning-parser", default_assignment)
        self.assertIn("--tool-call-parser hermes", default_assignment)

    def test_context_coverage(self) -> None:
        rows = [{"probe_id": "p1", "memory_context": "context"}]
        self.assertEqual(validate_memory_contexts(rows, ["p1"])["p1"]["memory_context"], "context")

    def test_native_cuda_adapters_are_bound_to_their_worker_gpu(self) -> None:
        self.assertEqual(CUDA_REQUIRED_ADAPTER_METHODS, {"mirix", "secom"})

if __name__ == "__main__":
    unittest.main()
