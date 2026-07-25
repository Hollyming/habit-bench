import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.medmemorybench_adapters.structured_memory import (
    default_med_repo,
    empty_contexts,
    extract_session_ids,
    group_user_jobs,
    resolve_user_state_root,
    require_successful_memory_build,
    render_session,
    run,
)


class MedMemoryBenchAdapterTest(unittest.TestCase):
    def test_default_source_is_vendored_inside_repository(self):
        source_root = default_med_repo()
        self.assertEqual(source_root.name, "medmemorybench")
        self.assertTrue((source_root / "src" / "agent.py").is_file())

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

    def test_user_jobs_are_deterministic_and_chronological(self):
        jobs = group_user_jobs(
            {
                "probes": [
                    {
                        "probe_id": "u2-late",
                        "user_id": "u2",
                        "visible_history_scope": {"max_session_index": 2},
                    },
                    {
                        "probe_id": "u1-late",
                        "user_id": "u1",
                        "visible_history_scope": {"max_session_index": 3},
                    },
                    {
                        "probe_id": "u1-early",
                        "user_id": "u1",
                        "visible_history_scope": {"max_session_index": 1},
                    },
                ],
                "sessions_by_user": {"u1": [], "u2": []},
            }
        )
        self.assertEqual([job["user_id"] for job in jobs], ["u1", "u2"])
        self.assertEqual(
            [probe["probe_id"] for probe in jobs[0]["probes"]],
            ["u1-early", "u1-late"],
        )
        self.assertEqual([job["context_id"] for job in jobs], [1, 2])

    def test_user_state_root_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_user_state_root(root, "user-1"), root / "user-1")
            with self.assertRaisesRegex(ValueError, "Unsafe user_id"):
                resolve_user_state_root(root, "../escape")


class MedMemoryBenchParallelAdapterTest(unittest.TestCase):
    def _write_fake_med_repo(self, root: Path) -> Path:
        med_repo = root / "fake_med"
        source = med_repo / "src"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("", encoding="utf-8")
        (source / "config.py").write_text(
            textwrap.dedent(
                """
                from types import SimpleNamespace

                class ConfigLoader:
                    def __init__(self, project_root):
                        self.project_root = project_root

                    def load_method_config(self, name):
                        return SimpleNamespace(method_name="fake", agent_params={})

                class DatasetConfig:
                    def __init__(self, dataset_name, language):
                        self.dataset_name = dataset_name
                        self.language = language
                """
            ),
            encoding="utf-8",
        )
        (source / "agent.py").write_text(
            textwrap.dedent(
                """
                import re
                import time
                from types import SimpleNamespace

                class AgentManager:
                    def __init__(self, method_config, dataset_config):
                        self.storage_root = method_config.agent_params["storage_root"]
                        self.sessions = []

                    def set_context_id(self, context_id):
                        self.context_id = context_id

                    def send_message(self, message, memorizing, context_id):
                        time.sleep(0.05)
                        session_id = re.search(
                            r"\\[SESSION_ID=([^\\]]+)\\]", message
                        ).group(1)
                        self.sessions.append(session_id)
                        return SimpleNamespace(success=True, method="fake", extra={})

                    def retrieve(self, question, context_id):
                        memory_context = " ".join(
                            f"[SESSION_ID={session_id}]"
                            for session_id in self.sessions
                        )
                        return {
                            "memory_context": memory_context,
                            "retrieved_count": len(self.sessions),
                            "retrieved_memories": list(self.sessions),
                        }

                    def reset(self):
                        pass
                """
            ),
            encoding="utf-8",
        )
        return med_repo

    def test_parallel_users_preserve_order_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            med_repo = self._write_fake_med_repo(root)
            payload = {
                "probes": [
                    {
                        "probe_id": "u2-p0",
                        "user_id": "u2",
                        "query": "remember?",
                        "visible_history_scope": {"max_session_index": 0},
                    },
                    {
                        "probe_id": "u1-p0",
                        "user_id": "u1",
                        "query": "remember?",
                        "visible_history_scope": {"max_session_index": 0},
                    },
                    {
                        "probe_id": "u1-p1",
                        "user_id": "u1",
                        "query": "remember?",
                        "visible_history_scope": {"max_session_index": 1},
                    },
                ],
                "sessions_by_user": {
                    "u1": [
                        {
                            "session_id": "u1-s0",
                            "session_index": 0,
                            "timestamp": "2026-01-01",
                            "domain": "test",
                            "messages": [{"role": "user", "content": "zero"}],
                        },
                        {
                            "session_id": "u1-s1",
                            "session_index": 1,
                            "timestamp": "2026-01-02",
                            "domain": "test",
                            "messages": [{"role": "user", "content": "one"}],
                        },
                    ],
                    "u2": [
                        {
                            "session_id": "u2-s0",
                            "session_index": 0,
                            "timestamp": "2026-01-01",
                            "domain": "test",
                            "messages": [{"role": "user", "content": "zero"}],
                        }
                    ],
                },
            }
            input_path = root / "input.json"
            output_path = root / "memory_contexts.jsonl"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            run(
                SimpleNamespace(
                    input=input_path,
                    output=output_path,
                    med_repo=med_repo,
                    method_config="fake",
                    state_root=root / "state",
                    user_workers=2,
                    progress_every=0,
                    dry_run_config=False,
                )
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["probe_id"] for row in rows],
                ["u2-p0", "u1-p0", "u1-p1"],
            )
            self.assertEqual(
                rows[2]["debug"]["retrieved_memories"],
                ["u1-s0", "u1-s1"],
            )
            self.assertEqual(
                extract_session_ids(rows[0]["memory_context"]),
                ["u2-s0"],
            )
            runtime = json.loads(
                (root / "medmemorybench_adapter_runtime.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(runtime["user_workers_effective"], 2)
            self.assertEqual(runtime["sessions_added"], 3)
            self.assertEqual(len({row["pid"] for row in runtime["user_runs"]}), 2)


if __name__ == "__main__":
    unittest.main()
