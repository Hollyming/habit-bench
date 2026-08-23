from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDMEMORY_ROOT = PROJECT_ROOT / "third_party" / "medmemorybench"
MEMOS_SRC = MEDMEMORY_ROOT / "methods" / "memOS" / "MemOS" / "src"
for source_root in (MEDMEMORY_ROOT, MEMOS_SRC):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

from methods.mem0.memory.main import Memory
from memos.configs.mem_user import UserManagerConfigFactory
from memos.mem_os.core import MOSCore
from memos.mem_user.user_manager import UserManager
from scripts import run_multigpu_plan as runner


def _initialize_shared_user_db(
    db_path: str,
    barrier: object,
    output_queue: object,
) -> None:
    barrier.wait()
    try:
        manager = UserManager(db_path=db_path, user_id="root")
        valid = manager.validate_user("root")
        manager.engine.dispose()
        output_queue.put({"ok": valid})
    except BaseException as exc:
        output_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class _SequencedLlm:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _MosConfig:
    def __init__(self, db_path: str):
        self.user_id = "benchmark-user"
        self.session_id = "benchmark-session"
        self.chat_model = object()
        self.mem_reader = object()
        self.user_manager = UserManagerConfigFactory(
            backend="sqlite",
            config={"db_path": db_path, "user_id": "root"},
        )
        self.enable_mem_scheduler = False

    def get(self, name: str, default=None):
        return getattr(self, name, default)


class _ValidUserManager:
    def validate_user(self, user_id: str) -> bool:
        return user_id == "benchmark-user"


class MemoryRuntimeRecoveryTest(unittest.TestCase):
    def test_h_launch_contract_requires_letta_cl100k_cache(self):
        expected_key = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
        self.assertEqual(runner.CL100K_CACHE_KEY, expected_key)

        create_script = (
            PROJECT_ROOT / "scripts" / "cluster" / "create_h_envs.sh"
        ).read_text(encoding="utf-8")
        submit_script = (PROJECT_ROOT / "scripts" / "submit_h_cluster.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cl100k_base"', create_script)
        self.assertIn(expected_key, create_script)
        self.assertIn("NEEDS_CL100K_CACHE", submit_script)
        self.assertIn(expected_key, submit_script)

    def test_mem0_retries_invalid_fact_json_from_scratch(self):
        memory = Memory.__new__(Memory)
        memory.custom_fact_extraction_prompt = "Return facts JSON."
        memory.strict_json_schema = True
        memory._fact_generation_attempts = 3
        memory.llm = _SequencedLlm(
            [
                '{"facts": ["unterminated"',
                json.dumps({"facts": ["valid fact"]}),
            ]
        )

        facts = memory._extract_facts("session text")

        self.assertEqual(facts, ["valid fact"])
        self.assertEqual(len(memory.llm.calls), 2)
        retry_prompt = memory.llm.calls[1]["messages"][1]["content"]
        self.assertIn("Regenerate the complete answer from scratch", retry_prompt)

    def test_mem0_stops_after_bounded_invalid_fact_json_attempts(self):
        memory = Memory.__new__(Memory)
        memory.custom_fact_extraction_prompt = "Return facts JSON."
        memory.strict_json_schema = False
        memory._fact_generation_attempts = 2
        memory.llm = _SequencedLlm(["not-json", "still-not-json"])

        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            memory._extract_facts("session text")
        self.assertEqual(len(memory.llm.calls), 2)

    def test_parallel_sqlite_bootstrap_is_process_safe(self):
        # H-cluster adapter workers use Linux processes.  Fork here avoids
        # re-importing the full vendored MemOS/ML dependency graph in every
        # test child while preserving independent file descriptors and flock
        # state, which are the concurrency boundary under test.
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "shared-users.db")
            worker_count = 8
            barrier = context.Barrier(worker_count)
            output_queue = context.Queue()
            processes = [
                context.Process(
                    target=_initialize_shared_user_db,
                    args=(db_path, barrier, output_queue),
                )
                for _ in range(worker_count)
            ]
            for process in processes:
                process.start()
            results = [output_queue.get(timeout=60) for _ in processes]
            for process in processes:
                process.join(timeout=60)
                self.assertEqual(process.exitcode, 0)

            self.assertTrue(all(result["ok"] for result in results), results)
            manager = UserManager(db_path=db_path, user_id="root")
            self.assertTrue(manager.validate_user("root"))
            self.assertEqual(len(manager.list_users()), 1)
            manager.engine.dispose()

    def test_mos_core_honours_configured_user_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "isolated-users.db")
            config = _MosConfig(db_path)
            with (
                patch(
                    "memos.mem_os.core.LLMFactory.from_config", return_value=object()
                ),
                patch(
                    "memos.mem_os.core.MemReaderFactory.from_config",
                    return_value=object(),
                ),
                patch(
                    "memos.mem_os.core.UserManagerFactory.from_config",
                    return_value=_ValidUserManager(),
                ) as manager_factory,
            ):
                core = MOSCore(config)

            factory_config = manager_factory.call_args.args[0]
            self.assertEqual(factory_config.config.db_path, db_path)
            self.assertEqual(factory_config.config.user_id, "benchmark-user")
            self.assertEqual(config.user_manager.config.user_id, "root")
            self.assertIsInstance(core.user_manager, _ValidUserManager)


if __name__ == "__main__":
    unittest.main()
