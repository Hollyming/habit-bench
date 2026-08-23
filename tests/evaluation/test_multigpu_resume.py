from __future__ import annotations

import json
import multiprocessing
import tempfile
import threading
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts import merge_shard_plan
from scripts import run_multigpu_plan as runner


def _consume_queue_in_process(
    raw_root: str,
    rows: list[dict[str, str]],
    replica_index: int,
    barrier: object,
    output_queue: object,
) -> None:
    coordinator = runner.DistributedTaskCoordinator(
        Path(raw_root),
        rows,
        coordinator_id="process-job",
        plan_sha256="process-plan-hash",
        replica_count=2,
    )
    barrier.wait()
    claimed = []
    while True:
        item = coordinator.claim_next(
            replica_index=replica_index,
            worker_index=0,
            host=f"process-host-{replica_index}",
        )
        if item is None:
            break
        row, claim = item
        claimed.append(int(row["task_id"]))
        coordinator.finish(
            claim,
            {"status": "succeeded", "returncode": 0},
        )
    output_queue.put(claimed)


class MultiGpuResumeTest(unittest.TestCase):
    @mock.patch.object(runner, "_wait_for_server")
    @mock.patch.object(runner.subprocess, "Popen")
    def test_vllm_workers_receive_disjoint_internal_port_ranges(
        self,
        popen: mock.Mock,
        wait_for_server: mock.Mock,
    ) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        popen.return_value = process
        with tempfile.TemporaryDirectory() as raw_root:
            workers = [
                runner._start_server(
                    worker_index=index,
                    gpu=str(index),
                    port=8100 + index,
                    env={
                        "HABITBENCH_VLLM_INTERNAL_PORT_BASE": "24000",
                        "HABITBENCH_VLLM_INTERNAL_PORT_STRIDE": "64",
                    },
                    log_root=Path(raw_root),
                )
                for index in range(8)
            ]
            self.assertEqual(
                [worker["internal_port"] for worker in workers],
                [24000 + index * 64 for index in range(8)],
            )
            self.assertEqual(
                [
                    call.kwargs["env"]["VLLM_PORT"]
                    for call in popen.call_args_list
                ],
                [str(24000 + index * 64) for index in range(8)],
            )
            self.assertEqual(wait_for_server.call_count, 8)
            for worker in workers:
                worker["log_handle"].close()

    def _complete_shard(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        for name in runner.COMPLETION_FILES:
            path = output_dir / name
            if name == "worker_runtime.json":
                path.write_text(
                    json.dumps({"status": "succeeded", "shard_index": 0}),
                    encoding="utf-8",
                )
            elif name == "run_manifest.json":
                path.write_text(
                    json.dumps({"execution": {"status": "succeeded"}}),
                    encoding="utf-8",
                )
            else:
                path.write_text("{}\n", encoding="utf-8")

    def test_verified_checkpoint_is_reused_without_rewriting(self):
        with tempfile.TemporaryDirectory() as raw_root:
            output_dir = Path(raw_root) / "shard_000_of_016"
            self._complete_shard(output_dir)
            sentinel = output_dir / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")

            self.assertTrue(
                runner._prepare_task_output(output_dir, force_rerun=False)
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_metrics_without_atomic_success_marker_is_discarded(self):
        with tempfile.TemporaryDirectory() as raw_root:
            output_dir = Path(raw_root) / "shard_001_of_016"
            output_dir.mkdir(parents=True)
            (output_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "partial-memory.db").write_text("partial", encoding="utf-8")

            self.assertFalse(
                runner._prepare_task_output(output_dir, force_rerun=False)
            )
            self.assertTrue(output_dir.is_dir())
            self.assertFalse((output_dir / "metrics.json").exists())
            self.assertFalse((output_dir / "partial-memory.db").exists())

    def test_two_replicas_dynamically_claim_each_shard_once(self):
        rows = [
            {
                "task_id": str(index),
                "method": "mem0",
                "dataset_name": "food",
                "method_output_root": "/tmp/results/food/mem0",
                "shard_index": str(index),
                "shard_count": "64",
            }
            for index in range(64)
        ]
        with tempfile.TemporaryDirectory() as raw_root:
            coordinators = [
                runner.DistributedTaskCoordinator(
                    Path(raw_root),
                    rows,
                    coordinator_id="job-123",
                    plan_sha256="plan-hash",
                    replica_count=2,
                )
                for _ in range(2)
            ]
            barrier = threading.Barrier(16)

            def consume(replica_index: int, worker_index: int) -> list[int]:
                barrier.wait()
                local = []
                while True:
                    item = coordinators[replica_index].claim_next(
                        replica_index=replica_index,
                        worker_index=worker_index,
                        host=f"host-{replica_index}",
                    )
                    if item is None:
                        return local
                    row, claim = item
                    local.append(int(row["task_id"]))
                    coordinators[replica_index].finish(
                        claim,
                        {"status": "succeeded", "returncode": 0},
                    )

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(consume, replica_index, worker_index)
                    for replica_index in range(2)
                    for worker_index in range(8)
                ]
                claimed = [
                    task_id for future in futures for task_id in future.result()
                ]

            self.assertEqual(sorted(claimed), list(range(64)))
            self.assertEqual(len(claimed), len(set(claimed)))
            self.assertIsNone(
                coordinators[0].claim_next(
                    replica_index=0,
                    worker_index=0,
                    host="host-0",
                )
            )
            state = coordinators[0].aggregate_state()
            self.assertEqual(state["claim_protocol"], "atomic-mkdir-per-task")
            self.assertEqual(state["claimed"], 64)
            self.assertEqual(state["finished"], 64)
            self.assertEqual(state["failed"], 0)
            self.assertEqual(state["active"], 0)
            self.assertEqual(state["unclaimed"], 0)

    def test_atomic_claims_are_unique_across_processes(self):
        rows = [
            {
                "task_id": str(index),
                "method": "mem0",
                "dataset_name": "food",
                "method_output_root": "/tmp/results/food/mem0",
                "shard_index": str(index),
                "shard_count": "128",
            }
            for index in range(128)
        ]
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as raw_root:
            barrier = context.Barrier(2)
            output_queue = context.Queue()
            processes = [
                context.Process(
                    target=_consume_queue_in_process,
                    args=(raw_root, rows, replica_index, barrier, output_queue),
                )
                for replica_index in range(2)
            ]
            for process in processes:
                process.start()
            claims = [output_queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            flattened = [task_id for claim_set in claims for task_id in claim_set]
            self.assertEqual(sorted(flattened), list(range(128)))
            self.assertEqual(len(flattened), len(set(flattened)))
            coordinator = runner.DistributedTaskCoordinator(
                Path(raw_root),
                rows,
                coordinator_id="process-job",
                plan_sha256="process-plan-hash",
                replica_count=2,
            )
            state = coordinator.aggregate_state()
            self.assertEqual(state["claimed"], 128)
            self.assertEqual(state["finished"], 128)
            self.assertEqual(state["active"], 0)

    def test_replica_runtime_merge_preserves_all_shard_records(self):
        with tempfile.TemporaryDirectory() as raw_root:
            paths = []
            for replica_index in range(2):
                path = Path(raw_root) / f"suite_runtime.replica-00{replica_index}-of-002.json"
                payload = {
                    "status": "succeeded",
                    "started_at": f"2026-08-20T00:0{replica_index}:00+00:00",
                    "finished_at": f"2026-08-20T00:1{replica_index}:00+00:00",
                    "host": f"host-{replica_index}",
                    "plan": "/tmp/plan.tsv",
                    "plan_sha256": "same",
                    "plan_manifest": "/tmp/plan.manifest.json",
                    "launcher": {"replicas": "2"},
                    "replica_index": replica_index,
                    "replica_count": 2,
                    "gpu_count": 8,
                    "gpus": [str(index) for index in range(8)],
                    "task_count": 1,
                    "global_task_count": 2,
                    "config": {"profile": "same"},
                    "preflight": {"status": "pass"},
                    "groups": [
                        {
                            "method": "mem0",
                            "dataset_name": "food",
                            "method_output_root": "/tmp/results/food/mem0",
                            "started_at": f"2026-08-20T00:0{replica_index}:00+00:00",
                            "finished_at": f"2026-08-20T00:1{replica_index}:00+00:00",
                            "wall_clock_sec": 600,
                            "status": "succeeded",
                            "tasks": [
                                {
                                    "shard_index": replica_index,
                                    "status": "succeeded",
                                }
                            ],
                        }
                    ],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)

            combined = merge_shard_plan._combine_replica_runtimes(paths)
            self.assertEqual(combined["status"], "succeeded")
            self.assertEqual(combined["gpu_count"], 16)
            self.assertEqual(combined["task_count"], 2)
            self.assertEqual(
                [task["shard_index"] for task in combined["groups"][0]["tasks"]],
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
