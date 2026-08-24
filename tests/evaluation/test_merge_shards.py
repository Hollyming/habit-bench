from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import write_json, write_jsonl
from eval.merge_shards import merge_shards


def _session(user_id: str, domain: str = "unknown") -> dict:
    return {
        "session_id": f"{user_id}-s0",
        "user_id": user_id,
        "session_index": 0,
        "domain": domain,
        "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "response"},
        ],
    }


def _write_fixture(root: Path, domain: str = "unknown") -> Path:
    dataset = root / "dataset"
    users = ["u0", "u1"]
    write_jsonl(
        dataset / "public/lifelines.jsonl",
        [_session(user, domain) for user in users],
    )
    write_jsonl(
        dataset / "public/probes.jsonl",
        [
            {
                "probe_id": f"p-{user}",
                "user_id": user,
                "domain": domain,
                "query": "q",
                "choices": [
                    {"choice_id": "A", "text": "a"},
                    {"choice_id": "B", "text": "b"},
                ],
            }
            for user in users
        ],
    )
    write_jsonl(
        dataset / "private/probe_key.jsonl",
        [
            {"probe_id": f"p-{user}", "gold_choice_id": "A"}
            for user in users
        ],
    )
    return dataset


def _write_shard(
    dataset: Path,
    shard_root: Path,
    index: int,
    count: int,
    domain_filter: str | None = None,
    max_users: int | None = None,
    max_probes: int | None = None,
    method_config: object = None,
) -> None:
    bundle = load_dataset(
        dataset,
        domain_filter=domain_filter,
        user_shard_index=index,
        user_shard_count=count,
        max_users=max_users,
        max_probes=max_probes,
    )
    shard_dir = shard_root / f"shard_{index:03d}_of_{count:03d}"
    implementation = {"kind": "control", "source": "test", "revision": "1"}
    base_model = {"model": "test-model"}
    write_json(
        shard_dir / "run_manifest.json",
        {
            "method_name": "no_memory",
            "implementation": implementation,
            "method_config": method_config,
            "base_model": base_model,
            "dataset": bundle.manifest,
            "adapter_runtime": {"elapsed_sec": 1},
            "answer_runtime": {"elapsed_sec": 1},
            "execution": {
                "started_at": f"2026-01-01T00:00:0{index}+00:00",
                "finished_at": f"2026-01-01T00:00:{10 + index:02d}+00:00",
                "wall_clock_sec": 10 + (2 * index),
                "host": "test-host",
                "cuda_visible_devices": str(index),
            },
            "result": {"accuracy": 1.0},
        },
    )
    write_jsonl(
        shard_dir / "memory_contexts.jsonl",
        [
            {"probe_id": probe["probe_id"], "memory_context": ""}
            for probe in bundle.probes
        ],
    )
    write_jsonl(
        shard_dir / "predictions.jsonl",
        [
            {"probe_id": probe["probe_id"], "choice_id": "A"}
            for probe in bundle.probes
        ],
    )


class MergeShardsTest(unittest.TestCase):
    def test_merge_rescores_complete_probe_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            for index in range(2):
                _write_shard(dataset, shard_root, index, 2)

            manifest = merge_shards(
                dataset, shard_root, root / "merged", "no_memory", 2
            )
            self.assertEqual(manifest["result"]["accuracy"], 1.0)
            self.assertEqual(manifest["result"]["total"], 2)
            self.assertEqual(manifest["timing"]["shard_count"], 2)
            self.assertEqual(manifest["timing"]["shard_wall_clock_sum_sec"], 22.0)
            self.assertEqual(manifest["timing"]["shard_wall_clock_max_sec"], 12.0)
            self.assertEqual(manifest["timing"]["observed_window_sec"], 11.0)

    def test_merge_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            _write_shard(dataset, shard_root, 0, 2)

            with self.assertRaisesRegex(DatasetContractError, "Shard coverage mismatch"):
                merge_shards(dataset, shard_root, root / "merged", "no_memory", 2)

    def test_merge_ignores_comment_only_method_config_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            for index, digest in enumerate(("old-hash", "new-hash")):
                _write_shard(
                    dataset,
                    shard_root,
                    index,
                    2,
                    method_config={
                        "name": "mirix",
                        "path": "/configs/mirix.yaml",
                        "sha256": digest,
                        "config": {"agent_params": {"attempts": 5}},
                    },
                )

            manifest = merge_shards(
                dataset, shard_root, root / "merged", "no_memory", 2
            )
            self.assertEqual(manifest["result"]["total"], 2)

    def test_merge_rejects_behavioral_method_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            for index, attempts in enumerate((4, 5)):
                _write_shard(
                    dataset,
                    shard_root,
                    index,
                    2,
                    method_config={
                        "name": "mirix",
                        "path": "/configs/mirix.yaml",
                        "sha256": str(attempts),
                        "config": {"agent_params": {"attempts": attempts}},
                    },
                )

            with self.assertRaisesRegex(
                DatasetContractError, "Method-config mismatch"
            ):
                merge_shards(dataset, shard_root, root / "merged", "no_memory", 2)

    def test_merge_rescores_declared_smoke_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            for index in range(2):
                _write_shard(
                    dataset,
                    shard_root,
                    index,
                    2,
                    max_users=2,
                    max_probes=2,
                )

            manifest = merge_shards(
                dataset,
                shard_root,
                root / "merged",
                "no_memory",
                2,
                max_users=2,
                max_probes=2,
            )
            self.assertEqual(manifest["result"]["total"], 2)
            self.assertEqual(manifest["dataset"]["subset"]["max_users"], 2)
            self.assertEqual(manifest["dataset"]["subset"]["max_probes"], 2)

            with self.assertRaisesRegex(DatasetContractError, "max-users mismatch"):
                merge_shards(
                    dataset,
                    shard_root,
                    root / "wrong-subset",
                    "no_memory",
                    2,
                )

    def test_merge_validates_domain_filtered_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root, domain="finance")
            shard_root = root / "shards"
            for index in range(2):
                _write_shard(
                    dataset,
                    shard_root,
                    index,
                    2,
                    domain_filter="finance",
                )

            manifest = merge_shards(
                dataset,
                shard_root,
                root / "merged",
                "no_memory",
                2,
                domain_filter="finance",
            )
            self.assertEqual(manifest["dataset"]["domain_filter"], "finance")
            with self.assertRaisesRegex(
                DatasetContractError, "domain filter mismatch"
            ):
                merge_shards(
                    dataset,
                    shard_root,
                    root / "wrong-domain",
                    "no_memory",
                    2,
                )


if __name__ == "__main__":
    unittest.main()
