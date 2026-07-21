from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import write_json, write_jsonl
from eval.merge_shards import merge_shards


def _session(user_id: str) -> dict:
    return {
        "session_id": f"{user_id}-s0",
        "user_id": user_id,
        "session_index": 0,
        "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "response"},
        ],
    }


def _write_fixture(root: Path) -> Path:
    dataset = root / "dataset"
    users = ["u0", "u1"]
    write_jsonl(dataset / "public/lifelines.jsonl", [_session(user) for user in users])
    write_jsonl(
        dataset / "public/probes.jsonl",
        [
            {
                "probe_id": f"p-{user}",
                "user_id": user,
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


def _write_shard(dataset: Path, shard_root: Path, index: int, count: int) -> None:
    bundle = load_dataset(dataset, user_shard_index=index, user_shard_count=count)
    shard_dir = shard_root / f"shard_{index:03d}_of_{count:03d}"
    implementation = {"kind": "control", "source": "test", "revision": "1"}
    base_model = {"model": "test-model"}
    write_json(
        shard_dir / "run_manifest.json",
        {
            "method_name": "no_memory",
            "implementation": implementation,
            "base_model": base_model,
            "dataset": bundle.manifest,
            "adapter_runtime": {"elapsed_sec": 1},
            "answer_runtime": {"elapsed_sec": 1},
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

    def test_merge_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_fixture(root)
            shard_root = root / "shards"
            _write_shard(dataset, shard_root, 0, 2)

            with self.assertRaisesRegex(DatasetContractError, "Shard coverage mismatch"):
                merge_shards(dataset, shard_root, root / "merged", "no_memory", 2)


if __name__ == "__main__":
    unittest.main()
