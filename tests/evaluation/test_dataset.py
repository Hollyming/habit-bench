from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.core.dataset import load_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def session(user_id: str, index: int) -> dict:
    return {
        "session_id": f"{user_id}_s{index:04d}",
        "user_id": user_id,
        "session_index": index,
        "timestamp": f"2026-01-{index + 1:02d}T00:00:00",
        "domain": "test",
        "messages": [
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"response {index}"},
        ],
    }


class DatasetTest(unittest.TestCase):
    def test_nested_lifeline_uses_private_scope_without_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = [session("u1", 0), session("u1", 1)]
            for row in sessions:
                row.pop("user_id")
                row.pop("domain")
            write_jsonl(
                root / "public/lifelines.jsonl",
                [{"user_id": "u1", "domain": "test", "session_count": 2, "sessions": sessions}],
            )
            write_jsonl(
                root / "public/probes.jsonl",
                [
                    {
                        "probe_id": "p1",
                        "user_id": "u1",
                        "domain": "test",
                        "query": "Which response fits this user?",
                        "choices": [
                            {"choice_id": "A", "text": "first"},
                            {"choice_id": "B", "text": "second"},
                        ],
                    }
                ],
            )
            write_jsonl(
                root / "private/probe_key.jsonl",
                [
                    {
                        "probe_id": "p1",
                        "gold_choice_id": "B",
                        "gold_action_text": "second",
                        "probe_type": "direct",
                        "visible_history_scope": {"through_session_index": 1},
                    }
                ],
            )

            bundle = load_dataset(root)
            payload = bundle.method_payload("method")
            serialized = json.dumps(payload)
            self.assertEqual(payload["probes"][0]["visible_history_scope"]["max_session_index"], 1)
            self.assertNotIn("gold_choice_id", serialized)
            self.assertNotIn("gold_action_text", serialized)
            self.assertEqual(len(payload["sessions_by_user"]["u1"]), 2)

    def test_flat_lifeline_maps_private_probe_id_to_public_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "public/lifelines.jsonl", [session("u1", 0)])
            write_jsonl(
                root / "public/probes.jsonl",
                [
                    {
                        "probe_id": "public-p1",
                        "user_id": "u1",
                        "query": "Which response fits this user?",
                        "choices": [
                            {"choice_id": "A", "text": "first"},
                            {"choice_id": "B", "text": "second"},
                        ],
                        "visible_history_scope": {"user_id": "u1", "max_session_index": 0},
                    }
                ],
            )
            write_jsonl(
                root / "private/probe_key.jsonl",
                [
                    {
                        "probe_id": "private-p1",
                        "public_probe_id": "public-p1",
                        "gold_choice_id": "A",
                        "probe_type": "direct_use",
                    }
                ],
            )

            bundle = load_dataset(root)
            self.assertIn("public-p1", bundle.keys)
            self.assertEqual(bundle.keys["public-p1"]["private_probe_id"], "private-p1")

    def test_user_shards_are_disjoint_and_cover_all_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            users = [f"u{index}" for index in range(5)]
            write_jsonl(
                root / "public/lifelines.jsonl",
                [session(user_id, 0) for user_id in users],
            )
            write_jsonl(
                root / "public/probes.jsonl",
                [
                    {
                        "probe_id": f"p-{user_id}",
                        "user_id": user_id,
                        "query": "Which response fits this user?",
                        "choices": [
                            {"choice_id": "A", "text": "first"},
                            {"choice_id": "B", "text": "second"},
                        ],
                        "visible_history_scope": {
                            "user_id": user_id,
                            "max_session_index": 0,
                        },
                    }
                    for user_id in users
                ],
            )
            write_jsonl(
                root / "private/probe_key.jsonl",
                [
                    {"probe_id": f"p-{user_id}", "gold_choice_id": "A"}
                    for user_id in users
                ],
            )

            shards = [
                load_dataset(root, user_shard_index=index, user_shard_count=2)
                for index in range(2)
            ]
            shard_users = [set(bundle.sessions_by_user) for bundle in shards]
            self.assertFalse(shard_users[0].intersection(shard_users[1]))
            self.assertEqual(shard_users[0].union(shard_users[1]), set(users))
            self.assertEqual(shards[0].manifest["subset"]["user_shard_count"], 2)

    def test_user_shard_arguments_must_be_paired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "public/lifelines.jsonl", [session("u1", 0)])
            write_jsonl(
                root / "public/probes.jsonl",
                [
                    {
                        "probe_id": "p1",
                        "user_id": "u1",
                        "query": "q",
                        "choices": [
                            {"choice_id": "A", "text": "a"},
                            {"choice_id": "B", "text": "b"},
                        ],
                    }
                ],
            )
            write_jsonl(
                root / "private/probe_key.jsonl",
                [{"probe_id": "p1", "gold_choice_id": "A"}],
            )

            with self.assertRaisesRegex(ValueError, "must be provided together"):
                load_dataset(root, user_shard_index=0)


if __name__ == "__main__":
    unittest.main()
