from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.core.dataset import DatasetBundle, load_dataset
from eval.core.io import write_json, write_jsonl
from eval.supplementary.analyze import build_supplementary_analysis
from eval.supplementary.compare import compare_runs
from eval.supplementary.human_audit import cohen_kappa
from eval.supplementary.merge_oracle import merge_oracle_shards
from eval.supplementary.oracle_controls import (
    build_oracle_contexts,
    build_oracle_habit_state_context,
    score_oracle_predictions,
)


def _session(user_id: str, index: int) -> dict:
    return {
        "session_id": f"{user_id}-s{index}",
        "user_id": user_id,
        "session_index": index,
        "timestamp": f"2026-01-{index + 1:02d}T00:00:00Z",
        "domain": "finance",
        "messages": [
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"response {index}"},
        ],
    }


def _signature(primary: str, secondary: str) -> dict:
    return {
        "variants": {"h1": primary, "h2": secondary},
        "order": ["h1", "h2"],
    }


def fixture_bundle() -> DatasetBundle:
    sessions = {
        user_id: [_session(user_id, index) for index in range(6)]
        for user_id in ("u1", "u2")
    }
    probes = []
    keys = {}
    for user_id in sessions:
        for suffix, probe_type in (("e", "explicit_retrieval"), ("h", "dual_asof_reversal")):
            probe_id = f"{user_id}-{suffix}"
            probes.append(
                {
                    "probe_id": probe_id,
                    "user_id": user_id,
                    "domain": "finance",
                    "query": "Which response is supported?",
                    "choices": [
                        {"choice_id": "A", "text": "active policy"},
                        {"choice_id": "B", "text": "decoy policy"},
                    ],
                    "visible_history_scope": {
                        "user_id": user_id,
                        "max_session_index": 5,
                    },
                }
            )
            keys[probe_id] = {
                "probe_id": probe_id,
                "gold_choice_id": "A",
                "probe_type": probe_type,
                "target_habit_ids": ["h1", "h2"],
                "gold_action_text": "Use both active policy variants.",
                "choice_policy_signatures": {
                    "A": _signature("active-1", "active-2"),
                    "B": _signature("decoy-1", "active-2"),
                },
                "surface_decoy_variants": {"h1": "decoy-1", "h2": "decoy-2"},
                "decision_evidence_session_ids": [
                    f"{user_id}-s1",
                    f"{user_id}-s4",
                ],
                "temporal_context_session_ids": [f"{user_id}-s0"],
                "nonbinding_evidence_session_ids": [f"{user_id}-s3"],
                "evidence_bands": ["early", "late"],
            }
    return DatasetBundle(
        Path("."),
        sessions,
        probes,
        keys,
        {"name": "supplementary-fixture"},
    )


def scored_rows(bundle: DatasetBundle, choice_id: str) -> list[dict]:
    rows = []
    for probe in bundle.probes:
        correct = choice_id == bundle.keys[probe["probe_id"]]["gold_choice_id"]
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "method_name": "fixture",
                "choice_id": choice_id,
                "correct": correct,
                "answer": {
                    "latency_sec": 0.5,
                    "memory_tokens_used": 100,
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 5,
                        "total_tokens": 125,
                    },
                },
                "memory_cost": {"retrieval_elapsed_sec": 0.1},
                "retrieval": {
                    "evaluable": True,
                    "evidence_hit_at_5": True,
                    "evidence_recall_at_5": 0.5,
                    "full_evidence_at_5": False,
                    "complete_chain_at_5": False,
                    "component_complete_coverage_at_5": 0.5,
                    "joint_answer_evidence_hit_at_5": correct,
                    "clean_grounded_answer_at_5": False,
                    "nonbinding_intrusion_rate_at_5": 0.2,
                },
            }
        )
    return rows


class SupplementaryTest(unittest.TestCase):
    def test_oracle_evidence_excludes_nonbinding_and_preserves_ranking(self) -> None:
        bundle = fixture_bundle()
        rows = build_oracle_contexts(bundle, "oracle_evidence")
        first = rows[0]
        user_id = bundle.probes[0]["user_id"]
        self.assertEqual(
            first["evidence_session_ids"],
            [f"{user_id}-s1", f"{user_id}-s4", f"{user_id}-s0"],
        )
        self.assertNotIn(f"SESSION_ID={user_id}-s3", first["memory_context"])

    def test_oracle_habit_state_uses_variants_without_choice_label(self) -> None:
        bundle = fixture_bundle()
        probe = bundle.probes[0]
        row = build_oracle_habit_state_context(
            probe, bundle.keys[probe["probe_id"]]
        )
        self.assertIn("active-1", row["memory_context"])
        self.assertIn("active-2", row["memory_context"])
        self.assertNotIn("choice_id", row["memory_context"])
        self.assertEqual(row["evidence_session_ids"], [])

    def test_food_oracle_uses_graph_action_and_condition_without_changing_travel_contract(
        self,
    ) -> None:
        food_probe = {
            "probe_id": "food-p",
            "probe_type": "direct_use",
        }
        food_key = {
            "habit_family": "content_constraints",
            "gold_action": "apply_split_decision_stable_habit",
            "hidden_habit_graph": {
                "family": "content_constraints",
                "condition": "salad or bowl with raw vegetables",
                "default_action": "cut vegetables into diagonal slivers",
            },
        }
        food_row = build_oracle_habit_state_context(food_probe, food_key)
        self.assertIn(
            "cut vegetables into diagonal slivers", food_row["memory_context"]
        )
        self.assertNotIn(
            "apply_split_decision_stable_habit", food_row["memory_context"]
        )
        self.assertIn(
            "salad or bowl with raw vegetables", food_row["memory_context"]
        )

        travel_probe = {
            "probe_id": "travel-p",
            "probe_type": "cross_context_transfer",
        }
        travel_key = {
            "habit_family": "trip_context_flight_policy",
            "gold_action": "specific-choice-opcode",
            "gold_action_text": "Business via New York for $1,540 total.",
            "hidden_habit_graph": {
                "family": "trip_context_flight_policy",
                "condition": "when comparing flight cabin options",
                "default_action": "prefer business-class options first",
            },
        }
        travel_row = build_oracle_habit_state_context(travel_probe, travel_key)
        travel_state = json.loads(travel_row["memory_context"].split("\n\n", 1)[1])
        self.assertEqual(
            travel_state["current_state"]["required_action"],
            "Business via New York for $1,540 total.",
        )
        self.assertIsNone(travel_state["default_policy"]["condition"])

    def test_oracle_habit_state_does_not_manufacture_retrieval_scores(self) -> None:
        bundle = fixture_bundle()
        predictions = [
            {
                "probe_id": probe["probe_id"],
                "choice_id": "A",
                "evidence_session_ids": [],
            }
            for probe in bundle.probes
        ]
        detailed, metrics, _ = score_oracle_predictions(
            predictions, bundle, "oracle_habit_state"
        )
        self.assertEqual(metrics["method_name"], "oracle_habit_state")
        self.assertEqual(metrics["overall"]["retrieval_mode"], "none")
        self.assertNotIn("evidence_recall_at_5", detailed[0]["retrieval"])

    def test_oracle_merger_preserves_no_retrieval_semantics(self) -> None:
        source = fixture_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset"
            write_jsonl(
                dataset_dir / "public/lifelines.jsonl",
                [
                    session
                    for sessions in source.sessions_by_user.values()
                    for session in sessions
                ],
            )
            write_jsonl(dataset_dir / "public/probes.jsonl", source.probes)
            write_jsonl(
                dataset_dir / "private/probe_key.jsonl",
                list(source.keys.values()),
            )
            shard_root = root / "shards"
            for shard_index in range(2):
                bundle = load_dataset(
                    dataset_dir,
                    user_shard_index=shard_index,
                    user_shard_count=2,
                )
                shard_dir = shard_root / f"shard_{shard_index:03d}_of_002"
                contexts = build_oracle_contexts(
                    bundle, "oracle_habit_state"
                )
                predictions = [
                    {
                        "probe_id": probe["probe_id"],
                        "choice_id": "A",
                        "evidence_session_ids": [],
                    }
                    for probe in bundle.probes
                ]
                write_jsonl(shard_dir / "memory_contexts.jsonl", contexts)
                write_jsonl(shard_dir / "predictions.jsonl", predictions)
                write_json(
                    shard_dir / "supplementary_manifest.json",
                    {
                        "method_name": "oracle_habit_state",
                        "dataset": bundle.manifest,
                        "base_model": {
                            "model": "fixture",
                            "base_url": f"http://worker-{shard_index}",
                        },
                        "execution": {
                            "status": "succeeded",
                            "wall_clock_sec": 1.0,
                        },
                    },
                )
            merged = merge_oracle_shards(
                dataset_dir=dataset_dir,
                shard_root=shard_root,
                output_dir=root / "merged",
                mode="oracle_habit_state",
                expected_shards=2,
            )
            self.assertEqual(merged["result"]["retrieval_mode"], "none")
            self.assertEqual(merged["result"]["accuracy"], 1.0)

    def test_analysis_reports_user_gap_components_and_missing_contracts(self) -> None:
        bundle = fixture_bundle()
        rows = scored_rows(bundle, "A")
        payload, slices, diagnostics, users = build_supplementary_analysis(
            rows, bundle, bootstrap_samples=100, seed=7
        )
        self.assertEqual(payload["accuracy"]["micro_accuracy"], 1.0)
        self.assertEqual(
            payload["explicit_to_habit_transfer"]["explicit_minus_latent_gap"],
            0.0,
        )
        self.assertEqual(
            payload["policy_components"]["policy_component_accuracy"], 1.0
        )
        self.assertEqual(payload["calibration"]["status"], "unavailable")
        self.assertEqual(payload["false_personalization"]["status"], "unavailable")
        self.assertEqual(
            payload["answer_retrieval_error_analysis"][
                "answer_x_evidence_hit"
            ]["correct_with_evidence_hit"]["rate"],
            1.0,
        )
        self.assertTrue(slices)
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(len(users), 2)

    def test_compare_is_paired_and_holm_adjusted(self) -> None:
        bundle = fixture_bundle()
        payload, methods, pairs = compare_runs(
            {
                "better": scored_rows(bundle, "A"),
                "worse": scored_rows(bundle, "B"),
            },
            bundle,
            bootstrap_samples=100,
            seed=3,
            allow_partial=False,
        )
        self.assertEqual(payload["coverage"]["probes"], 4)
        self.assertEqual(len(methods), 2)
        self.assertEqual(pairs[0]["micro_accuracy_difference"], 1.0)
        self.assertIn("two_sided_exact_p_holm", pairs[0])

    def test_human_audit_kappa(self) -> None:
        result = cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"])
        self.assertEqual(result["raw_agreement"], 1.0)
        self.assertEqual(result["cohen_kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
