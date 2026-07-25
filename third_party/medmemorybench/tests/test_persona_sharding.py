import unittest

from main import parse_persona_ids, parse_sample_ids
from src.config import ConfigLoader
from src.evaluator import create_evaluator


class PersonaShardingTests(unittest.TestCase):
    def test_ranges_and_deduplication(self):
        self.assertEqual(parse_persona_ids("1-3,3,7"), [1, 2, 3, 7])

    def test_reverse_range_rejected(self):
        with self.assertRaisesRegex(Exception, "Invalid persona range"):
            parse_persona_ids("4-2")

    def test_sample_ids_preserve_order_and_deduplicate(self):
        self.assertEqual(
            parse_sample_ids("conv-30,conv-26,conv-30"),
            ["conv-30", "conv-26"],
        )

    def test_effective_shard_is_recorded_in_manifest_config(self):
        evaluator = create_evaluator(
            method_config_name="mem0_qwen3-8b_smoke",
            dataset_name="medmemorybench_paper_efficient",
            config_loader=ConfigLoader(),
            dataset_overrides={"persona_ids": [3, 4]},
        )
        self.assertEqual(evaluator.dataset_config.persona_ids, [3, 4])
        self.assertEqual(
            evaluator.dataset_config.raw_config["evaluation"]["persona_ids"],
            [3, 4],
        )

    def test_locomo_sample_override_is_recorded(self):
        evaluator = create_evaluator(
            method_config_name="bm25_rag_qwen3-8b_smoke",
            dataset_name="locomo_local_efficient",
            config_loader=ConfigLoader(),
            dataset_overrides={"sample_ids": ["conv-26", "conv-30"]},
        )
        self.assertEqual(evaluator.dataset_config.sample_ids, ["conv-26", "conv-30"])
        self.assertEqual(
            evaluator.dataset_config.raw_config["evaluation"]["sample_ids"],
            ["conv-26", "conv-30"],
        )


if __name__ == "__main__":
    unittest.main()
