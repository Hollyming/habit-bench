from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from eval.context_windows import WINDOW_TIERS, resolve_context_window


class ContextWindowTierTest(unittest.TestCase):
    def test_auto_selects_40k_for_current_qwen_capacity(self):
        resolved = resolve_context_window("auto", 40_960)
        self.assertEqual(resolved.resolved_tier, "40k")
        self.assertEqual(resolved.max_input_tokens, 40_000)
        self.assertEqual(resolved.history_token_budget, 38_000)

    def test_auto_selects_32k_for_32768_model(self):
        resolved = resolve_context_window("auto", 32_768)
        self.assertEqual(resolved.resolved_tier, "32k")
        self.assertEqual(resolved.history_token_budget, 30_000)

    def test_explicit_tier_cannot_exceed_model_capacity(self):
        with self.assertRaisesRegex(ValueError, "configured model capacity"):
            resolve_context_window("64k", 40_960)

    def test_custom_tier_and_history_override_are_recorded(self):
        resolved = resolve_context_window(
            "custom",
            96_000,
            custom_max_input_tokens=90_000,
            max_history_tokens=84_000,
        )
        self.assertEqual(resolved.max_input_tokens, 90_000)
        self.assertEqual(resolved.history_token_budget, 84_000)
        self.assertEqual(resolved.reserved_prompt_tokens, 6_000)
        self.assertEqual(resolved.budget_source, "explicit_history_override")

    def test_documented_yaml_tiers_match_runtime_resolver(self):
        project_root = Path(__file__).resolve().parents[2]
        config = yaml.safe_load(
            (project_root / "configs/methods/full_memory.yaml").read_text(
                encoding="utf-8"
            )
        )
        documented = config["history"]["tiers"]
        for name, tier in WINDOW_TIERS.items():
            self.assertEqual(
                documented[name],
                {
                    "max_input_tokens": tier.max_input_tokens,
                    "reserved_prompt_tokens": tier.reserved_prompt_tokens,
                    "history_token_budget": tier.history_token_budget,
                },
            )


if __name__ == "__main__":
    unittest.main()
