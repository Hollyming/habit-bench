#!/usr/bin/env python
"""Resolve reproducible long-context window tiers for HABIT-Bench controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WindowTier:
    name: str
    max_input_tokens: int
    reserved_prompt_tokens: int

    @property
    def history_token_budget(self) -> int:
        return self.max_input_tokens - self.reserved_prompt_tokens


@dataclass(frozen=True)
class ResolvedWindow:
    requested_tier: str
    resolved_tier: str
    model_context_tokens: int
    max_input_tokens: int
    reserved_prompt_tokens: int
    history_token_budget: int
    budget_source: str

    def public_dict(self) -> dict[str, str | int]:
        return asdict(self)


WINDOW_TIERS = {
    tier.name: tier
    for tier in (
        WindowTier("8k", 8_000, 2_000),
        WindowTier("16k", 16_000, 2_000),
        WindowTier("32k", 32_000, 2_000),
        WindowTier("40k", 40_000, 2_000),
        WindowTier("64k", 64_000, 4_000),
        WindowTier("128k", 128_000, 8_000),
    )
}
WINDOW_TIER_CHOICES = ("auto", *WINDOW_TIERS, "custom")


def resolve_context_window(
    requested_tier: str,
    model_context_tokens: int,
    *,
    custom_max_input_tokens: int | None = None,
    reserved_prompt_tokens: int | None = None,
    max_history_tokens: int | None = None,
) -> ResolvedWindow:
    if requested_tier not in WINDOW_TIER_CHOICES:
        raise ValueError(
            f"Unknown context-window tier {requested_tier!r}; "
            f"choose one of {WINDOW_TIER_CHOICES}"
        )
    if model_context_tokens < 1:
        raise ValueError("model_context_tokens must be positive")

    if requested_tier == "auto":
        compatible = [
            tier
            for tier in WINDOW_TIERS.values()
            if tier.max_input_tokens <= model_context_tokens
        ]
        if not compatible:
            raise ValueError(
                f"No standard window tier fits model capacity {model_context_tokens}; "
                "use tier=custom with an explicit max input"
            )
        tier = max(compatible, key=lambda item: item.max_input_tokens)
    elif requested_tier == "custom":
        if custom_max_input_tokens is None:
            raise ValueError(
                "tier=custom requires custom_max_input_tokens"
            )
        tier = WindowTier(
            "custom",
            custom_max_input_tokens,
            max(2_000, custom_max_input_tokens // 16),
        )
    else:
        tier = WINDOW_TIERS[requested_tier]

    if tier.max_input_tokens > model_context_tokens:
        raise ValueError(
            f"Tier {tier.name} needs {tier.max_input_tokens} input tokens, but "
            f"the configured model capacity is {model_context_tokens}"
        )
    if tier.max_input_tokens < 2:
        raise ValueError("max input tokens must be at least 2")

    reserve = (
        tier.reserved_prompt_tokens
        if reserved_prompt_tokens is None
        else reserved_prompt_tokens
    )
    if reserve < 8 or reserve >= tier.max_input_tokens:
        raise ValueError(
            "reserved_prompt_tokens must be at least 8 and smaller than max input"
        )

    if max_history_tokens is None:
        history_budget = tier.max_input_tokens - reserve
        budget_source = "tier_default"
    else:
        if max_history_tokens < 1:
            raise ValueError("max_history_tokens must be positive")
        if max_history_tokens > tier.max_input_tokens - 8:
            raise ValueError(
                "max_history_tokens must leave at least 8 input tokens outside history"
            )
        history_budget = max_history_tokens
        reserve = tier.max_input_tokens - history_budget
        budget_source = "explicit_history_override"

    return ResolvedWindow(
        requested_tier=requested_tier,
        resolved_tier=tier.name,
        model_context_tokens=model_context_tokens,
        max_input_tokens=tier.max_input_tokens,
        reserved_prompt_tokens=reserve,
        history_token_budget=history_budget,
        budget_source=budget_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=WINDOW_TIER_CHOICES, default="auto"
    )
    parser.add_argument("--model-context-tokens", type=int, required=True)
    parser.add_argument("--custom-max-input-tokens", type=int)
    parser.add_argument("--reserved-prompt-tokens", type=int)
    parser.add_argument("--max-history-tokens", type=int)
    parser.add_argument(
        "--field",
        choices=(
            "resolved_tier",
            "max_input_tokens",
            "reserved_prompt_tokens",
            "history_token_budget",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_context_window(
        args.tier,
        args.model_context_tokens,
        custom_max_input_tokens=args.custom_max_input_tokens,
        reserved_prompt_tokens=args.reserved_prompt_tokens,
        max_history_tokens=args.max_history_tokens,
    )
    if args.field:
        print(getattr(resolved, args.field))
    else:
        print(json.dumps(resolved.public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
