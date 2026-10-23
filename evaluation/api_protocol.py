"""Provider request policy; tokenizer template flags are not API capabilities."""

import os
from typing import Any


def has_forced_thinking(model: str) -> bool:
    return model.lower().rsplit("/", 1)[-1] in {"glm-5.3", "glm-5.3-flash"}


def thinking_body(model: str, enable_thinking: bool = False) -> dict[str, Any]:
    # GLM-5.3-FLASH always opens a thinking block, but the API still accepts
    # an explicit reasoning effort. Omitting it silently selects ``max``;
    # that can spend the entire completion budget on hidden reasoning before
    # emitting the required choice object. Keep thinking enabled while using
    # the benchmark's bounded effort profile (low by default).
    if has_forced_thinking(model):
        effort = os.getenv("HABITBENCH_API_REASONING_EFFORT", "low").strip().lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError("HABITBENCH_API_REASONING_EFFORT must be low, high, or max")
        return {"thinking": {"type": "enabled"}, "reasoning_effort": effort}
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def choice_response_format(valid_choice_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice_id": {"type": "string", "enum": valid_choice_ids}
                },
                "required": ["choice_id"],
                "additionalProperties": False,
            },
        },
    }
