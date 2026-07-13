"""Minimal litellm shim for no-LLM official adapter smoke runs.

The A-MEM adapter disables memory evolution and should not call litellm.
If an unexpected path calls completion, fail clearly instead of silently
returning fabricated LLM output.
"""


def completion(*args, **kwargs):
    raise RuntimeError(
        "litellm.completion was called, but this HABIT-Bench official adapter "
        "run is configured for no-LLM retrieval only."
    )
