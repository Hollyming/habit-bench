"""Dependency-free opt-in policy shared by schema, client and tool executor.

Do not import the agent package here: its eager imports depend on the schema
generator, including when LocalClient is the first MIRIX entry point.
"""
import os


def uses_native_memory_text_fields() -> bool:
    return (os.environ.get("HABITBENCH_INFERENCE_BACKEND") == "external-openai-compatible"
            and os.environ.get("MIRIX_API_NATIVE_TEXT_FIELDS") == "1")
