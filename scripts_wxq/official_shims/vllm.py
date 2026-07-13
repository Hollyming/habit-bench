"""Minimal vllm shim for SeCom retrieval-only adapter runs."""


class LLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("vLLM is unavailable; the SeCom adapter does not use LocalLLM.")


class SamplingParams:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("vLLM is unavailable; the SeCom adapter does not use LocalLLM.")
