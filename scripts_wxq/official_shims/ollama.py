"""Minimal ollama shim for no-LLM official adapter smoke runs."""


def chat(*args, **kwargs):
    raise RuntimeError(
        "ollama.chat was called, but this HABIT-Bench official adapter run is "
        "configured for no-LLM retrieval only."
    )


class Client:
    def __init__(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        raise RuntimeError(
            "ollama.Client.chat was called, but this HABIT-Bench official "
            "adapter run is configured for no-LLM retrieval only."
        )
