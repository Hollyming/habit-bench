"""Minimal llmlingua shim for SeCom retrieval-only adapter runs."""


class PromptCompressor:
    def __init__(self, *args, **kwargs):
        pass

    def compress_prompt(self, prompt, *args, **kwargs):
        return {"compressed_prompt_list": prompt}
