"""Small tiktoken-compatible shim for SeCom adapter token accounting.

The SeCom retrieval adapter only needs `encoding_for_model(...).encode(text)`
to estimate retrieved-token cost. This lightweight fallback avoids requiring
Rust-built tiktoken in the isolated evaluation environment.
"""

import re


class _SimpleEncoding:
    def encode(self, text):
        return re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)


def encoding_for_model(model_name):
    return _SimpleEncoding()
