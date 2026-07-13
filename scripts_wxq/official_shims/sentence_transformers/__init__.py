"""Deterministic local SentenceTransformer shim for offline adapter runs.

This fallback exposes the small subset of the sentence-transformers API used by
the HABIT-Bench official adapters. It avoids network downloads in environments
where HuggingFace access is blocked. Embeddings are deterministic hashed
bag-of-token vectors, suitable for smoke/adapter evaluation but not a full
reproduction of HuggingFace embedding quality.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List

import numpy as np


class SentenceTransformer:
    def __init__(self, model_name_or_path: str = "offline-hash-embedding", device: str | None = None, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.device = device or "cpu"
        self.dim = int(kwargs.get("embedding_dims") or kwargs.get("truncate_dim") or 384)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def to(self, device: str):
        self.device = device
        return self

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode(
        self,
        sentences,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        is_single = isinstance(sentences, str)
        items: List[str] = [sentences] if is_single else list(sentences)
        arr = np.vstack([self._embed_one(str(item)) for item in items]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        if is_single:
            return arr[0] if convert_to_numpy else arr[0].tolist()
        return arr if convert_to_numpy else arr.tolist()
