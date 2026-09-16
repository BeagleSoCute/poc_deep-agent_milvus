"""Embedding providers.

- ``openai``: any OpenAI-compatible /embeddings endpoint (e.g. the LiteLLM proxy).
- ``hash``: local, deterministic, dependency-free. Good enough to prove that data
  lands in Milvus and for offline tests; semantic quality is only lexical.
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import settings


class HashEmbeddings(Embeddings):
    """Bag of word + character-trigram features hashed into a fixed-size vector."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"\w+", text)
        compact = re.sub(r"\s+", " ", text)
        grams = [compact[i : i + 3] for i in range(max(len(compact) - 2, 0))]
        return words + words + grams  # words weighted x2

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text) or ["<empty>"]:
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:  # all features cancelled out; COSINE needs a non-zero vector
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_embeddings(provider: str | None = None) -> Embeddings:
    provider = (provider or settings.embedding_provider).lower()
    if provider == "hash":
        return HashEmbeddings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.openai_base_url or None,
            api_key=settings.openai_api_key or None,
            # Proxies (LiteLLM) expect raw strings, not tiktoken token ids.
            check_embedding_ctx_length=False,
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (use 'openai' or 'hash')")
