"""Embedding providers.

- huggingface: local sentence-transformers model (default, semantic, no API key).
- hash: deterministic bag-of-words hashing (no download; used by tests / CI).
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from research_agent.config import Settings
from research_agent.text_utils import STOPWORDS, tokens


class HashingEmbeddings(Embeddings):
    """Lexical embedding: hashed unigram+bigram counts, L2-normalised."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = [t for t in tokens(text) if t not in STOPWORDS]
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if "_" not in g else 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "hash":
        return HashingEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
