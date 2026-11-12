"""Local, free embeddings with a deterministic test fallback."""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol

import numpy as np

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Minimal interface shared by real and test embedders."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Free local embeddings using sentence-transformers.

    The model is loaded lazily so importing the app (and running the unit
    tests) does not download model weights.
    """

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - dependency error path
                    raise RuntimeError(
                        "sentence-transformers is not installed. Install with "
                        "`pip install -e '.[embeddings]'` or set EMBEDDING_BACKEND=hash."
                    ) from exc
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder:
    """Deterministic feature hashing used only in tests and offline demos.

    It is deliberately dependency-free and not a substitute for semantic
    embeddings in production.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _tokens(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings: Settings) -> Embedder:
    """Build the configured embedder, preferring the free local model."""

    if settings.embedding_backend == "hash":
        return HashEmbedder(settings.embedding_dim)
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
