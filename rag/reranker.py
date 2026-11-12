"""Free local cross-encoder re-ranking."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from rag.types import RetrievalHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Re-order fused candidates using a query-document scoring model."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]: ...


class NoOpReranker:
    """Deterministic fallback that preserves reciprocal-rank-fusion order."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        return hits[:top_k]


class CrossEncoderReranker:
    """A local sentence-transformers cross-encoder.

    The model is loaded lazily to keep imports and CI fast.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:  # pragma: no cover - dependency error path
                    raise RuntimeError(
                        "sentence-transformers is required for cross-encoder re-ranking"
                    ) from exc
                self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        model = self._load()
        pairs = [(query, hit.document.content) for hit in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        reranked = [
            RetrievalHit(
                document=hit.document,
                score=float(score),
                channels=(*hit.channels, "reranker"),
            )
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked[:top_k]
