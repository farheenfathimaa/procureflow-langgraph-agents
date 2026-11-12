"""Hybrid BM25 + pgvector retrieval with reciprocal rank fusion and re-ranking."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

from app.config import Settings
from rag.bm25 import BM25Index
from rag.embeddings import Embedder
from rag.reranker import NoOpReranker, Reranker
from rag.types import Document, RetrievalHit, hit_to_citation
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: Iterable[list[RetrievalHit]], rrf_k: int = 60
) -> list[RetrievalHit]:
    """Merge ranked lists with RRF: sum(1 / (k + rank)) per document."""

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    channels: dict[str, set[str]] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            doc_id = hit.document.doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            documents[doc_id] = hit.document
            channels.setdefault(doc_id, set()).update(hit.channels or ("retrieval",))

    fused = [
        RetrievalHit(
            document=documents[doc_id],
            score=score,
            channels=tuple(sorted(channels[doc_id])),
        )
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


class HybridRetriever:
    """Two-channel retrieval with RRF and an optional local cross-encoder."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker or (NoOpReranker() if not settings.reranker_enabled else None)
        if self.reranker is None:
            from rag.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(settings.reranker_model)
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()

    def refresh_index(self) -> None:
        """Rebuild the in-process BM25 side from the vector store's documents."""

        documents = self.store.all_documents()
        with self._lock:
            self._bm25 = BM25Index(documents) if documents else None

    def _ensure_index(self) -> BM25Index | None:
        if self._bm25 is None:
            self.refresh_index()
        return self._bm25

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Return re-ranked hybrid results for a query."""

        limit = top_k or self.settings.rag_top_k
        candidates = candidate_k or self.settings.rag_candidate_k
        query = query.strip()
        if not query:
            return []

        vector_hits = self.store.search(query, k=candidates, doc_type=doc_type)
        bm25 = self._ensure_index()
        sparse_hits = bm25.search(query, k=candidates, doc_type=doc_type) if bm25 else []
        fused = reciprocal_rank_fusion([vector_hits, sparse_hits], rrf_k=self.settings.rrf_k)

        if not fused:
            return []
        try:
            return self.reranker.rerank(query, fused, top_k=limit)
        except Exception:
            logger.exception("Re-ranker failed; falling back to RRF order")
            return fused[:limit]

    def citations(self, hits: list[RetrievalHit]):
        """Convert retrieval hits to API citations."""

        return [hit_to_citation(hit) for hit in hits]
