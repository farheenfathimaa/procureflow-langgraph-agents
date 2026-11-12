"""Sparse BM25 retrieval over the same local corpus."""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from rag.types import Document, RetrievalHit

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer tuned for short procurement documents."""

    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


class BM25Index:
    """In-process BM25 index. The 20-document corpus is tiny by design."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = {document.doc_id: document for document in documents}
        self._ordered = list(self.documents.values())
        self._index = BM25Okapi([tokenize(document.content) for document in self._ordered])
        self._lock = threading.Lock()

    def search(self, query: str, k: int, doc_type: str | None = None) -> list[RetrievalHit]:
        with self._lock:
            scores = self._index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
        hits: list[RetrievalHit] = []
        for index, score in ranked:
            document = self._ordered[index]
            if doc_type and document.doc_type != doc_type:
                continue
            if float(score) <= 0:
                continue
            hits.append(RetrievalHit(document=document, score=float(score), channels=("bm25",)))
            if len(hits) >= k:
                break
        return hits
