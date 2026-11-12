"""Internal retrieval data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas import Citation


@dataclass(frozen=True)
class Document:
    """A source document stored in BM25 and pgvector."""

    doc_id: str
    title: str
    content: str
    doc_type: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    """A document returned by one or more retrieval channels."""

    document: Document
    score: float
    channels: tuple[str, ...] = ()


def hit_to_citation(hit: RetrievalHit, snippet_length: int = 280) -> Citation:
    """Convert a retrieval hit to a compact, traceable API citation."""

    content = hit.document.content.strip().replace("\n", " ")
    snippet = content if len(content) <= snippet_length else f"{content[:snippet_length].rstrip()}…"
    return Citation(
        doc_id=hit.document.doc_id,
        title=hit.document.title,
        source=hit.document.source,
        doc_type=hit.document.doc_type,
        snippet=snippet,
        score=round(float(hit.score), 6),
    )
