"""Vector stores: pgvector in production and an in-memory store for tests."""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Protocol

import numpy as np
from psycopg.types.json import Jsonb

from rag.embeddings import Embedder
from rag.types import Document, RetrievalHit

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Small storage interface used by ingestion and retrieval."""

    def upsert_documents(self, documents: list[Document]) -> int: ...

    def all_documents(self) -> list[Document]: ...

    def search(self, query: str, k: int, doc_type: str | None = None) -> list[RetrievalHit]: ...

    def count(self) -> int: ...


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


class InMemoryVectorStore:
    """Cosine-similarity vector store with no external services.

    Used by unit tests and by ``CHECKPOINTER_BACKEND=memory`` local mode.
    """

    def __init__(self, embedder: Embedder, documents: list[Document] | None = None) -> None:
        self.embedder = embedder
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        if documents:
            self.upsert_documents(documents)

    def upsert_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed_documents([doc.content for doc in documents])
        with self._lock:
            for document, vector in zip(documents, vectors, strict=True):
                self._documents[document.doc_id] = document
                self._vectors[document.doc_id] = np.asarray(vector, dtype=np.float32)
        return len(documents)

    def all_documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def search(self, query: str, k: int, doc_type: str | None = None) -> list[RetrievalHit]:
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        with self._lock:
            candidates = [
                (doc_id, document, vector)
                for doc_id, document in self._documents.items()
                if (doc_type is None or document.doc_type == doc_type)
                for vector in [self._vectors[doc_id]]
            ]
        scored: list[RetrievalHit] = []
        for _, document, vector in candidates:
            score = float(np.dot(query_vector, vector))
            scored.append(RetrievalHit(document=document, score=score, channels=("vector",)))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        with self._lock:
            return len(self._documents)

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._vectors.clear()


class PgVectorStore:
    """PostgreSQL + pgvector store with cosine distance search."""

    def __init__(self, pool: Any, embedder: Embedder, table: str = "rag_documents") -> None:
        # Table names are code-controlled, never user input.
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        self.pool = pool
        self.embedder = embedder
        self.table = table

    def upsert_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed_documents([doc.content for doc in documents])
        rows = [
            (
                document.doc_id,
                document.title,
                document.content,
                document.doc_type,
                document.source,
                Jsonb(document.metadata),
                _vector_literal(vector),
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {self.table}
                    (doc_id, title, content, doc_type, source, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    doc_type = EXCLUDED.doc_type,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                rows,
            )
        return len(rows)

    def all_documents(self) -> list[Document]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT doc_id, title, content, doc_type, source, metadata FROM {self.table}"
            ).fetchall()
        return [
            Document(
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"],
                doc_type=row["doc_type"],
                source=row["source"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    def search(self, query: str, k: int, doc_type: str | None = None) -> list[RetrievalHit]:
        vector = _vector_literal(self.embedder.embed_query(query))
        where = "WHERE doc_type = %s" if doc_type else ""
        params: list[Any] = [vector]
        if doc_type:
            params.append(doc_type)
        params.extend([vector, k])
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT doc_id, title, content, doc_type, source, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table}
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [
            RetrievalHit(
                document=Document(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    content=row["content"],
                    doc_type=row["doc_type"],
                    source=row["source"],
                    metadata=row["metadata"] or {},
                ),
                score=float(row["score"]),
                channels=("vector",),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {self.table}").fetchone()
        return int(row["count"])

    def clear(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(f"DELETE FROM {self.table}")


def document_fingerprint(documents: list[Document]) -> str:
    """Stable fingerprint used to decide whether an index refresh is needed."""

    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.doc_id):
        digest.update(document.doc_id.encode())
        digest.update(document.content.encode())
    return digest.hexdigest()


__all__ = ["InMemoryVectorStore", "PgVectorStore", "VectorStore", "document_fingerprint"]
