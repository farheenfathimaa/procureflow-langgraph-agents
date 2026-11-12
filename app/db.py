"""PostgreSQL connection pool and RAG schema management."""

from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class Database:
    """Thin psycopg pool wrapper shared by RAG and the LangGraph checkpointer."""

    def __init__(self, database_url: str, embedding_dim: int = 384) -> None:
        self.database_url = database_url
        self.embedding_dim = embedding_dim
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            open=False,
            timeout=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

    def connect(self) -> None:
        """Open the pool and verify connectivity early."""

        self.pool.open(wait=True, timeout=30)

    def initialize(self) -> None:
        """Create the pgvector extension and RAG schema if missing."""

        self.connect()
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_doc_type_idx ON rag_documents (doc_type)"
            )
            # HNSW is available in modern pgvector. Fall back gracefully if the
            # extension server does not support it (the table still works).
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS rag_documents_embedding_idx "
                    "ON rag_documents USING hnsw (embedding vector_cosine_ops)"
                )
            except Exception as exc:  # pragma: no cover - depends on pgvector build
                logger.warning("Could not create HNSW index: %s", exc)

    def ping(self) -> bool:
        """Return whether a trivial query succeeds."""

        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close all pooled connections."""

        try:
            self.pool.close()
        except Exception:  # pragma: no cover - defensive shutdown path
            logger.exception("Error closing database pool")


def build_checkpointer(database: Database) -> Any:
    """Build and initialise the Postgres LangGraph checkpointer."""

    from langgraph.checkpoint.postgres import PostgresSaver

    checkpointer = PostgresSaver(database.pool)
    checkpointer.setup()
    return checkpointer
