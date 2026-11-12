"""Application runtime construction and dependency wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.db import Database, build_checkpointer
from app.llm import LLMClient
from app.memory import MemoryStore
from app.workflow import WorkflowService
from connectors.google_oauth import GoogleOAuthManager
from connectors.google_sheets import (
    BudgetProvider,
    GoogleSheetsConnector,
    build_budget_provider,
)
from rag.dataset import load_documents
from rag.embeddings import build_embedder
from rag.hybrid import HybridRetriever
from rag.ingest import ingest_into_store
from rag.reranker import CrossEncoderReranker, NoOpReranker
from rag.vector_store import InMemoryVectorStore, PgVectorStore, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """Long-lived dependencies shared by FastAPI requests."""

    settings: Settings
    memory: MemoryStore
    embedder: Any
    store: VectorStore
    retriever: HybridRetriever
    oauth: GoogleOAuthManager
    sheets_connector: GoogleSheetsConnector | None
    budget_provider: BudgetProvider
    llm: LLMClient
    workflow: WorkflowService
    db: Database | None = None

    @property
    def document_count(self) -> int:
        try:
            return self.store.count()
        except Exception:
            return 0

    def close(self) -> None:
        self.memory.close()
        if self.db:
            self.db.close()


def build_runtime(settings: Settings) -> AppRuntime:
    """Build storage, retrieval, connectors and the compiled LangGraph.

    Postgres is used when configured. A completely in-memory mode is also
    supported for tests and local experiments (``CHECKPOINTER_BACKEND=memory``
    plus ``EMBEDDING_BACKEND=hash``).
    """

    settings.configure_langsmith()
    memory = MemoryStore(settings.redis_url)
    oauth = GoogleOAuthManager(settings, memory)
    budget_provider, sheets_connector = build_budget_provider(settings, oauth)
    embedder = build_embedder(settings)

    database: Database | None = None
    use_postgres = settings.checkpointer_backend == "postgres" and bool(settings.database_url)
    documents = load_documents()

    if use_postgres:
        database = Database(settings.database_url, settings.embedding_dim)
        database.initialize()
        store: VectorStore = PgVectorStore(database.pool, embedder, table="rag_documents")
        checkpointer = build_checkpointer(database)
        logger.info("Using Postgres checkpointer and pgvector store")
    else:
        store = InMemoryVectorStore(embedder, documents if settings.auto_ingest else [])
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer and vector store")

    if settings.auto_ingest and isinstance(store, PgVectorStore):
        written = ingest_into_store(store, force=False)
        logger.info("RAG corpus ingest written=%s total=%s", written, store.count())

    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.reranker_enabled
        else NoOpReranker()
    )
    retriever = HybridRetriever(store, embedder, settings, reranker=reranker)
    retriever.refresh_index()

    llm = LLMClient(settings, memory)
    workflow = WorkflowService(
        settings=settings,
        llm=llm,
        retriever=retriever,
        budget_provider=budget_provider,
        memory=memory,
        checkpointer=checkpointer,
    )
    return AppRuntime(
        settings=settings,
        memory=memory,
        embedder=embedder,
        store=store,
        retriever=retriever,
        oauth=oauth,
        sheets_connector=sheets_connector,
        budget_provider=budget_provider,
        llm=llm,
        workflow=workflow,
        db=database,
    )
