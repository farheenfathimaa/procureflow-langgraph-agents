"""Corpus ingestion CLI and reusable seeding helper."""

from __future__ import annotations

import argparse
import logging

from app.config import get_settings
from rag.dataset import load_documents
from rag.embeddings import build_embedder
from rag.vector_store import PgVectorStore

logger = logging.getLogger(__name__)


def ingest_into_store(store, force: bool = False) -> int:
    """Upsert the bundled corpus and return the number of documents written."""

    documents = load_documents()
    if not force and store.count() >= len(documents):
        logger.info("Corpus already present (%s documents); skipping ingest", store.count())
        return 0
    written = store.upsert_documents(documents)
    logger.info("Upserted %s documents", written)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the synthetic RAG corpus")
    parser.add_argument("--force", action="store_true", help="Re-embed every document")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    from app.db import Database

    database = Database(settings.database_url)
    database.initialize()
    try:
        embedder = build_embedder(settings)
        store = PgVectorStore(database.pool, embedder)
        count = ingest_into_store(store, force=args.force)
        total = store.count()
        print(f"ingested={count} total_documents={total}")
    finally:
        database.close()


if __name__ == "__main__":
    main()
