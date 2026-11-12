"""MCP server exposing supplier lookup as reusable tools.

Run standalone with stdio (for Claude Desktop/Inspector):

    python -m mcp_server.server

Or as an HTTP service (used by Docker Compose):

    MCP_TRANSPORT=streamable-http MCP_PORT=8100 python -m mcp_server.server
"""

from __future__ import annotations

import logging
import os

from mcp.server.mcpserver import MCPServer

from rag.bm25 import BM25Index
from rag.dataset import load_documents
from rag.types import Document

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_SUPPLIERS: list[Document] = [
    document for document in load_documents() if document.doc_type == "supplier"
]
_INDEX = BM25Index(_SUPPLIERS)
_BY_ID = {document.doc_id: document for document in _SUPPLIERS}

server = MCPServer(
    name="procureflow-suppliers",
    version="0.1.0",
    instructions=(
        "Search the synthetic ProcureFlow supplier catalogue. Prices are USD and "
        "documents are fictional. Always preserve the returned doc_id as a citation."
    ),
)


def _supplier_payload(document: Document, score: float | None = None) -> dict:
    metadata = document.metadata
    return {
        "supplier_id": document.doc_id,
        "name": metadata.get("name", document.title),
        "category": metadata.get("category", "general"),
        "country": metadata.get("country", "Unknown"),
        "unit_price_usd": metadata.get("unit_price_usd"),
        "lead_time_days": metadata.get("lead_time_days"),
        "rating": metadata.get("rating"),
        "certifications": metadata.get("certifications", []),
        "summary": document.content,
        "citation": {
            "doc_id": document.doc_id,
            "title": document.title,
            "source": document.source,
        },
        "relevance_score": round(float(score), 4) if score is not None else None,
    }


@server.tool()
def search_suppliers(query: str, category: str | None = None, max_results: int = 3) -> list[dict]:
    """Search suppliers by free-text requirement and optional category."""

    limit = max(1, min(int(max_results), 10))
    hits = _INDEX.search(query, k=max(limit * 4, 12), doc_type="supplier")
    if category:
        normalised = category.strip().lower()
        hits = [hit for hit in hits if hit.document.metadata.get("category") == normalised]
    return [_supplier_payload(hit.document, hit.score) for hit in hits[:limit]]


@server.tool()
def get_supplier(supplier_id: str) -> dict:
    """Return one supplier by catalogue id (for example SUP-LAP-001)."""

    document = _BY_ID.get(supplier_id.strip().upper())
    if document is None:
        raise ValueError(f"Unknown supplier_id: {supplier_id}")
    return _supplier_payload(document)


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        server.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8100")),
        )
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
