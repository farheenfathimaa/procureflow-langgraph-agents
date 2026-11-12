"""Shared agent helpers: citations, confidence and safe error reporting."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm import LLMClient
from rag.hybrid import HybridRetriever
from rag.types import RetrievalHit, hit_to_citation

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """A recoverable specialist-agent failure."""


class BaseAgent:
    """Base class for structured, citation-producing specialist agents."""

    name = "agent"

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.retriever = retriever

    def resolve_citations(
        self, hits: list[RetrievalHit], citation_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Resolve LLM-cited ids against actual retrieval hits.

        Hallucinated ids are dropped. If the model returned no valid citation,
        the two strongest retrieved documents are cited so the result remains
        auditable.
        """

        by_id = {hit.document.doc_id: hit for hit in hits}
        selected: list[RetrievalHit] = []
        for doc_id in citation_ids or []:
            hit = by_id.get(doc_id)
            if hit and all(existing.document.doc_id != doc_id for existing in selected):
                selected.append(hit)
        if not selected:
            selected = hits[:2]
        return [hit_to_citation(hit).model_dump(mode="json") for hit in selected]

    @staticmethod
    def failure(agent: str, exc: Exception) -> dict[str, Any]:
        """Build a bounded error record without leaking stack traces or secrets."""

        logger.exception("Agent %s failed", agent)
        message = str(exc).strip()[:500] or type(exc).__name__
        return {
            "agent": agent,
            "type": type(exc).__name__,
            "message": message,
        }

    @staticmethod
    def append_error(state: dict[str, Any], error: dict[str, Any]) -> list[dict[str, Any]]:
        return [*state.get("errors", []), error]
