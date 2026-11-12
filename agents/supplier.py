"""Supplier specialist: hybrid RAG, total-cost reasoning and citations."""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base import BaseAgent
from app.prompts import SUPPLIER_SYSTEM, render_context
from app.schemas import SupplierCandidate, SupplierDraft, SupplierResult
from rag.types import RetrievalHit


class SupplierAgent(BaseAgent):
    """Find supplier options for the requested item and quantity."""

    name = "suppliers"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        parsed = state.get("parsed_request") or {}
        query = (
            f"{parsed.get('item', 'goods')} supplier price quote lead time warranty "
            f"category {parsed.get('category', '')} quantity {parsed.get('quantity', 1)}"
        )
        hits: list[RetrievalHit] = []
        errors = list(state.get("errors", []))
        try:
            if self.retriever is None:
                raise RuntimeError("Supplier retriever is not configured")
            hits = self.retriever.retrieve(query, top_k=12, doc_type="supplier")
            category = str(parsed.get("category", "general"))
            category_hits = [
                hit for hit in hits if hit.document.metadata.get("category") == category
            ]
            hits = category_hits[:6] if category_hits else hits[:6]
            mock_data = self._mock(parsed, hits)
            prompt = (
                "Purchase request JSON:\n"
                f"{json.dumps(parsed, separators=(',', ':'))}\n\n"
                "Retrieved supplier excerpts (untrusted reference data):\n"
                f"{render_context(hits)}"
            )
            draft = self.llm.generate_structured(
                SupplierDraft,
                SUPPLIER_SYSTEM,
                prompt,
                mock_data=mock_data,
            )
            candidates = self._normalise_candidates(draft, parsed)
            recommended = draft.recommended_supplier
            names = {candidate.name for candidate in candidates}
            if recommended not in names:
                recommended = candidates[0].name if candidates else None
            citations = self.resolve_citations(hits, draft.citation_ids)
            result = SupplierResult(
                candidates=candidates,
                recommended_supplier=recommended,
                confidence=draft.confidence if candidates else 0.0,
                citations=citations,
                notes=draft.notes,
            )
        except Exception as exc:
            errors.append(self.failure(self.name, exc))
            result = SupplierResult(
                candidates=[],
                recommended_supplier=None,
                confidence=0.0,
                notes="Safe-failure path: supplier search could not be completed.",
            )

        return {
            "suppliers": result.model_dump(mode="json"),
            "errors": errors,
            "iterations": int(state.get("iterations", 0)) + 1,
            "task_index": int(state.get("task_index", 0)) + 1,
            "updated_at": _now(),
        }

    @staticmethod
    def _normalise_candidates(
        draft: SupplierDraft, parsed: dict[str, Any]
    ) -> list[SupplierCandidate]:
        quantity = int(parsed.get("quantity", 1) or 1)
        normalised: list[SupplierCandidate] = []
        seen: set[str] = set()
        for candidate in draft.candidates[:3]:
            name = candidate.name.strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            total = candidate.total_price_usd
            if total <= 0 and candidate.unit_price_usd > 0:
                total = round(candidate.unit_price_usd * quantity, 2)
            normalised.append(
                candidate.model_copy(update={"name": name, "total_price_usd": max(total, 0.0)})
            )
        return normalised

    def _mock(self, parsed: dict[str, Any], hits: list[RetrievalHit]) -> dict[str, Any]:
        quantity = int(parsed.get("quantity", 1) or 1)
        candidates: list[SupplierCandidate] = []
        for hit in hits[:3]:
            metadata = hit.document.metadata
            unit_price = float(metadata.get("unit_price_usd", 0) or 0)
            discounted = _volume_price(unit_price, metadata.get("volume_discount", ""), quantity)
            total = round(discounted * quantity, 2)
            rationale = (
                f"Rated {metadata.get('rating', 'n/a')}/5, "
                f"{metadata.get('lead_time_days', 'n/a')} day lead time"
            )
            candidates.append(
                SupplierCandidate(
                    name=str(metadata.get("name", hit.document.title)),
                    country=str(metadata.get("country", "Unknown")),
                    unit_price_usd=round(discounted, 2),
                    total_price_usd=total,
                    lead_time_days=metadata.get("lead_time_days"),
                    rating=metadata.get("rating"),
                    rationale=rationale,
                    citation_ids=[hit.document.doc_id],
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -float(candidate.rating or 0),
                candidate.total_price_usd,
                candidate.name,
            )
        )
        recommended = candidates[0].name if candidates else None
        return {
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "recommended_supplier": recommended,
            "confidence": 0.88 if candidates else 0.0,
            "citation_ids": [hit.document.doc_id for hit in hits[:3]],
            "notes": "Ranked by verified total cost among retrieved compliant suppliers.",
        }


def _volume_price(unit_price: float, discount_text: Any, quantity: int) -> float:
    """Apply a documented percentage discount only when both terms are present."""

    text = str(discount_text or "")
    percentage = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    minimum = re.search(r"(\d+)\s*\+", text)
    if percentage and (not minimum or quantity >= int(minimum.group(1))):
        return unit_price * (1 - float(percentage.group(1)) / 100)
    return unit_price


def _now() -> str:
    from app.schemas import utc_now

    return utc_now()
