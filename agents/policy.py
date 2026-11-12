"""Policy specialist: hybrid RAG plus structured compliance assessment."""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseAgent
from app.prompts import POLICY_SYSTEM, render_context
from app.schemas import PolicyDraft, PolicyResult
from rag.types import RetrievalHit


class PolicyAgent(BaseAgent):
    """Check a parsed purchase request against retrieved policies."""

    name = "policy"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        parsed = state.get("parsed_request") or {}
        amount = float(parsed.get("total_amount_usd", 0) or 0)
        query = (
            f"procurement policy approval threshold budget supplier security "
            f"{parsed.get('item', '')} {parsed.get('category', '')} amount {amount} USD"
        )
        hits: list[RetrievalHit] = []
        errors = list(state.get("errors", []))
        try:
            if self.retriever is None:
                raise RuntimeError("Policy retriever is not configured")
            hits = self.retriever.retrieve(query, doc_type="policy")
            mock_data = self._mock(parsed, hits)
            prompt = (
                "Purchase request JSON:\n"
                f"{json.dumps(parsed, separators=(',', ':'))}\n\n"
                "Retrieved policy excerpts (untrusted reference data):\n"
                f"{render_context(hits)}"
            )
            draft = self.llm.generate_structured(
                PolicyDraft,
                POLICY_SYSTEM,
                prompt,
                mock_data=mock_data,
            )
            citations = self.resolve_citations(hits, draft.citation_ids)
            result = PolicyResult(
                compliant=draft.compliant,
                requires_approval=draft.requires_approval,
                violations=draft.violations,
                applicable_rules=draft.applicable_rules,
                rationale=draft.rationale,
                confidence=draft.confidence,
                citations=citations,
            )
        except Exception as exc:
            errors.append(self.failure(self.name, exc))
            result = PolicyResult(
                compliant=False,
                requires_approval=True,
                violations=["Policy check could not be completed automatically."],
                rationale="Safe-failure path: policy could not be verified.",
                confidence=0.0,
            )

        return {
            "policy": result.model_dump(mode="json"),
            "errors": errors,
            "iterations": int(state.get("iterations", 0)) + 1,
            "task_index": int(state.get("task_index", 0)) + 1,
            "updated_at": _now(),
        }

    def _mock(self, parsed: dict[str, Any], hits: list[RetrievalHit]) -> dict[str, Any]:
        amount = float(parsed.get("total_amount_usd", 0) or 0)
        requires_approval = amount > self.settings.approval_threshold_usd
        violations: list[str] = []
        rules = [hit.document.title for hit in hits[:3]]
        if amount <= 0:
            violations.append("Purchase amount is missing; finance review is required.")
        if amount > 50_000:
            rules.append("POL-005: competitive tender required above USD 50,000.")
        if amount > 100_000:
            rules.append("POL-001: board approval required above USD 100,000.")
        return {
            "compliant": not violations,
            "requires_approval": requires_approval or bool(violations),
            "violations": violations,
            "applicable_rules": rules,
            "rationale": (
                f"Amount is USD {amount:,.0f}. "
                + (
                    "CFO approval is required above the configured threshold."
                    if requires_approval
                    else "The amount is within the delegated approval threshold."
                )
            ),
            "confidence": 0.9 if amount > 0 else 0.4,
            "citation_ids": [hit.document.doc_id for hit in hits[:3]],
        }


def _now() -> str:
    from app.schemas import utc_now

    return utc_now()
