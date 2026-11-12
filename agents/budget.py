"""Budget specialist backed by Google Sheets or a labelled local demo ledger."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from app.schemas import BudgetResult
from connectors.google_sheets import BudgetConnectorError, BudgetProvider


class BudgetAgent(BaseAgent):
    """Check available funds without silently hiding connector errors."""

    name = "budget"

    def __init__(
        self,
        llm,
        settings,
        budget_provider: BudgetProvider,
        retriever=None,
    ) -> None:
        super().__init__(llm, settings, retriever)
        self.budget_provider = budget_provider

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        parsed = state.get("parsed_request") or {}
        requested = float(parsed.get("total_amount_usd", 0) or 0)
        errors = list(state.get("errors", []))
        try:
            snapshot = self.budget_provider.fetch_budget()
            within_budget = requested <= snapshot.available_usd if requested > 0 else None
            confidence = 0.98 if snapshot.source.startswith("google_sheets:") else 0.85
            if requested <= 0:
                notes = "Requested amount is missing; a human must confirm the amount."
            elif within_budget:
                notes = "Requested total fits within the available budget."
            else:
                notes = "Requested total exceeds the available budget."
            result = BudgetResult(
                budget_usd=snapshot.total_budget_usd,
                committed_usd=snapshot.committed_usd,
                available_usd=snapshot.available_usd,
                requested_usd=requested,
                within_budget=within_budget,
                source=snapshot.source,
                confidence=confidence,
                notes=notes,
            )
        except BudgetConnectorError as exc:
            errors.append(self.failure(self.name, exc))
            result = BudgetResult(
                requested_usd=requested,
                within_budget=None,
                source="unavailable",
                confidence=0.0,
                notes=str(exc),
            )
        except Exception as exc:
            errors.append(self.failure(self.name, exc))
            result = BudgetResult(
                requested_usd=requested,
                within_budget=None,
                source="unavailable",
                confidence=0.0,
                notes="Budget could not be verified automatically.",
            )

        return {
            "budget": result.model_dump(mode="json"),
            "errors": errors,
            "iterations": int(state.get("iterations", 0)) + 1,
            "task_index": int(state.get("task_index", 0)) + 1,
            "updated_at": _now(),
        }


def _now() -> str:
    from app.schemas import utc_now

    return utc_now()
