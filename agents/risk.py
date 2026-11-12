"""Deterministic risk gate that decides whether a human must approve."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.schemas import RiskAssessment


class RiskAssessor:
    """Combine confidence, amount, policy, budget and failure signals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess(self, state: dict[str, Any]) -> RiskAssessment:
        parsed = state.get("parsed_request") or {}
        policy = state.get("policy") or {}
        suppliers = state.get("suppliers") or {}
        budget = state.get("budget") or {}
        orchestrator = state.get("orchestrator") or {}
        amount = float(parsed.get("total_amount_usd", 0) or 0)

        confidences = [
            float(value)
            for value in (
                parsed.get("confidence"),
                orchestrator.get("confidence"),
                policy.get("confidence"),
                suppliers.get("confidence"),
                budget.get("confidence"),
            )
            if value is not None
        ]
        confidence = min(confidences) if confidences else 0.0
        reasons: list[str] = []

        if amount > self.settings.approval_threshold_usd:
            reasons.append(
                f"Amount USD {amount:,.2f} exceeds the USD "
                f"{self.settings.approval_threshold_usd:,.0f} delegated limit."
            )
        if confidence < self.settings.confidence_threshold:
            reasons.append(
                f"Agent confidence {confidence:.2f} is below the "
                f"{self.settings.confidence_threshold:.2f} threshold."
            )
        if policy.get("requires_approval"):
            reasons.append("Policy requires human approval.")
        if policy and not policy.get("compliant", True):
            reasons.append("Policy violations were reported.")
        if budget.get("within_budget") is False:
            reasons.append("The requested amount exceeds available budget.")
        if budget.get("within_budget") is None:
            reasons.append("Budget availability could not be verified.")
        if not suppliers:
            reasons.append("Supplier search did not return a verified recommendation.")
        elif not suppliers.get("recommended_supplier"):
            reasons.append("No suitable supplier was recommended.")
        if state.get("errors"):
            reasons.append("One or more agent steps failed and need review.")
        if amount <= 0:
            reasons.append("Purchase amount is missing.")

        return RiskAssessment(
            approval_required=bool(reasons),
            reasons=reasons,
            confidence=round(confidence, 4),
            amount_usd=amount,
        )
