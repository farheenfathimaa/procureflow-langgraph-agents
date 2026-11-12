"""Orchestrator: parse a request and decompose it into specialist tasks."""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base import BaseAgent
from app.config import Settings
from app.llm import LLMClient
from app.prompts import ORCHESTRATOR_SYSTEM
from app.schemas import AgentName, OrchestratorOutput, PlanTask, PurchaseRequest
from rag.hybrid import HybridRetriever

_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "laptops": ("laptop", "notebook", "macbook", "computer"),
    "monitors": ("monitor", "display", "screen"),
    "furniture": ("desk", "chair", "table", "furniture"),
    "software": ("software", "saas", "licence", "license", "subscription"),
    "security_software": ("security", "endpoint", "encryption", "phishing"),
    "mobile_devices": ("phone", "smartphone", "mobile", "tablet"),
    "ppe": ("ppe", "glove", "goggle", "vest", "mask", "safety"),
    "cloud": ("cloud", "compute", "storage", "credits"),
}
_TASK_OBJECTIVES: dict[AgentName, str] = {
    "policy": "Check the request against procurement, security and approval policies.",
    "suppliers": "Find and compare compliant suppliers for the requested item.",
    "budget": "Check available budget and whether the total fits.",
}


def _parse_amount(text: str) -> float:
    patterns = [
        r"\b(?:budget|total|max(?:imum)?|up\s+to|not\s+to\s+exceed)\b[^\d$]{0,20}"
        r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|thousand|million|mn)?",
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|thousand|million|mn)?",
        r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|thousand|million|mn)?\s*(?:usd|dollars?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1).replace(",", ""))
        suffix = (match.group(2) or "").lower()
        if suffix in {"k", "thousand"}:
            value *= 1_000
        elif suffix in {"m", "million", "mn"}:
            value *= 1_000_000
        return round(value, 2)
    return 0.0


def _parse_item(text: str) -> tuple[str, int]:
    phrase_match = re.search(
        r"\b(?:buy|purchase|order|procure|get|acquire)\b\s+(.*?)(?:,|\.|;|\bbudget\b|\bwith\b|\bfor\b|$)",
        text,
        re.IGNORECASE,
    )
    phrase = phrase_match.group(1).strip() if phrase_match else text.strip()
    quantity_match = re.search(r"\b(\d{1,6})\b", phrase)
    if quantity_match:
        quantity = int(quantity_match.group(1))
        phrase = phrase.replace(quantity_match.group(0), "", 1)
    else:
        any_number = re.search(r"\b(\d{1,6})\b", text)
        quantity = int(any_number.group(1)) if any_number else 1
    item = re.sub(r"\s+", " ", phrase).strip(" ,.-")
    item = re.sub(r"^(?:x\s*)", "", item, flags=re.IGNORECASE)
    item = re.sub(r"\b(?:units?|pcs?|pieces?)\b", "", item, flags=re.IGNORECASE).strip()
    return (item or "requested goods")[:300], max(quantity, 1)


def _parse_category(item: str) -> str:
    lowered = item.lower()
    for category, hints in _CATEGORY_HINTS.items():
        if any(hint in lowered for hint in hints):
            return category
    return "general"


def deterministic_parse(text: str) -> tuple[PurchaseRequest, list[PlanTask], float]:
    """Cheap regex parser used only if the orchestrator LLM is unavailable."""

    item, quantity = _parse_item(text)
    amount = _parse_amount(text)
    confidence = 0.9 if amount > 0 else 0.55
    request = PurchaseRequest(
        item=item,
        quantity=quantity,
        total_amount_usd=amount,
        category=_parse_category(item),
        confidence=confidence,
    )
    tasks = [
        PlanTask(agent=agent, objective=objective) for agent, objective in _TASK_OBJECTIVES.items()
    ]
    return request, tasks, confidence


class OrchestratorAgent(BaseAgent):
    """LangGraph node that owns request parsing and task decomposition."""

    name = "orchestrator"

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        retriever: HybridRetriever | None = None,
    ) -> None:
        super().__init__(llm, settings, retriever)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_request = state["user_request"]
        mock_request, mock_tasks, mock_confidence = deterministic_parse(user_request)
        mock_data = {
            "request": mock_request.model_dump(mode="json"),
            "tasks": [task.model_dump(mode="json") for task in mock_tasks],
            "rationale": "Policy, supplier and budget checks are required for every purchase.",
            "confidence": mock_confidence,
        }
        prompt = (
            "Purchase request:\n"
            f"{user_request}\n\n"
            "Requester: "
            f"{state.get('requester', 'unknown')}\n\n"
            "Create the structured plan. Always include policy, suppliers and budget tasks. "
            "Use the original item, quantity and total budget."
        )
        errors = list(state.get("errors", []))
        try:
            output = self.llm.generate_structured(
                OrchestratorOutput,
                ORCHESTRATOR_SYSTEM,
                prompt,
                mock_data=mock_data,
            )
            request = output.request
            tasks = self._normalise_tasks(output.tasks)
            confidence = output.confidence
            orchestrator = {
                "rationale": output.rationale,
                "confidence": confidence,
                "source": "llm",
            }
        except Exception as exc:
            errors.append(self.failure(self.name, exc))
            request, tasks, confidence = deterministic_parse(user_request)
            orchestrator = {
                "rationale": (
                    "LLM unavailable; used deterministic parsing and a safe approval path."
                ),
                "confidence": confidence,
                "source": "fallback_parser",
            }

        return {
            "parsed_request": request.model_dump(mode="json"),
            "plan": [task.model_dump(mode="json") for task in tasks],
            "task_index": 0,
            "iterations": 0,
            "orchestrator": orchestrator,
            "status": "processing",
            "errors": errors,
        }

    @staticmethod
    def _normalise_tasks(tasks: list[PlanTask]) -> list[PlanTask]:
        """Deduplicate and bound the plan; every purchase needs all three checks."""

        found: dict[AgentName, PlanTask] = {}
        for task in tasks:
            found.setdefault(task.agent, task)
        ordered: list[PlanTask] = []
        for agent in ("policy", "suppliers", "budget"):
            ordered.append(
                found.get(agent) or PlanTask(agent=agent, objective=_TASK_OBJECTIVES[agent])
            )
        return ordered[:6]


def request_summary(state: dict[str, Any]) -> str:
    """Compact request text used in retrieval queries."""

    parsed = state.get("parsed_request") or {}
    return json.dumps(
        {
            "item": parsed.get("item"),
            "quantity": parsed.get("quantity"),
            "amount_usd": parsed.get("total_amount_usd"),
            "category": parsed.get("category"),
            "specifications": parsed.get("specifications", []),
        },
        separators=(",", ":"),
    )
