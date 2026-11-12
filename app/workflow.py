"""LangGraph workflow with explicit handoffs, safe failure and HITL interrupt."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.budget import BudgetAgent
from agents.orchestrator import OrchestratorAgent
from agents.policy import PolicyAgent
from agents.risk import RiskAssessor
from agents.supplier import SupplierAgent
from app.config import Settings
from app.guardrails import inspect_input
from app.llm import LLMClient
from app.memory import MemoryStore
from app.schemas import (
    ApprovalDecisionRequest,
    ApprovalRecord,
    Citation,
    Recommendation,
    WorkflowView,
    utc_now,
)
from app.state import ProcurementState
from connectors.google_sheets import BudgetProvider
from rag.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class WorkflowNotFoundError(KeyError):
    """No checkpoint exists for the requested thread."""


class WorkflowConflictError(RuntimeError):
    """The workflow is not currently waiting for an approval decision."""


class WorkflowService:
    """Build and operate the checkpointed procurement graph."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        retriever: HybridRetriever,
        budget_provider: BudgetProvider,
        memory: MemoryStore,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.orchestrator = OrchestratorAgent(llm, settings, retriever)
        self.policy_agent = PolicyAgent(llm, settings, retriever)
        self.supplier_agent = SupplierAgent(llm, settings, retriever)
        self.budget_agent = BudgetAgent(llm, settings, budget_provider)
        self.risk_assessor = RiskAssessor(settings)
        self.graph = self._build_graph(checkpointer)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(ProcurementState)
        graph.add_node("guardrail", self._guardrail_node)
        graph.add_node("orchestrator", self.orchestrator.run)
        graph.add_node("router", self._router_node)
        graph.add_node("policy_agent", self.policy_agent.run)
        graph.add_node("supplier_agent", self.supplier_agent.run)
        graph.add_node("budget_agent", self.budget_agent.run)
        graph.add_node("risk_gate", self._risk_node)
        graph.add_node("approval_gate", self._approval_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_node("blocked", self._blocked_node)

        graph.add_edge(START, "guardrail")
        graph.add_conditional_edges(
            "guardrail",
            self._guardrail_route,
            {"orchestrator": "orchestrator", "blocked": "blocked"},
        )
        graph.add_edge("orchestrator", "router")
        graph.add_conditional_edges(
            "router",
            self._select_next,
            {
                "policy_agent": "policy_agent",
                "supplier_agent": "supplier_agent",
                "budget_agent": "budget_agent",
                "risk_gate": "risk_gate",
            },
        )
        for agent_node in ("policy_agent", "supplier_agent", "budget_agent"):
            graph.add_edge(agent_node, "router")
        graph.add_conditional_edges(
            "risk_gate",
            self._risk_route,
            {"approval_gate": "approval_gate", "finalize": "finalize"},
        )
        graph.add_edge("approval_gate", "finalize")
        graph.add_edge("finalize", END)
        graph.add_edge("blocked", END)
        return graph.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _guardrail_node(self, state: ProcurementState) -> dict[str, Any]:
        result = inspect_input(state.get("user_request", ""))
        updates: dict[str, Any] = {
            "guardrail": result.model_dump(mode="json"),
            "updated_at": utc_now(),
        }
        if result.allowed:
            updates["status"] = "processing"
        else:
            updates["status"] = "blocked"
            updates["errors"] = [
                *state.get("errors", []),
                {
                    "agent": "guardrail",
                    "type": "InputBlocked",
                    "message": result.reason or "Input blocked by guardrail.",
                },
            ]
        return updates

    def _guardrail_route(self, state: ProcurementState) -> Literal["orchestrator", "blocked"]:
        return "orchestrator" if state.get("guardrail", {}).get("allowed") else "blocked"

    def _router_node(self, state: ProcurementState) -> dict[str, Any]:
        iterations = int(state.get("iterations", 0))
        plan = state.get("plan", [])
        index = int(state.get("task_index", 0))
        if index < len(plan) and iterations >= self.settings.max_graph_iterations:
            return {
                "task_index": len(plan),
                "status": "needs_review",
                "errors": [
                    *state.get("errors", []),
                    {
                        "agent": "router",
                        "type": "MaxIterationsExceeded",
                        "message": (
                            f"Stopped after {iterations} agent iterations; human review required."
                        ),
                    },
                ],
                "updated_at": utc_now(),
            }
        return {}

    def _select_next(self, state: ProcurementState) -> str:
        plan = state.get("plan", [])
        index = int(state.get("task_index", 0))
        if index >= len(plan):
            return "risk_gate"
        agent = plan[index].get("agent")
        return {
            "policy": "policy_agent",
            "suppliers": "supplier_agent",
            "budget": "budget_agent",
        }.get(agent, "risk_gate")

    def _risk_node(self, state: ProcurementState) -> dict[str, Any]:
        assessment = self.risk_assessor.assess(state)
        return {
            "risk": assessment.model_dump(mode="json"),
            "status": "awaiting_approval" if assessment.approval_required else "ready",
            "updated_at": utc_now(),
        }

    @staticmethod
    def _risk_route(state: ProcurementState) -> Literal["approval_gate", "finalize"]:
        risk = state.get("risk") or {}
        return "approval_gate" if risk.get("approval_required") else "finalize"

    def _approval_node(self, state: ProcurementState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        payload = self._approval_payload(state)
        self.memory.remember_request(thread_id, payload)
        self.memory.add_pending_approval(thread_id)

        resume_value = interrupt(payload)
        self.memory.remove_pending_approval(thread_id)
        try:
            decision = ApprovalDecisionRequest.model_validate(resume_value)
        except Exception as exc:
            return {
                "status": "needs_review",
                "errors": [
                    *state.get("errors", []),
                    {
                        "agent": "approval",
                        "type": "InvalidApprovalDecision",
                        "message": f"Approval resume payload was invalid: {type(exc).__name__}",
                    },
                ],
                "updated_at": utc_now(),
            }

        record = ApprovalRecord(
            decision=decision.decision,
            approver=decision.approver,
            comment=decision.comment,
        )
        return {
            "approval": record.model_dump(mode="json"),
            "status": "approved" if decision.decision == "approve" else "rejected",
            "updated_at": utc_now(),
        }

    def _finalize_node(self, state: ProcurementState) -> dict[str, Any]:
        parsed = state.get("parsed_request") or {}
        policy = state.get("policy") or {}
        suppliers = state.get("suppliers") or {}
        budget = state.get("budget") or {}
        risk = state.get("risk") or {}
        approval = state.get("approval") or {}
        status = state.get("status", "completed")

        if status not in {"approved", "rejected"}:
            status = "completed"

        chosen = suppliers.get("recommended_supplier")
        candidate = next(
            (item for item in suppliers.get("candidates", []) if item.get("name") == chosen),
            None,
        )
        amount = float(parsed.get("total_amount_usd", 0) or 0)
        if candidate and candidate.get("total_price_usd"):
            amount = float(candidate["total_price_usd"])

        citations = _dedupe_citations(
            [
                *policy.get("citations", []),
                *suppliers.get("citations", []),
            ]
        )
        policy_summary = _policy_summary(policy)
        budget_summary = _budget_summary(budget)
        item = parsed.get("item", "requested goods")
        quantity = parsed.get("quantity", 1)
        supplier_text = chosen or "no supplier selected"
        summary = (
            f"{status.replace('_', ' ').title()}: {quantity} × {item} from {supplier_text}. "
            f"Estimated total USD {amount:,.2f}. {policy_summary} {budget_summary}"
        ).strip()
        if status == "rejected":
            summary = f"Rejected by {approval.get('approver', 'reviewer')}. {summary}"

        recommendation = Recommendation(
            summary=summary,
            status=status,
            chosen_supplier=chosen,
            total_amount_usd=round(amount, 2),
            supplier_rationale=_supplier_rationale(suppliers, candidate),
            policy_summary=policy_summary,
            budget_summary=budget_summary,
            confidence=float(risk.get("confidence", 0) or 0),
            citations=citations,
        )
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "status": status,
            "citations": [citation.model_dump(mode="json") for citation in citations],
            "updated_at": utc_now(),
        }

    def _blocked_node(self, state: ProcurementState) -> dict[str, Any]:
        guardrail = state.get("guardrail") or {}
        recommendation = Recommendation(
            summary=f"Request blocked: {guardrail.get('reason', 'guardrail violation')}",
            status="blocked",
            confidence=0.0,
        )
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "status": "blocked",
            "updated_at": utc_now(),
        }

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    def start(self, user_request: str, requester: str, currency: str = "USD") -> WorkflowView:
        """Start a new checkpointed workflow and return its first observable state."""

        thread_id = str(uuid.uuid4())
        now = utc_now()
        initial: ProcurementState = {
            "thread_id": thread_id,
            "request_id": thread_id,
            "requester": requester,
            "user_request": user_request,
            "status": "queued",
            "plan": [],
            "task_index": 0,
            "iterations": 0,
            "errors": [],
            "citations": [],
            "created_at": now,
            "updated_at": now,
        }
        if currency.upper() != "USD":
            initial["errors"] = [
                {
                    "agent": "orchestrator",
                    "type": "CurrencyAssumption",
                    "message": (
                        f"Request currency {currency.upper()} is recorded, but budget "
                        "comparisons assume the supplied amount is USD."
                    ),
                }
            ]
        config = self._config(thread_id)
        self.graph.invoke(initial, config=config, recursion_limit=self._recursion_limit())
        return self.get(thread_id)

    def resume(self, thread_id: str, decision: ApprovalDecisionRequest) -> WorkflowView:
        """Resume an interrupted workflow from its Postgres checkpoint."""

        snapshot = self.graph.get_state(self._config(thread_id))
        if not _snapshot_values(snapshot):
            raise WorkflowNotFoundError(thread_id)
        if not getattr(snapshot, "interrupts", ()):
            raise WorkflowConflictError("Workflow is not waiting for an approval decision.")

        self.graph.invoke(
            Command(resume=decision.model_dump(mode="json")),
            config=self._config(thread_id),
            recursion_limit=self._recursion_limit(),
        )
        return self.get(thread_id)

    def get(self, thread_id: str) -> WorkflowView:
        """Read a workflow snapshot and turn it into an API-safe view."""

        snapshot = self.graph.get_state(self._config(thread_id))
        values = _snapshot_values(snapshot)
        if not values:
            raise WorkflowNotFoundError(thread_id)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        interrupt_payload = _interrupt_value(interrupts)
        can_resume = bool(interrupts) and "approval_gate" in tuple(
            getattr(snapshot, "next", ()) or ()
        )
        view = WorkflowView(
            thread_id=thread_id,
            status="awaiting_approval" if can_resume else values.get("status", "unknown"),
            requester=values.get("requester"),
            user_request=values.get("user_request"),
            parsed_request=values.get("parsed_request"),
            plan=values.get("plan", []),
            guardrail=values.get("guardrail"),
            policy=values.get("policy"),
            suppliers=values.get("suppliers"),
            budget=values.get("budget"),
            risk=values.get("risk"),
            approval=values.get("approval"),
            recommendation=values.get("recommendation"),
            interrupt=interrupt_payload,
            errors=values.get("errors", []),
            can_resume=can_resume,
            created_at=values.get("created_at"),
            updated_at=values.get("updated_at"),
        )
        return view

    def list_pending(self) -> list[WorkflowView]:
        """Return workflows currently waiting at the approval gate."""

        pending: list[WorkflowView] = []
        for thread_id in self.memory.pending_approvals():
            try:
                view = self.get(thread_id)
            except WorkflowNotFoundError:
                self.memory.remove_pending_approval(thread_id)
                continue
            if view.can_resume:
                pending.append(view)
            else:
                self.memory.remove_pending_approval(thread_id)
        return pending

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _recursion_limit(self) -> int:
        return max(self.settings.max_graph_iterations + 6, 10)

    def _approval_payload(self, state: ProcurementState) -> dict[str, Any]:
        parsed = state.get("parsed_request") or {}
        policy = state.get("policy") or {}
        suppliers = state.get("suppliers") or {}
        budget = state.get("budget") or {}
        risk = state.get("risk") or {}
        candidate = next(
            (
                item
                for item in suppliers.get("candidates", [])
                if item.get("name") == suppliers.get("recommended_supplier")
            ),
            None,
        )
        return {
            "thread_id": state["thread_id"],
            "request_id": state["request_id"],
            "request_summary": (
                f"{parsed.get('quantity', 1)} × {parsed.get('item', 'requested goods')}"
            ),
            "amount_usd": float(parsed.get("total_amount_usd", 0) or 0),
            "confidence": float(risk.get("confidence", 0) or 0),
            "reasons": risk.get("reasons", []),
            "recommended_supplier": suppliers.get("recommended_supplier"),
            "recommended_total_usd": candidate.get("total_price_usd") if candidate else None,
            "policy_summary": policy.get("rationale", ""),
            "budget_available_usd": budget.get("available_usd"),
            "requested_by": state.get("requester"),
            "created_at": utc_now(),
        }


def _snapshot_values(snapshot: Any) -> dict[str, Any]:
    values = getattr(snapshot, "values", None)
    return dict(values) if values else {}


def _interrupt_value(interrupts: Any) -> dict[str, Any] | None:
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", None)
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[Citation]:
    seen: set[str] = set()
    result: list[Citation] = []
    for raw in citations:
        try:
            citation = Citation.model_validate(raw)
        except Exception:
            continue
        if citation.doc_id in seen:
            continue
        seen.add(citation.doc_id)
        result.append(citation)
    return result


def _policy_summary(policy: dict[str, Any]) -> str:
    if not policy:
        return "Policy check unavailable."
    rules = ", ".join(policy.get("applicable_rules", [])[:2])
    base = "Policy compliant." if policy.get("compliant") else "Policy violations reported."
    if policy.get("requires_approval"):
        base += " Approval required."
    return f"{base} {rules}".strip()


def _budget_summary(budget: dict[str, Any]) -> str:
    if not budget:
        return "Budget check unavailable."
    if budget.get("within_budget") is True:
        return (
            f"Within budget (USD {budget.get('available_usd', 0):,.0f} available, "
            f"source: {budget.get('source', 'unknown')})."
        )
    if budget.get("within_budget") is False:
        return (
            f"Over budget (USD {budget.get('available_usd', 0):,.0f} available, "
            f"requested USD {budget.get('requested_usd', 0):,.0f})."
        )
    return f"Budget unverified (source: {budget.get('source', 'unknown')})."


def _supplier_rationale(suppliers: dict[str, Any], candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return suppliers.get("notes", "No supplier recommended.")
    parts = [candidate.get("rationale", "")]
    if candidate.get("lead_time_days") is not None:
        parts.append(f"Lead time {candidate['lead_time_days']} days.")
    return " ".join(part for part in parts if part).strip()
