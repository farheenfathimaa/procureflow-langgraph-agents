"""Pydantic contracts used at every agent and API boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgentName = Literal["policy", "suppliers", "budget"]
Decision = Literal["approve", "reject"]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON state."""

    return datetime.now(UTC).isoformat()


class Citation(BaseModel):
    """A traceable reference to a supplier or policy document."""

    doc_id: str
    title: str
    source: str
    doc_type: str
    snippet: str
    score: float = 0.0


class GuardrailResult(BaseModel):
    """Outcome of the deterministic input guardrail."""

    allowed: bool
    reason: str | None = None
    matched_pattern: str | None = None


class PurchaseRequest(BaseModel):
    """Normalised purchase request extracted by the orchestrator."""

    model_config = ConfigDict(extra="ignore")

    item: str = Field(min_length=1, max_length=300)
    quantity: int = Field(ge=1, le=1_000_000)
    total_amount_usd: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    category: str = "general"
    specifications: list[str] = Field(default_factory=list)
    needed_by: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, value: str) -> str:
        return value.upper()


class PlanTask(BaseModel):
    """One specialist assignment produced by the orchestrator."""

    agent: AgentName
    objective: str = Field(min_length=3, max_length=500)


class OrchestratorOutput(BaseModel):
    """Structured decomposition of the user's request."""

    request: PurchaseRequest
    tasks: list[PlanTask] = Field(min_length=1, max_length=6)
    rationale: str = Field(default="", max_length=1000)
    confidence: float = Field(default=0.5, ge=0, le=1)


class SupplierCandidate(BaseModel):
    """A supplier option with commercial details."""

    name: str
    country: str = "Unknown"
    unit_price_usd: float = Field(default=0, ge=0)
    total_price_usd: float = Field(default=0, ge=0)
    lead_time_days: int | None = Field(default=None, ge=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    rationale: str = ""
    citation_ids: list[str] = Field(default_factory=list)


class SupplierDraft(BaseModel):
    """LLM-facing supplier response; citations are resolved afterwards."""

    candidates: list[SupplierCandidate] = Field(default_factory=list, max_length=5)
    recommended_supplier: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class SupplierResult(BaseModel):
    """Validated supplier agent output stored in graph state."""

    candidates: list[SupplierCandidate] = Field(default_factory=list)
    recommended_supplier: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""


class PolicyDraft(BaseModel):
    """LLM-facing policy response; citations are resolved afterwards."""

    compliant: bool
    requires_approval: bool
    violations: list[str] = Field(default_factory=list)
    applicable_rules: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list)


class PolicyResult(BaseModel):
    """Validated policy agent output stored in graph state."""

    compliant: bool = True
    requires_approval: bool = False
    violations: list[str] = Field(default_factory=list)
    applicable_rules: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)


class BudgetResult(BaseModel):
    """Budget check result. ``within_budget`` is None when it cannot be known."""

    budget_usd: float = 0.0
    committed_usd: float = 0.0
    available_usd: float = 0.0
    requested_usd: float = 0.0
    within_budget: bool | None = None
    source: str = "unknown"
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: str = ""


class RiskAssessment(BaseModel):
    """Deterministic aggregation of confidence, amount and agent failures."""

    approval_required: bool
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    amount_usd: float = 0.0


class ApprovalDecisionRequest(BaseModel):
    """Reviewer decision submitted to the resume endpoint."""

    decision: Decision
    approver: str = Field(min_length=2, max_length=200)
    comment: str = Field(default="", max_length=2000)


class ApprovalActionRequest(BaseModel):
    """Body for the convenience approve/reject endpoints."""

    approver: str = Field(min_length=2, max_length=200)
    comment: str = Field(default="", max_length=2000)


class ApprovalRecord(BaseModel):
    """A recorded human decision."""

    decision: Decision
    approver: str
    comment: str = ""
    decided_at: str = Field(default_factory=utc_now)


class Recommendation(BaseModel):
    """Final, human-readable procurement recommendation."""

    summary: str
    status: str
    chosen_supplier: str | None = None
    total_amount_usd: float = 0.0
    supplier_rationale: str = ""
    policy_summary: str = ""
    budget_summary: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)


class RequestCreate(BaseModel):
    """Public API input for starting a procurement workflow."""

    request: str = Field(min_length=5, max_length=4000)
    requester: str = Field(default="demo@company.com", min_length=3, max_length=200)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class WorkflowView(BaseModel):
    """Serializable view of a checkpointed workflow."""

    thread_id: str
    status: str
    requester: str | None = None
    user_request: str | None = None
    parsed_request: dict | None = None
    plan: list[dict] = Field(default_factory=list)
    guardrail: dict | None = None
    policy: dict | None = None
    suppliers: dict | None = None
    budget: dict | None = None
    risk: dict | None = None
    approval: dict | None = None
    recommendation: dict | None = None
    interrupt: dict | None = None
    errors: list[dict] = Field(default_factory=list)
    can_resume: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class OAuthStatus(BaseModel):
    """Non-secret OAuth connection status."""

    configured: bool
    authorized: bool
    provider: str = "google"
    token_path: str | None = None
    expiry: str | None = None
    scopes: list[str] = Field(default_factory=list)
    message: str | None = None


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: Literal["ok", "degraded"]
    app: str
    environment: str
    database: str
    redis: str
    vector_documents: int
    llm: dict[str, bool]
    langsmith: bool
