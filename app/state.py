"""Shared typed LangGraph state.

Values are plain JSON-serialisable dictionaries because the Postgres
checkpointer persists state between HTTP requests. Pydantic models are
validated at each agent boundary and dumped with ``model_dump(mode="json")``
before entering state.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ProcurementState(TypedDict, total=False):
    """State passed explicitly between orchestrator and specialist nodes."""

    thread_id: str
    request_id: str
    requester: str
    user_request: str
    status: str

    guardrail: dict[str, Any]
    orchestrator: dict[str, Any]
    parsed_request: dict[str, Any]
    plan: list[dict[str, Any]]
    task_index: int
    iterations: int

    policy: dict[str, Any]
    suppliers: dict[str, Any]
    budget: dict[str, Any]
    risk: dict[str, Any]
    approval: dict[str, Any]
    recommendation: dict[str, Any]

    errors: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    created_at: str
    updated_at: str
