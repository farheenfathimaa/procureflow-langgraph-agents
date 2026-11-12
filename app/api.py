"""FastAPI routes for procurement workflows, approvals, OAuth and health."""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.runtime import AppRuntime
from app.schemas import (
    ApprovalActionRequest,
    ApprovalDecisionRequest,
    HealthResponse,
    OAuthStatus,
    RequestCreate,
    WorkflowView,
)
from app.workflow import WorkflowConflictError, WorkflowNotFoundError
from connectors.google_oauth import (
    OAuthError,
    OAuthNotAuthorizedError,
    OAuthNotConfiguredError,
)
from connectors.google_sheets import BudgetConnectorError

logger = logging.getLogger(__name__)

router = APIRouter()
auth_router = APIRouter(prefix="/auth", tags=["oauth"])


def get_runtime(request: Request) -> AppRuntime:
    """Resolve the runtime created by the application lifespan."""

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - lifespan guarantees this
        raise HTTPException(status_code=503, detail="Application runtime is not ready.")
    return runtime


RuntimeDep = Annotated[AppRuntime, Depends(get_runtime)]


# ---------------------------------------------------------------------------
# Procurement workflow
# ---------------------------------------------------------------------------
@router.post(
    "/api/v1/requests",
    response_model=WorkflowView,
    status_code=201,
    tags=["procurement"],
    summary="Submit a purchase request and run the agent workflow",
)
async def create_request(payload: RequestCreate, runtime: RuntimeDep) -> WorkflowView:
    """Run policy, supplier and budget checks and pause for approval if needed."""

    try:
        return await asyncio.to_thread(
            runtime.workflow.start,
            payload.request,
            payload.requester,
            payload.currency,
        )
    except Exception as exc:
        logger.exception("Failed to start workflow")
        raise HTTPException(status_code=500, detail="Workflow could not be started.") from exc


@router.get(
    "/api/v1/requests/{thread_id}",
    response_model=WorkflowView,
    tags=["procurement"],
    summary="Get a workflow snapshot from the Postgres checkpoint",
)
async def get_request(thread_id: str, runtime: RuntimeDep) -> WorkflowView:
    """Read the current state, including any pending approval interrupt."""

    try:
        return await asyncio.to_thread(runtime.workflow.get, thread_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workflow not found.") from exc


@router.get(
    "/api/v1/approvals",
    response_model=list[WorkflowView],
    tags=["approvals"],
    summary="List workflows waiting for human approval",
)
async def list_approvals(runtime: RuntimeDep) -> list[WorkflowView]:
    return await asyncio.to_thread(runtime.workflow.list_pending)


@router.post(
    "/api/v1/approvals/{thread_id}/decision",
    response_model=WorkflowView,
    tags=["approvals"],
    summary="Approve or reject a paused workflow and resume from its checkpoint",
)
async def decide(
    thread_id: str, payload: ApprovalDecisionRequest, runtime: RuntimeDep
) -> WorkflowView:
    """Resume the graph with a validated human decision."""

    return await _resume(runtime, thread_id, payload)


@router.post(
    "/api/v1/requests/{thread_id}/approve",
    response_model=WorkflowView,
    tags=["approvals"],
    summary="Convenience endpoint to approve a paused workflow",
)
async def approve(
    thread_id: str,
    payload: ApprovalActionRequest,
    runtime: RuntimeDep,
) -> WorkflowView:
    decision = ApprovalDecisionRequest(
        decision="approve",
        approver=payload.approver,
        comment=payload.comment,
    )
    return await _resume(runtime, thread_id, decision)


@router.post(
    "/api/v1/requests/{thread_id}/reject",
    response_model=WorkflowView,
    tags=["approvals"],
    summary="Convenience endpoint to reject a paused workflow",
)
async def reject(
    thread_id: str,
    payload: ApprovalActionRequest,
    runtime: RuntimeDep,
) -> WorkflowView:
    decision = ApprovalDecisionRequest(
        decision="reject",
        approver=payload.approver,
        comment=payload.comment,
    )
    return await _resume(runtime, thread_id, decision)


async def _resume(
    runtime: AppRuntime, thread_id: str, decision: ApprovalDecisionRequest
) -> WorkflowView:
    try:
        return await asyncio.to_thread(runtime.workflow.resume, thread_id, decision)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workflow not found.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to resume workflow %s", thread_id)
        raise HTTPException(status_code=500, detail="Workflow could not be resumed.") from exc


# ---------------------------------------------------------------------------
# Google OAuth + Sheets
# ---------------------------------------------------------------------------
@auth_router.get(
    "/google",
    summary="Start the Google OAuth 2.0 Authorization Code flow",
    response_class=RedirectResponse,
)
async def google_authorize(runtime: RuntimeDep) -> RedirectResponse:
    try:
        url = await asyncio.to_thread(runtime.oauth.authorization_url)
        return RedirectResponse(url)
    except OAuthNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@auth_router.get(
    "/google/callback",
    summary="Google OAuth callback; exchanges the code and stores refreshable tokens",
)
async def google_callback(
    request: Request,
    runtime: RuntimeDep,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> Any:
    accepts_json = "application/json" in request.headers.get("accept", "")
    if error:
        if accepts_json:
            return JSONResponse({"status": "error", "detail": error}, status_code=400)
        return HTMLResponse(_oauth_page("Google authorization failed", error), status_code=400)
    try:
        await asyncio.to_thread(runtime.oauth.exchange_code, code or "", state or "")
    except OAuthNotConfiguredError as exc:
        if accepts_json:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)
        return HTMLResponse(_oauth_page("OAuth not configured", str(exc)), status_code=503)
    except OAuthError as exc:
        if accepts_json:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=400)
        return HTMLResponse(_oauth_page("Authorization failed", str(exc)), status_code=400)

    if accepts_json:
        return {"status": "connected", "provider": "google"}
    return HTMLResponse(
        _oauth_page(
            "Google account connected",
            "The refreshable token is stored locally. Set GOOGLE_SHEET_ID and call "
            "/auth/google/budget to verify the budget sheet.",
        )
    )


@auth_router.get(
    "/google/status",
    response_model=OAuthStatus,
    summary="Show whether Google is configured and authorized (no secrets returned)",
)
async def google_status(runtime: RuntimeDep) -> OAuthStatus:
    return OAuthStatus(provider="google", **runtime.oauth.status())


@auth_router.post(
    "/google/disconnect",
    summary="Delete the locally stored Google token",
)
async def google_disconnect(runtime: RuntimeDep) -> dict[str, str]:
    await asyncio.to_thread(runtime.oauth.disconnect)
    return {"status": "disconnected"}


@auth_router.get(
    "/google/budget",
    summary="Read the configured Google Sheet budget using the OAuth connector",
)
async def google_budget(runtime: RuntimeDep) -> dict[str, Any]:
    if runtime.sheets_connector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Sheets is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
                "GOOGLE_SHEET_ID and connect the account at /auth/google."
            ),
        )
    try:
        snapshot = await asyncio.to_thread(runtime.sheets_connector.fetch_budget)
        return snapshot.model_dump(mode="json")
    except OAuthNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OAuthNotAuthorizedError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (OAuthError, BudgetConnectorError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Health and service metadata
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(runtime: RuntimeDep) -> HealthResponse:
    db_status = "disabled"
    if runtime.db is not None:
        db_status = "ok" if runtime.db.ping() else "error"
    redis_status = "ok" if runtime.memory.ping() else "unavailable"
    docs = runtime.document_count
    degraded = (
        db_status == "error"
        or redis_status == "unavailable"
        or docs == 0
        or not any(
            (
                runtime.settings.mock_llm,
                runtime.settings.groq_api_key,
                runtime.settings.gemini_api_key,
            )
        )
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        app=runtime.settings.app_name,
        environment=runtime.settings.environment,
        database=db_status,
        redis=redis_status,
        vector_documents=docs,
        llm=runtime.settings.provider_status,
        langsmith=runtime.settings.langsmith_enabled,
    )


@router.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {
        "service": "ProcureFlow",
        "docs": "/docs",
        "health": "/health",
        "submit_request": "POST /api/v1/requests",
        "pending_approvals": "GET /api/v1/approvals",
        "google_oauth": "GET /auth/google",
    }


def _oauth_page(title: str, message: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:680px;margin:10vh auto;padding:24px;
background:#f6f7f9;color:#17202a}}main{{background:white;border-radius:12px;padding:32px;
box-shadow:0 8px 30px rgba(0,0,0,.08)}}code{{background:#eef1f4;padding:2px 5px;border-radius:4px}}
</style></head>
<body><main><h1>{html.escape(title)}</h1><p>{html.escape(message)}</p>
<p><a href="/docs">Open API docs</a> · <a href="/">Service home</a></p></main></body></html>"""
