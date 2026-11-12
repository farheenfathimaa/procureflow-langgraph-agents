"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth_router, router
from app.config import Settings, get_settings
from app.runtime import AppRuntime, build_runtime

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    runtime: AppRuntime | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Supplying a pre-built runtime keeps tests fast and lets them use the
    in-memory checkpointer without touching Postgres or Redis.
    """

    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, resolved_settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        active_runtime = runtime
        if active_runtime is None:
            logger.info("Building ProcureFlow runtime")
            active_runtime = build_runtime(resolved_settings)
        app.state.runtime = active_runtime
        logger.info(
            "ProcureFlow ready: checkpointer=%s documents=%s mock_llm=%s",
            resolved_settings.checkpointer_backend,
            active_runtime.document_count,
            resolved_settings.mock_llm,
        )
        try:
            yield
        finally:
            logger.info("Shutting down ProcureFlow runtime")
            active_runtime.close()

    application = FastAPI(
        title="ProcureFlow — Multi-Agent Procurement Assistant",
        description=(
            "LangGraph orchestrator with policy, supplier and budget specialists, "
            "hybrid RAG, OAuth 2.0 connectors and human-in-the-loop approval."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    application.include_router(auth_router)
    return application


app = create_app()
