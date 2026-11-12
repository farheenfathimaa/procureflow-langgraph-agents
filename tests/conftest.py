"""Shared pytest fixtures. External services and LLMs are never called."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runtime import AppRuntime, build_runtime


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        mock_llm=True,
        checkpointer_backend="memory",
        database_url="",
        redis_url="",
        embedding_backend="hash",
        auto_ingest=True,
        reranker_enabled=False,
        approval_threshold_usd=10_000,
        confidence_threshold=0.75,
        max_graph_iterations=16,
        google_client_id="",
        google_client_secret="",
        google_sheet_id="",
        google_token_path=str(tmp_path / "google_token.json"),
        langsmith_enabled=False,
    )


@pytest.fixture()
def runtime(settings: Settings) -> AppRuntime:
    instance = build_runtime(settings)
    yield instance
    instance.close()


@pytest.fixture()
def client(settings: Settings, runtime: AppRuntime):
    app = create_app(settings=settings, runtime=runtime)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def high_amount_state() -> dict:
    return {
        "parsed_request": {
            "item": "laptops",
            "quantity": 20,
            "total_amount_usd": 30000.0,
            "currency": "USD",
            "category": "laptops",
            "specifications": [],
            "confidence": 0.9,
        },
        "orchestrator": {"confidence": 0.9},
        "errors": [],
    }
