"""Configuration validation and LangSmith opt-in behaviour."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_provider_and_google_flags():
    settings = Settings(
        _env_file=None,
        groq_api_key="groq-key",
        gemini_api_key="",
        mock_llm=False,
        google_client_id="client",
        google_client_secret="secret",
        google_sheet_id="",
        database_url="",
        redis_url="",
    )

    assert settings.provider_status == {"groq": True, "gemini": False, "mock": False}
    assert settings.google_oauth_configured is True
    assert settings.google_sheets_configured is False
    assert settings.google_sheets_configured != (
        settings.model_copy(update={"google_sheet_id": "sheet"}).google_sheets_configured
    )
    assert settings.google_scopes_list == ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def test_cors_origins_are_trimmed():
    settings = Settings(
        _env_file=None,
        cors_origins=" http://a.test , http://b.test ,,",
        database_url="",
        redis_url="",
    )
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence_threshold": 1.5},
        {"approval_threshold_usd": -1},
        {"embedding_dim": 0},
    ],
)
def test_invalid_thresholds_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="", redis_url="", **overrides)


def test_langsmith_disabled_clears_tracing(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    settings = Settings(_env_file=None, langsmith_enabled=False, database_url="", redis_url="")

    settings.configure_langsmith()
    assert "LANGCHAIN_TRACING_V2" not in os.environ
    assert "LANGSMITH_TRACING" not in os.environ


def test_langsmith_enabled_requires_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    settings = Settings(
        _env_file=None,
        langsmith_enabled=True,
        langsmith_api_key="",
        database_url="",
        redis_url="",
    )

    settings.configure_langsmith()
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_langsmith_enabled_sets_environment(monkeypatch):
    settings = Settings(
        _env_file=None,
        langsmith_enabled=True,
        langsmith_api_key="ls-key",
        langsmith_project="test-project",
        langsmith_endpoint="https://example.test",
        database_url="",
        redis_url="",
    )

    settings.configure_langsmith()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://example.test"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
