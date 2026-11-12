"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Typed runtime configuration.

    Every value can be supplied through a ``.env`` file or an environment
    variable. Secrets intentionally have empty defaults so the application
    can boot in mock mode without credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "ProcureFlow"
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    app_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Storage
    database_url: str = "postgresql://procureflow:procureflow@localhost:5432/procureflow"
    redis_url: str = "redis://localhost:6379/0"
    checkpointer_backend: Literal["postgres", "memory"] = "postgres"

    # LLM providers
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "groq/openai/gpt-oss-120b"
    gemini_model: str = "gemini/gemini-3.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1600
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_cache_ttl_seconds: int = 3600
    mock_llm: bool = False

    # RAG
    embedding_backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_top_k: int = 4
    rag_candidate_k: int = 12
    rrf_k: int = 60
    auto_ingest: bool = True

    # Workflow
    confidence_threshold: float = 0.75
    approval_threshold_usd: float = 10_000.0
    max_graph_iterations: int = 16
    local_total_budget_usd: float = 120_000.0
    local_committed_usd: float = 42_000.0

    # Google OAuth and Sheets
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    google_sheet_id: str = ""
    google_sheet_range: str = "Budget!A1:C10"
    google_token_path: str = "./data/google_token.json"
    google_scopes: str = "https://www.googleapis.com/auth/spreadsheets.readonly"

    # LangSmith
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "procureflow-langgraph-agents"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # MCP
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8100
    mcp_transport: Literal["stdio", "streamable-http"] = "stdio"

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        return value

    @field_validator("approval_threshold_usd", "local_total_budget_usd")
    @classmethod
    def _validate_positive_amount(cls, value: float) -> float:
        if value < 0:
            raise ValueError("amount thresholds must be non-negative")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def _validate_embedding_dim(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("EMBEDDING_DIM must be positive")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def google_scopes_list(self) -> list[str]:
        return [scope.strip() for scope in self.google_scopes.split(",") if scope.strip()]

    @property
    def google_oauth_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def google_sheets_configured(self) -> bool:
        return bool(self.google_oauth_configured and self.google_sheet_id)

    @property
    def provider_status(self) -> dict[str, bool]:
        return {
            "groq": bool(self.groq_api_key),
            "gemini": bool(self.gemini_api_key),
            "mock": self.mock_llm,
        }

    def configure_langsmith(self) -> None:
        """Enable LangSmith tracing only when explicitly opted in.

        LangGraph/LangChain read tracing configuration from environment
        variables, so the settings are mirrored here at application startup.
        Secrets are never logged.
        """

        if not self.langsmith_enabled:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGSMITH_TRACING", None)
            return
        if not self.langsmith_api_key:
            logger.warning("LANGSMITH_ENABLED=true but LANGSMITH_API_KEY is empty; tracing is off")
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGSMITH_TRACING", None)
            return

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        logger.info("LangSmith tracing enabled for project %s", self.langsmith_project)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached for cheap FastAPI dependencies)."""

    return Settings()
