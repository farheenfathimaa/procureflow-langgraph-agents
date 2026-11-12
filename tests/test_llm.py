"""LLM client parsing, retry and fallback tests (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import LLMClient, LLMUnavailableError
from app.schemas import PolicyDraft


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "mock_llm": False,
        "groq_api_key": "test-groq",
        "gemini_api_key": "test-gemini",
        "llm_max_retries": 1,
        "database_url": "",
        "redis_url": "",
    }
    values.update(overrides)
    return Settings(**values)


def _mock_data() -> dict:
    return {
        "compliant": True,
        "requires_approval": False,
        "violations": [],
        "applicable_rules": [],
        "rationale": "ok",
        "confidence": 0.8,
        "citation_ids": ["POL-001"],
    }


def test_mock_mode_never_calls_litellm(monkeypatch):
    settings = _settings(mock_llm=True)

    def explode(**kwargs):
        raise AssertionError("network should not be called in mock mode")

    import litellm

    monkeypatch.setattr(litellm, "completion", explode)
    result = LLMClient(settings).generate_structured(
        PolicyDraft, "system", "user", mock_data=_mock_data()
    )
    assert result.compliant is True


def test_json_fences_are_parsed():
    assert LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert LLMClient._parse_json('Here is JSON: {"a": 2} trailing') == {"a": 2}
    with pytest.raises(ValueError):
        LLMClient._parse_json("no json")


def test_primary_failure_falls_back_to_gemini(monkeypatch):
    calls: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"].startswith("groq/"):
            raise RuntimeError("primary down")
        return _response(
            '{"compliant": true, "requires_approval": false, "violations": [], '
            '"applicable_rules": [], "rationale": "fallback", "confidence": 0.9, '
            '"citation_ids": ["POL-001"]}'
        )

    import litellm

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda attempt: None))
    result = LLMClient(_settings()).generate_structured(
        PolicyDraft, "system", "user", mock_data=_mock_data()
    )
    assert result.rationale == "fallback"
    assert calls.count("groq/openai/gpt-oss-120b") == 2
    assert calls.count("gemini/gemini-3.5-flash") == 1


def test_all_providers_failing_raises_bounded_error(monkeypatch):
    calls: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        raise RuntimeError("down")

    import litellm

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda attempt: None))
    with pytest.raises(LLMUnavailableError):
        LLMClient(_settings()).generate_structured(
            PolicyDraft, "system", "user", mock_data=_mock_data()
        )
    # 2 providers x (1 retry + 1 initial) = 4, never unbounded.
    assert len(calls) == 4
