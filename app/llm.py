"""LiteLLM wrapper: retries, backoff, provider fallback and response caching."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.memory import MemoryStore

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised after every configured provider has failed."""


class LLMClient:
    """Structured-output LLM client with a mock mode for offline tests.

    The order is deliberately fixed: Groq first (fast free tier), Gemini as a
    fallback. Each provider gets bounded retries with exponential backoff, so
    a failing provider can never create an infinite loop.
    """

    def __init__(self, settings: Settings, memory: MemoryStore | None = None) -> None:
        self.settings = settings
        self.memory = memory
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.drop_params = True
        except Exception:  # pragma: no cover - import error is surfaced on first call
            logger.exception("LiteLLM import failed")

    @property
    def providers(self) -> list[tuple[str, str]]:
        providers: list[tuple[str, str]] = []
        if self.settings.groq_api_key:
            providers.append(("groq", self.settings.groq_model))
        if self.settings.gemini_api_key:
            providers.append(("gemini", self.settings.gemini_model))
        return providers

    def generate_structured(
        self,
        schema: type[ModelT],
        system_prompt: str,
        user_prompt: str,
        *,
        mock_data: dict[str, Any] | Callable[[], dict[str, Any]],
        temperature: float | None = None,
    ) -> ModelT:
        """Generate and validate a Pydantic object.

        ``mock_data`` is required so the mock mode is deterministic and tests
        never accidentally reach the network.
        """

        if self.settings.mock_llm:
            payload = mock_data() if callable(mock_data) else mock_data
            return schema.model_validate(payload)

        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        full_system = f"{system_prompt}\n\nJSON SCHEMA:\n{schema_json}"
        cache_key = self._cache_key(schema.__name__, full_system, user_prompt)
        if self.memory:
            cached = self.memory.get_cached_llm_response(cache_key)
            if cached is not None:
                try:
                    return schema.model_validate(cached)
                except ValidationError:
                    logger.warning("Ignoring stale cache entry that no longer validates")

        if not self.providers:
            raise LLMUnavailableError(
                "No LLM provider is configured. Set GROQ_API_KEY and/or GEMINI_API_KEY, "
                "or enable MOCK_LLM=true."
            )

        last_error: Exception | None = None
        for provider, model in self.providers:
            api_key = (
                self.settings.groq_api_key if provider == "groq" else self.settings.gemini_api_key
            )
            attempts = self.settings.llm_max_retries + 1
            for attempt in range(attempts):
                try:
                    payload = self._complete_json(
                        model=model,
                        api_key=api_key,
                        system_prompt=full_system,
                        user_prompt=user_prompt,
                        temperature=self.settings.llm_temperature
                        if temperature is None
                        else temperature,
                    )
                    validated = schema.model_validate(payload)
                    if self.memory:
                        self.memory.cache_llm_response(
                            cache_key,
                            validated.model_dump(mode="json"),
                            ttl=self.settings.llm_cache_ttl_seconds,
                        )
                    return validated
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "LLM %s attempt %s/%s failed: %s",
                        provider,
                        attempt + 1,
                        attempts,
                        type(exc).__name__,
                    )
                    if attempt + 1 < attempts:
                        self._backoff(attempt)
            logger.warning("Provider %s exhausted; trying the next provider", provider)

        raise LLMUnavailableError(
            "All LLM providers failed. Check provider status, API keys and quotas."
        ) from last_error

    def _complete_json(
        self,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, Any]:
        import litellm

        response = litellm.completion(
            model=model,
            api_key=api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=self.settings.llm_max_tokens,
            timeout=self.settings.llm_timeout_seconds,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned an empty response")
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed

    def _cache_key(self, schema_name: str, system_prompt: str, user_prompt: str) -> str:
        digest = hashlib.sha256()
        for part in (
            schema_name,
            system_prompt,
            user_prompt,
            self.settings.groq_model,
            self.settings.gemini_model,
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def _backoff(attempt: int) -> None:
        delay = min(2**attempt, 8) + random.uniform(0, 0.25)
        time.sleep(delay)
