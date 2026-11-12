"""Deterministic input guardrails for prompt injection and unsafe requests."""

from __future__ import annotations

import re
import unicodedata

from app.schemas import GuardrailResult

# Deliberately simple and explainable. These patterns catch the common
# "ignore previous instructions" family without trying to be a classifier.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|system)\b"
            r".{0,30}\b(instruction|prompt|rule|message)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|dump|leak)\b.{0,40}\b(system prompt|hidden prompt|"
            r"initial instruction|developer message)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(you are now|act as|pretend to be|from now on)\b.{0,40}\b(dan|jailbreak|"
            r"unrestricted|admin|developer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "policy_bypass",
        re.compile(
            r"\b(bypass|skip|disable|circumvent)\b.{0,30}\b(policy|guardrail|approval|"
            r"budget|security check)",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_request",
        re.compile(
            r"\b(print|show|reveal|send|leak)\b.{0,30}\b(api key|secret|password|token|"
            r"credential)",
            re.IGNORECASE,
        ),
    ),
]

_MAX_REQUEST_LENGTH = 4000


def inspect_input(text: str) -> GuardrailResult:
    """Check raw user input before any LLM call.

    Unicode is normalised first so zero-width characters cannot trivially
    split a blocked phrase.
    """

    normalised = unicodedata.normalize("NFKC", text or "")
    normalised = re.sub(r"[\u200b-\u200d\ufeff]", "", normalised)

    if not normalised.strip():
        return GuardrailResult(allowed=False, reason="Request is empty.")
    if len(normalised) > _MAX_REQUEST_LENGTH:
        return GuardrailResult(
            allowed=False,
            reason=f"Request exceeds the {_MAX_REQUEST_LENGTH}-character limit.",
        )

    for name, pattern in _INJECTION_PATTERNS:
        match = pattern.search(normalised)
        if match:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "Potential prompt-injection attempt detected. Submit a plain purchase "
                    "request; instructions in the request are not treated as system commands."
                ),
                matched_pattern=name,
            )

    return GuardrailResult(allowed=True)
