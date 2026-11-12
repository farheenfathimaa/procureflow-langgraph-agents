"""Redis-backed short-term memory with a graceful in-process fallback."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

PENDING_APPROVALS_KEY = "procureflow:pending_approvals"


class MemoryStore:
    """Small Redis cache for workflow context, approvals and LLM responses.

    Redis is optional at runtime: if it is unavailable, the same API is
    served from a process-local dictionary so the demo still works. The
    Postgres checkpointer remains the durable source of truth.
    """

    def __init__(self, redis_url: str = "", prefix: str = "procureflow") -> None:
        self.prefix = prefix
        self._redis: Any | None = None
        self._fallback: dict[str, tuple[float | None, str]] = {}
        self._lock = threading.Lock()
        if redis_url:
            try:
                import redis

                client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                client.ping()
                self._redis = client
                logger.info("Connected to Redis")
            except Exception as exc:
                logger.warning("Redis unavailable (%s); using in-process memory", exc)

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def ping(self) -> bool:
        if not self._redis:
            return False
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        if self._redis:
            try:
                self._redis.set(self._key(key), payload, ex=ttl_seconds)
                return
            except Exception:
                logger.exception("Redis set failed; using fallback")
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else None
        with self._lock:
            self._fallback[self._key(key)] = (expires_at, payload)

    def get_json(self, key: str) -> Any | None:
        if self._redis:
            try:
                raw = self._redis.get(self._key(key))
                return json.loads(raw) if raw else None
            except Exception:
                logger.exception("Redis get failed; using fallback")
        with self._lock:
            item = self._fallback.get(self._key(key))
            if not item:
                return None
            expires_at, payload = item
            if expires_at is not None and time.monotonic() > expires_at:
                self._fallback.pop(self._key(key), None)
                return None
            return json.loads(payload)

    def delete(self, key: str) -> None:
        if self._redis:
            try:
                self._redis.delete(self._key(key))
            except Exception:
                logger.exception("Redis delete failed")
        with self._lock:
            self._fallback.pop(self._key(key), None)

    def add_pending_approval(self, thread_id: str) -> None:
        if self._redis:
            try:
                self._redis.sadd(self._key(PENDING_APPROVALS_KEY), thread_id)
                return
            except Exception:
                logger.exception("Redis sadd failed; using fallback")
        with self._lock:
            self._fallback.setdefault(self._key(PENDING_APPROVALS_KEY), (None, "[]"))
            expires_at, payload = self._fallback[self._key(PENDING_APPROVALS_KEY)]
            values = set(json.loads(payload))
            values.add(thread_id)
            self._fallback[self._key(PENDING_APPROVALS_KEY)] = (
                expires_at,
                json.dumps(sorted(values)),
            )

    def remove_pending_approval(self, thread_id: str) -> None:
        if self._redis:
            try:
                self._redis.srem(self._key(PENDING_APPROVALS_KEY), thread_id)
                return
            except Exception:
                logger.exception("Redis srem failed; using fallback")
        with self._lock:
            item = self._fallback.get(self._key(PENDING_APPROVALS_KEY))
            if not item:
                return
            expires_at, payload = item
            values = set(json.loads(payload))
            values.discard(thread_id)
            self._fallback[self._key(PENDING_APPROVALS_KEY)] = (
                expires_at,
                json.dumps(sorted(values)),
            )

    def pending_approvals(self) -> list[str]:
        if self._redis:
            try:
                return sorted(self._redis.smembers(self._key(PENDING_APPROVALS_KEY)))
            except Exception:
                logger.exception("Redis smembers failed; using fallback")
        with self._lock:
            item = self._fallback.get(self._key(PENDING_APPROVALS_KEY))
            if not item:
                return []
            return sorted(json.loads(item[1]))

    def remember_request(self, thread_id: str, payload: dict[str, Any], ttl: int = 86_400) -> None:
        self.set_json(f"request:{thread_id}", payload, ttl_seconds=ttl)

    def get_request(self, thread_id: str) -> dict[str, Any] | None:
        return self.get_json(f"request:{thread_id}")

    def cache_llm_response(self, key: str, payload: dict[str, Any], ttl: int) -> None:
        self.set_json(f"llm:{key}", payload, ttl_seconds=ttl)

    def get_cached_llm_response(self, key: str) -> dict[str, Any] | None:
        return self.get_json(f"llm:{key}")

    def close(self) -> None:
        if self._redis:
            try:
                self._redis.close()
            except Exception:  # pragma: no cover - defensive shutdown path
                logger.exception("Error closing Redis client")
