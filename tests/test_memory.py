"""MemoryStore fallback behaviour (no Redis required)."""

from __future__ import annotations

from app.memory import MemoryStore


def test_json_round_trip_and_delete():
    memory = MemoryStore("")
    memory.set_json("key", {"value": 1, "items": [1, 2]})

    assert memory.get_json("key") == {"value": 1, "items": [1, 2]}
    memory.delete("key")
    assert memory.get_json("key") is None


def test_missing_keys_return_none():
    memory = MemoryStore("")
    assert memory.get_json("absent") is None
    assert memory.get_request("absent") is None
    assert memory.get_cached_llm_response("absent") is None


def test_pending_approvals_are_a_set():
    memory = MemoryStore("")
    memory.add_pending_approval("thread-b")
    memory.add_pending_approval("thread-a")
    memory.add_pending_approval("thread-a")  # idempotent

    assert memory.pending_approvals() == ["thread-a", "thread-b"]

    memory.remove_pending_approval("thread-a")
    assert memory.pending_approvals() == ["thread-b"]
    memory.remove_pending_approval("thread-a")
    assert memory.pending_approvals() == ["thread-b"]


def test_request_and_llm_cache_namespaces_are_separate():
    memory = MemoryStore("")
    memory.remember_request("thread", {"request": "laptops"})
    memory.cache_llm_response("abc", {"answer": "cached"}, ttl=60)

    assert memory.get_request("thread") == {"request": "laptops"}
    assert memory.get_cached_llm_response("abc") == {"answer": "cached"}


def test_fallback_reports_unavailable():
    memory = MemoryStore("")
    assert memory.available is False
    assert memory.ping() is False
