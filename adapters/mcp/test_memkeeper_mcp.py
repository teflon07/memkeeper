"""Tests for the memkeeper MCP adapter.

These verify that each tool builds the JSON payload / command the engine
expects. `_run_memkeeper` is monkeypatched, so no store, binary, model, or
network is needed — the tests guard against silent drift when the engine's
request schema changes (e.g. a renamed field) by pinning what each tool sends.
"""

from __future__ import annotations

import json
import importlib

import pytest

mod = importlib.import_module("memkeeper_mcp")


@pytest.fixture
def calls(monkeypatch):
    """Capture every _run_memkeeper invocation; return a benign envelope."""
    recorded: list[dict] = []

    def fake_run(command, payload=None, extra_args=None, env_overrides=None):
        recorded.append({"command": command, "payload": payload, "extra_args": extra_args})
        return json.dumps(
            {"ok": True, "result": {"results": [], "memory": {}, "pack": {"memory_ids": []}}}
        )

    monkeypatch.setattr(mod, "_run_memkeeper", fake_run)
    return recorded


def _last(calls, command):
    """Return the payload of the most recent call to `command`."""
    for entry in reversed(calls):
        if entry["command"] == command:
            return entry["payload"]
    raise AssertionError(f"no call to {command}; saw {[c['command'] for c in calls]}")


def test_remember_sends_mode_and_source_type(calls):
    mod.remember("decision: use rustls", mode="append", source_type="explicit-user")
    p = _last(calls, "remember")
    assert p["content"] == "decision: use rustls"
    assert p["mode"] == "append"
    assert p["derive_keys"] is True
    assert p["source"]["source_type"] == "explicit-user"


def test_remember_defaults(calls):
    mod.remember("fact: tabs over spaces")
    p = _last(calls, "remember")
    assert p["mode"] == "auto"
    assert p["source"]["source_type"] == "assistant-inference"
    assert "sensitivity" not in p["source"]


def test_remember_sensitivity_included_when_set(calls):
    mod.remember("fact: secret-ish", sensitivity="sensitive")
    assert _last(calls, "remember")["source"]["sensitivity"] == "sensitive"


def test_candidate_submit(calls):
    mod.candidate_submit("fact: maybe true", rationale="plausible", kind="fact")
    p = _last(calls, "candidate-submit")
    assert p["content"] == "fact: maybe true"
    assert p["source_type"] == "assistant-inference"
    assert p["rationale"] == "plausible"
    assert p["kind"] == "fact"


def test_candidate_list(calls):
    mod.candidate_list(status="pending", limit=25)
    p = _last(calls, "candidate-list")
    assert p["status"] == "pending"
    assert p["limit"] == 25


def test_pack(calls):
    mod.pack(["roadmap", "deploy"], title="ctx", max_memories=5, space="work")
    p = _last(calls, "pack")
    assert p["queries"] == ["roadmap", "deploy"]
    assert p["title"] == "ctx"
    assert p["max_memories"] == 5
    assert p["filters"]["spaces"] == ["work"]


def test_search_builds_filters(calls):
    mod.search("query text", space="work", tags=["t1"], entity_key="ek")
    p = _last(calls, "search")
    assert p["query"] == "query text"
    assert p["filters"]["spaces"] == ["work"]
    assert p["filters"]["tags"] == ["t1"]
    assert p["filters"]["entity_keys"] == ["ek"]


def test_forget(calls):
    mod.forget("mem_123", reason="duplicate")
    p = _last(calls, "forget")
    assert p["id"] == "mem_123"
    assert p["reason"] == "duplicate"


def test_expected_tool_surface_exists():
    """Smoke: every public tool is present and callable."""
    expected = [
        "remember", "search", "get", "memory_list", "pack",
        "candidate_submit", "candidate_list", "forget", "verify",
        "entity_search", "graph_neighbors", "graph_context",
        "entity_upsert", "relationship_upsert", "stats", "dream_graph",
    ]
    for name in expected:
        assert callable(getattr(mod, name, None)), f"missing tool: {name}"
