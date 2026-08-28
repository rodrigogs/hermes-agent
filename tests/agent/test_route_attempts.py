"""Contract tests for executor-side route attempt journaling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent.route_attempts as attempts


@pytest.fixture(autouse=True)
def _reset_attempt_state(monkeypatch, tmp_path):
    """Every test starts as a distinct worker process would."""
    attempts._AGENT_STATES.clear()
    monkeypatch.setattr(attempts, "_FLUSH_REGISTERED", False)
    monkeypatch.setenv("HERMES_ROUTE_ATTEMPTS_FILE", str(tmp_path / "attempts.jsonl"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_route")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9")


def _agent(model="primary", provider="p1"):
    return SimpleNamespace(model=model, provider=provider, session_id="s-route")


def _records(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_inert_without_file_gate(monkeypatch):
    monkeypatch.delenv("HERMES_ROUTE_ATTEMPTS_FILE")
    agent = _agent()
    attempts.bind_hop(agent)
    attempts.note_served(agent)
    attempts.flush_all()
    assert attempts._AGENT_STATES == {}


def test_served_attempt_is_full_contract_and_is_one_based(tmp_path, monkeypatch):
    ticks = iter((100.25, 100.75))
    monkeypatch.setattr(attempts.time, "time", lambda: next(ticks))
    agent = _agent("head", "rail-a")

    attempts.bind_hop(agent)
    attempts.note_served(agent)
    attempts.flush_all()

    assert _records(tmp_path) == [{
        "schema": "route-attempts/1", "task_id": "t_route", "run_id": 9,
        "session_id": "s-route", "n": 1, "model": "head", "provider": "rail-a",
        "started_at": 100.25, "duration_ms": 500, "outcome": "served",
    }]


def test_failed_then_fallback_served_keeps_hop_order_and_error(tmp_path, monkeypatch):
    ticks = iter((10.0, 10.3, 11.0, 11.8))
    monkeypatch.setattr(attempts.time, "time", lambda: next(ticks))
    agent = _agent("head", "rail-a")

    attempts.bind_hop(agent)
    attempts.close_hop(agent, "failed", error_code="rate_limit", error_message="quota exhausted")
    agent.model, agent.provider = "fallback", "rail-b"
    attempts.bind_hop(agent)
    attempts.note_served(agent)
    attempts.flush_all()

    first, second = _records(tmp_path)
    assert (first["n"], first["model"], first["provider"], first["outcome"]) == (1, "head", "rail-a", "failed")
    assert first["duration_ms"] == 300
    assert first["error"] == {"code": "rate_limit", "message": "quota exhausted"}
    assert (second["n"], second["model"], second["provider"], second["outcome"]) == (2, "fallback", "rail-b", "served")
    assert second["duration_ms"] == 800


def test_skipped_has_no_error_and_consumes_its_chain_ordinal(tmp_path, monkeypatch):
    ticks = iter((1.0, 1.5, 2.0, 3.0, 4.0))
    monkeypatch.setattr(attempts.time, "time", lambda: next(ticks))
    agent = _agent()

    attempts.bind_hop(agent)
    attempts.close_hop(agent, "failed", error_code="server_error")
    attempts.note_skip(agent, "rail-b", "unavailable")
    agent.model, agent.provider = "fallback", "rail-c"
    attempts.bind_hop(agent)
    attempts.note_served(agent)
    attempts.flush_all()

    failed, skipped, served = _records(tmp_path)
    assert [record["n"] for record in (failed, skipped, served)] == [1, 2, 3]
    assert skipped == {
        "schema": "route-attempts/1", "task_id": "t_route", "run_id": 9,
        "session_id": "s-route", "n": 2, "model": "unavailable", "provider": "rail-b",
        "started_at": 2.0, "duration_ms": 0, "outcome": "skipped",
    }
    assert "error" not in skipped


def test_failed_record_truncates_unbounded_provider_message(tmp_path, monkeypatch):
    ticks = iter((1.0, 1.1))
    monkeypatch.setattr(attempts.time, "time", lambda: next(ticks))
    agent = _agent()

    attempts.bind_hop(agent)
    attempts.close_hop(agent, "failed", error_code="x" * 100, error_message="m" * 500)

    record = _records(tmp_path)[0]
    assert record["error"] == {"code": "x" * 64, "message": "m" * 240}


def test_close_as_served_without_success_is_never_a_lie(tmp_path, monkeypatch):
    ticks = iter((1.0, 1.1))
    monkeypatch.setattr(attempts.time, "time", lambda: next(ticks))
    agent = _agent()

    attempts.bind_hop(agent)
    attempts.close_hop(agent, "served")

    record = _records(tmp_path)[0]
    assert record["outcome"] == "failed"
    assert record["error"]["code"] == "closed_without_success"
