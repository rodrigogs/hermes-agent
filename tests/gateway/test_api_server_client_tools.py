"""Client tool bridge — relayed tool catalogs for the API server.

Contract tests for the split-runtime channel: a chat request that declares a
``tools`` catalog gets those tools relayed to the client, which executes them
on its own host and posts the result back through
``/api/sessions/{id}/chat/tool-result``.

Design: docs/design/2026-08-31-client-tools-relaying.md (D1-D7).

The validation + bridge lifecycle tests are pure invariants (no aiohttp);
the handler-level tests drive the real aiohttp TestClient like
``tests/gateway/test_session_api.py`` does, with the agent faked the same
way that file's FakeAgent pattern does.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_client_tools import (
    ClientToolBridge,
    ClientToolsError,
    install_bridge,
    validate_client_tools,
)
from hermes_state import SessionDB


# ── Catalog validation (D1/D7) ────────────────────────────────────────


class TestCatalogValidation:
    def test_openai_wire_form_accepted(self):
        catalog = validate_client_tools([
            {
                "type": "function",
                "function": {
                    "name": "trama_echo",
                    "description": "Echo on the client host",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ])
        assert "trama_echo" in catalog
        assert catalog["trama_echo"]["type"] == "function"
        assert catalog["trama_echo"]["function"]["parameters"]["required"] == ["text"]

    def test_short_form_accepted(self):
        catalog = validate_client_tools([
            {"name": "trama_ping", "description": "d", "parameters": {"type": "object"}}
        ])
        assert catalog["trama_ping"]["function"]["name"] == "trama_ping"

    def test_absent_and_empty_catalogs_are_empty(self):
        assert validate_client_tools(None) == {}
        assert validate_client_tools([]) == {}

    def test_missing_name_rejected(self):
        with pytest.raises(ClientToolsError):
            validate_client_tools([{"description": "no name"}])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ClientToolsError):
            validate_client_tools([
                {"name": "dup", "parameters": {"type": "object"}},
                {"name": "dup", "parameters": {"type": "object"}},
            ])

    def test_non_object_parameters_rejected(self):
        with pytest.raises(ClientToolsError):
            validate_client_tools([{"name": "bad", "parameters": {"type": "string"}}])

    def test_non_list_tools_rejected(self):
        with pytest.raises(ClientToolsError):
            validate_client_tools({"name": "x"})

    def test_oversized_catalog_rejected(self):
        from gateway.platforms.api_server_client_tools import _MAX_CATALOG_TOOLS

        with pytest.raises(ClientToolsError):
            validate_client_tools(
                [
                    {"name": f"t_{i}", "parameters": {"type": "object"}}
                    for i in range(_MAX_CATALOG_TOOLS + 1)
                ]
            )


# ── Bridge lifecycle (D2/D5) ──────────────────────────────────────────


class TestBridgeDispatch:
    def test_dispatch_parks_until_resolve_delivers_output(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_slow", "parameters": {"type": "object"}}]),
            timeout=5.0,
        )
        results = {}

        def park():
            results["out"] = bridge.dispatch("trama_slow", {"x": 1}, tool_call_id="call_1")

        t = threading.Thread(target=park, daemon=True)
        t.start()
        # The dispatch parks; the emit hook fired synchronously before the park.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not bridge.pending:
            time.sleep(0.01)
        assert "call_1" in bridge.pending
        assert bridge.resolve("call_1", "{\"echo\": true}")
        t.join(timeout=2)
        assert not t.is_alive()
        assert json.loads(results["out"]) == {"echo": True}

    def test_dispatch_timeout_returns_structured_error(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_never", "parameters": {"type": "object"}}]),
            timeout=0.2,
        )
        out = json.loads(bridge.dispatch("trama_never", {}, tool_call_id="call_x"))
        assert out["code"] == "client_tools_timeout"
        assert "timed out" in out["error"]

    def test_close_wakes_parked_calls_fail_closed(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_wait", "parameters": {"type": "object"}}]),
            timeout=30.0,
        )
        holder = {}

        def park():
            holder["out"] = bridge.dispatch("trama_wait", {}, tool_call_id="call_c")

        t = threading.Thread(target=park, daemon=True)
        t.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "call_c" not in bridge.pending:
            time.sleep(0.01)
        bridge.close()
        t.join(timeout=2)
        assert not t.is_alive()
        out = json.loads(holder["out"])
        assert out["code"] == "client_tools_no_result"

    def test_resolve_twice_is_single_delivery(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "t", "parameters": {"type": "object"}}]),
            timeout=5.0,
        )
        bridge.pending["c1"] = __import__(
            "gateway.platforms.api_server_client_tools", fromlist=["_PendingCall"]
        )._PendingCall("c1", "t", {})
        assert bridge.resolve("c1", "one") is True
        assert bridge.resolve("c1", "two") is False

    def test_client_error_flag_shapes_error_result(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "t", "parameters": {"type": "object"}}]),
            timeout=5.0,
        )
        box = {}

        def park():
            box["out"] = bridge.dispatch("t", {}, tool_call_id="c9")

        t = threading.Thread(target=park, daemon=True)
        t.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "c9" not in bridge.pending:
            time.sleep(0.01)
        bridge.resolve("c9", "host exploded", is_error=True)
        t.join(timeout=2)
        out = json.loads(box["out"])
        assert out["code"] == "client_tools_client_error"
        assert out["error"] == "host exploded"

    def test_emit_failure_does_not_wedge_the_turn(self):
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "t", "parameters": {"type": "object"}}]),
            timeout=5.0,
        )

        def boom(call_id, name, args):
            raise RuntimeError("sink down")

        bridge.emit = boom
        out = json.loads(bridge.dispatch("t", {}, tool_call_id="c2"))
        assert out["code"] == "client_tools_emit_failed"


# ── Agent-side injection + interception (D3/D4) ───────────────────────


class _RecordingAgent:
    def __init__(self):
        self.tools = []
        self.valid_tool_names = set()


class TestBridgeInjection:
    def test_install_bridge_appends_schemas_and_names(self):
        agent = _RecordingAgent()
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_a", "parameters": {"type": "object"}}])
        )
        install_bridge(agent, bridge)
        assert "trama_a" in agent.valid_tool_names
        assert agent.tools[0]["function"]["name"] == "trama_a"

    def test_install_bridge_never_shadows_native_tools(self):
        agent = _RecordingAgent()
        native = {"type": "function", "function": {"name": "terminal"}}
        agent.tools.append(native)
        agent.valid_tool_names.add("terminal")
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "terminal", "parameters": {"type": "object"}}])
        )
        install_bridge(agent, bridge)
        assert [t for t in agent.tools if t["function"]["name"] == "terminal"] == [native]

    def test_handle_function_call_routes_catalog_tool_to_bridge(self):
        import model_tools

        agent = _RecordingAgent()
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_b", "parameters": {"type": "object"}}]),
            timeout=5.0,
        )
        install_bridge(agent, bridge)
        box = {}

        def deliver():
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not bridge.pending:
                time.sleep(0.01)
            (call_id, _), = bridge.pending.items()
            bridge.resolve(call_id, "relayed-output")

        t = threading.Thread(target=deliver, daemon=True)
        t.start()
        result = model_tools.handle_function_call(
            "trama_b", {"k": "v"}, "task-1", agent=agent
        )
        t.join(timeout=2)
        assert result == "relayed-output"

    def test_handle_function_call_unknown_name_in_bridge_session_hints_catalog(self):
        import model_tools

        agent = _RecordingAgent()
        bridge = ClientToolBridge(
            validate_client_tools([{"name": "trama_b", "parameters": {"type": "object"}}])
        )
        install_bridge(agent, bridge)
        result = model_tools.handle_function_call(
            "no_such_tool", {}, "task-1", agent=agent
        )
        assert "Unknown tool" in result
        assert "'tools' catalog" in result

    def test_agent_without_bridge_is_untouched(self):
        import model_tools

        result = model_tools.handle_function_call(
            "whatever_tool", {}, "task-1", agent=_RecordingAgent()
        )
        # No bridge → registry path → normal unknown-tool error, no catalog hint.
        assert "Unknown tool" in result
        assert "'tools' catalog" not in result


# ── HTTP surface ──────────────────────────────────────────────────────


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post(
        "/api/sessions/{session_id}/chat/tool-result",
        adapter._handle_session_tool_result,
    )
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_client_tools(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()
    assert data["features"]["session_client_tools"] is True
    assert data["endpoints"]["session_tool_result"]["path"] == (
        "/api/sessions/{session_id}/chat/tool-result"
    )


class _ToolCallingAgent:
    """FakeAgent whose loop calls the client tool once, then finishes."""

    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, session_id, bridge_holder, tool_name="trama_echo"):
        self.session_id = session_id
        self._bridge_holder = bridge_holder
        self._tool_name = tool_name

    def run_conversation(self, user_message, conversation_history, task_id):
        from gateway.platforms.api_server_client_tools import bridge_for

        bridge = bridge_for(self)
        self._bridge_holder["bridge"] = bridge
        result = bridge.dispatch(self._tool_name, {"text": "hi"}, tool_call_id="call_t1")
        self._bridge_holder["result"] = result
        return {
            "final_response": f"tool said: {result}",
            "session_id": self.session_id,
        }


@pytest.mark.asyncio
async def test_sync_chat_with_catalog_reaches_tool_result(adapter, session_db, monkeypatch):
    """Sync path: catalog accepted, bridge registered during the turn, the
    client's tool-result POST (issued concurrently) resolves the parked call,
    and the turn ends with the relayed output in final_response."""
    monkeypatch.setattr(APIServerAdapter, "_client_tools_timeout", lambda self: 10.0)
    session_id = session_db.create_session("bridge-sync", "api_server")
    holder = {}
    observed = {}

    def fake_create_agent(**kwargs):
        observed["session_id"] = kwargs.get("session_id")
        return _ToolCallingAgent(kwargs["session_id"], holder)

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        body = {
            "message": "use the relayed tool",
            "tools": [
                {
                    "name": "trama_echo",
                    "description": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                }
            ],
        }

        async def deliver():
            """The client side: wait for a pending call, then POST its result."""
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                bridge = adapter._session_client_tool_bridges.get(session_id)
                if bridge is not None and bridge.pending:
                    (call_id, _pending), = bridge.pending.items()
                    r = await cli.post(
                        f"/api/sessions/{session_id}/chat/tool-result",
                        json={
                            "tool_call_id": call_id,
                            "output": json.dumps({"echo": True}),
                        },
                    )
                    assert r.status == 200
                    assert (await r.json())["delivered"] is True
                    return
                await asyncio.sleep(0.02)
            raise AssertionError("no pending client tool call appeared during the turn")

        chat_task = asyncio.create_task(
            cli.post(f"/api/sessions/{session_id}/chat", json=body)
        )
        deliver_task = asyncio.create_task(deliver())
        resp = await chat_task
        await deliver_task
        assert resp.status == 200
        payload = await resp.json()

        # After the turn, the bridge is unregistered (fail-closed cleanup)…
        assert not adapter._session_client_tool_bridges
        # …so a late result gets the stable 409, never a silent drop.
        late = await cli.post(
            f"/api/sessions/{session_id}/chat/tool-result",
            json={"tool_call_id": "call_t1", "output": "x"},
        )
        assert late.status == 409
        assert (await late.json())["error"]["code"] == "client_tools_not_active"

    # The catalog reached the bridge the loop dispatched through (validated
    # at the HTTP edge, installed per-request in _run_agent), and the relayed
    # output re-entered the loop as the tool result the model saw.
    assert "trama_echo" in holder["bridge"].catalog
    assert holder["result"] == '{"echo": true}'
    assert "tool said:" in payload["message"]["content"]
    assert '{"echo": true}' in payload["message"]["content"]


@pytest.mark.asyncio
async def test_tool_result_409_when_no_channel(adapter, session_db):
    session_id = session_db.create_session("bridge-none", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat/tool-result",
            json={"tool_call_id": "nope", "output": "x"},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["error"]["code"] == "client_tools_not_active"


@pytest.mark.asyncio
async def test_invalid_catalog_is_a_stable_400(adapter, session_db):
    session_id = session_db.create_session("bridge-bad", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={
                "message": "hi",
                "tools": [{"name": ""}],
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["code"] == "invalid_client_tools"


@pytest.mark.asyncio
async def test_chat_without_tools_is_unchanged(adapter, session_db, monkeypatch):
    """Regression: a request without `tools` never builds a bridge."""
    observed = {}

    class _PlainAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.platforms.api_server_client_tools import bridge_for

            observed["bridge"] = bridge_for(self)
            return {"final_response": "ok", "session_id": self.session_id}

    def fake_create_agent(**kwargs):
        observed["client_tools"] = kwargs.get("client_tools")
        return _PlainAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    session_id = session_db.create_session("bridge-plain", "api_server")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "plain turn"},
        )
        assert resp.status == 200
        payload = await resp.json()
    assert observed["client_tools"] in (None, {})
    assert observed["bridge"] is None
    assert payload["message"]["content"] == "ok"
