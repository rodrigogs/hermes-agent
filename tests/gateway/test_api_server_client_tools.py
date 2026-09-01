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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_client_tools import (
    ClientToolBridge,
    ClientToolsChannelActive,
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


# ── P1 regression: the REAL sequential executor must reach the bridge ──
#
# A turn with a single tool call goes through
# ``_execute_tool_calls_sequential`` (run_agent.py dispatches ≤1 call
# there).  The bridge intercept in ``handle_function_call`` is agent-gated:
# if the executor's inline dispatch drops ``agent=``, the relayed call
# falls through to the process registry and dies as "Unknown tool" before
# ever parking on the bridge.  These tests drive the REAL executor — not a
# fake that calls ``bridge.dispatch`` directly — exactly like
# tests/run_agent/test_sequential_tool_timeout.py does.


def _p1_sequential_agent():
    """Minimal duck-typed agent for ``execute_tool_calls_sequential``.

    Mirrors every attribute the single-call verbose path touches, including
    the persistence tail (flush → guardrails → result-content unwrap) that
    runs after the tool result lands.
    """
    from types import SimpleNamespace

    agent = SimpleNamespace()
    agent.session_id = "seq-session"
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.log_prefix = ""
    agent.log_prefix_chars = 100
    agent.platform = "api"
    agent.suppress_status_output = True
    agent._print_fn = None
    agent._should_emit_quiet_tool_messages = lambda: False
    agent._should_start_quiet_spinner = lambda: False
    agent._vprint = lambda *a, **k: None
    agent._tool_worker_threads = set()
    agent._tool_worker_threads_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._executing_tools = False
    agent.valid_tool_names = {"trama_echo"}
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent._current_turn_id = ""
    agent._current_api_request_id = ""
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent._context_engine_tool_names = frozenset()
    agent._memory_manager = None
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent.context_compressor = None
    agent._subdirectory_hints = SimpleNamespace(
        check_tool_call=lambda *a, **k: ""
    )
    agent._flush_messages_to_session_db = lambda messages: True
    agent._touch_activity = lambda *a, **k: None
    agent._wrap_verbose = lambda prefix, text: f"{prefix}{text}"
    agent._tool_guardrails = __import__(
        "agent.tool_guardrails", fromlist=["ToolCallGuardrailController"]
    ).ToolCallGuardrailController()
    agent._apply_pending_steer_to_tool_results = lambda messages, n: None
    agent._stall_guards = False
    agent._append_guardrail_observation = (
        lambda name, args, result, **k: result
    )
    agent._tool_result_content_for_active_model = (
        lambda name, result: result
    )
    agent._record_file_mutation_result = lambda *a, **k: None
    return agent


def _p1_tool_call(call_id: str, name: str = "trama_echo"):
    from types import SimpleNamespace

    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments='{"text": "hi"}'),
    )


def test_sequential_executor_single_call_parks_on_bridge():
    """P1: one relayed call in a sequential turn reaches the client."""
    from agent.tool_executor import execute_tool_calls_sequential

    agent = _p1_sequential_agent()
    bridge = ClientToolBridge(
        validate_client_tools([
            {"name": "trama_echo", "parameters": {"type": "object"}}
        ]),
        timeout=10.0,
    )
    install_bridge(agent, bridge)

    def deliver():
        """The client side: resolve the pending relayed call."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not bridge.pending:
            time.sleep(0.01)
        assert bridge.pending, (
            "sequential executor never parked the relayed call on the "
            "bridge (agent= dropped between executor and dispatcher?)"
        )
        (call_id, _pending), = bridge.pending.items()
        bridge.resolve(call_id, '{"echo": true}')

    t = threading.Thread(target=deliver, daemon=True)
    t.start()
    messages = []
    assistant = SimpleNamespace(tool_calls=[_p1_tool_call("call_seq1")])
    execute_tool_calls_sequential(agent, assistant, messages, "task-1")
    t.join(timeout=5)

    assert not bridge.pending
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_seq1"
    assert messages[0]["content"] == '{"echo": true}'


def test_sequential_executor_unknown_name_in_bridge_session_hints_catalog():
    """P1: an unknown name in a bridge session gets the catalog hint (the
    agent-gated branch of handle_function_call), not a registry mismatch."""
    from agent.tool_executor import execute_tool_calls_sequential

    agent = _p1_sequential_agent()
    bridge = ClientToolBridge(
        validate_client_tools([
            {"name": "trama_echo", "parameters": {"type": "object"}}
        ]),
        timeout=10.0,
    )
    install_bridge(agent, bridge)

    messages = []
    assistant = SimpleNamespace(
        tool_calls=[_p1_tool_call("call_seq2", name="not_in_catalog")]
    )
    execute_tool_calls_sequential(agent, assistant, messages, "task-1")

    assert len(messages) == 1
    assert "Unknown tool" in messages[0]["content"]
    assert "'tools' catalog" in messages[0]["content"]


def test_sequential_executor_relayed_timeout_keeps_turn_alive():
    """P1 + D5: a relayed call whose client never answers returns the
    structured timeout error to the loop instead of hanging or raising."""
    from agent.tool_executor import execute_tool_calls_sequential

    agent = _p1_sequential_agent()
    bridge = ClientToolBridge(
        validate_client_tools([
            {"name": "trama_slow", "parameters": {"type": "object"}}
        ]),
        timeout=0.3,
    )
    install_bridge(agent, bridge)

    messages = []
    assistant = SimpleNamespace(tool_calls=[_p1_tool_call("call_seq3", name="trama_slow")])
    execute_tool_calls_sequential(agent, assistant, messages, "task-1")

    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload["code"] == "client_tools_timeout"


# ── P2 regression: a live channel is never silently overwritten ───────


def test_register_refuses_second_bridge_while_first_is_live(adapter, monkeypatch):
    """P2: registering over a live (open) channel raises
    ClientToolsChannelActive and the first turn's channel stays untouched."""
    adapter._session_client_tool_bridges.clear()
    live = ClientToolBridge(
        validate_client_tools([
            {"name": "trama_a", "parameters": {"type": "object"}}
        ])
    )
    adapter._session_client_tool_bridges["sess-live"] = live

    def _fake_create(**kwargs):
        raise ClientToolsChannelActive("second chat while first is live")

    monkeypatch.setattr(adapter, "_create_agent", _fake_create)
    monkeypatch.setattr(
        APIServerAdapter, "_client_tools_timeout", lambda self: 5.0
    )

    with pytest.raises(ClientToolsChannelActive):
        asyncio.run(adapter._run_agent(
            user_message="second turn",
            conversation_history=[],
            session_id="sess-live",
            client_tools=validate_client_tools([
                {"name": "trama_b", "parameters": {"type": "object"}}
            ]),
        ))

    # The FIRST turn's channel is untouched and still the registered one.
    assert adapter._session_client_tool_bridges.get("sess-live") is live
    assert not live.closed


def test_register_replaces_closed_leftover_entry(adapter, monkeypatch):
    """P2: a CLOSED leftover (crash path that skipped the finally) must not
    wedge the session forever — the new request registers over it."""
    adapter._session_client_tool_bridges.clear()
    stale = ClientToolBridge(
        validate_client_tools([
            {"name": "trama_old", "parameters": {"type": "object"}}
        ])
    )
    stale.close()
    adapter._session_client_tool_bridges["sess-stale"] = stale

    class _PlainAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id, holder):
            self.session_id = session_id
            self._holder = holder

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.platforms.api_server_client_tools import bridge_for

            self._holder["bridge"] = bridge_for(self)
            return {"final_response": "ok", "session_id": self.session_id}

    holder = {}

    def fake_create(**kwargs):
        return _PlainAgent(kwargs["session_id"], holder)

    monkeypatch.setattr(adapter, "_create_agent", fake_create)
    monkeypatch.setattr(
        APIServerAdapter, "_client_tools_timeout", lambda self: 5.0
    )

    result, _usage = asyncio.run(adapter._run_agent(
        user_message="turn after crash",
        conversation_history=[],
        session_id="sess-stale",
        client_tools=validate_client_tools([
            {"name": "trama_new", "parameters": {"type": "object"}}
        ]),
    ))
    assert result["final_response"] == "ok"
    assert holder["bridge"] is not None
    assert "trama_new" in holder["bridge"].catalog
    # The closed leftover was replaced by the new turn's bridge.
    assert adapter._session_client_tool_bridges.get("sess-stale") is not stale


@pytest.mark.asyncio
async def test_sync_second_chat_while_channel_live_is_409(adapter, session_db, monkeypatch):
    """P2 end-to-end: while the first /chat's channel is live, a second
    /chat with `tools` on the SAME session gets the stable 409 and the
    first turn's pending calls stay resolvable."""
    monkeypatch.setattr(
        APIServerAdapter, "_client_tools_timeout", lambda self: 10.0
    )
    session_id = session_db.create_session("bridge-guard", "api_server")
    holder = {}

    class _ParkedToolAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id, holder):
            self.session_id = session_id
            self._holder = holder

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.platforms.api_server_client_tools import bridge_for

            bridge = bridge_for(self)
            self._holder["bridge"] = bridge
            # Park like a real relayed call and hold the channel open until
            # the test observed the 409 and posted the tool-result.
            result = bridge.dispatch(
                "trama_echo", {"text": "hi"}, tool_call_id="call_first"
            )
            self._holder["result"] = result
            return {"final_response": f"tool said: {result}", "session_id": self.session_id}

    def fake_create_agent(**kwargs):
        return _ParkedToolAgent(kwargs["session_id"], holder)

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        tools_body = {
            "message": "first turn",
            "tools": [
                {"name": "trama_echo", "description": "echo",
                 "parameters": {"type": "object",
                                "properties": {"text": {"type": "string"}}}}
            ],
        }

        first_task = asyncio.create_task(
            cli.post(f"/api/sessions/{session_id}/chat", json=tools_body)
        )

        # Wait until the first turn's bridge is live with a pending call.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            bridge = adapter._session_client_tool_bridges.get(session_id)
            if bridge is not None and bridge.pending:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("first turn never parked a call")
        live_bridge = adapter._session_client_tool_bridges[session_id]

        second = await cli.post(
            f"/api/sessions/{session_id}/chat", json=tools_body
        )
        assert second.status == 409
        data = await second.json()
        assert data["error"]["code"] == "client_tools_channel_active"

        # The first channel is untouched: still the registered bridge, its
        # pending call still resolvable.
        assert (
            adapter._session_client_tool_bridges.get(session_id) is live_bridge
        )
        r = await cli.post(
            f"/api/sessions/{session_id}/chat/tool-result",
            json={"tool_call_id": "call_first", "output": '{"echo": true}'},
        )
        assert r.status == 200
        assert (await r.json())["delivered"] is True

        first_resp = await first_task
        assert first_resp.status == 200

    assert holder["result"] == '{"echo": true}'
    # Fail-closed cleanup ran for the first turn.
    assert not adapter._session_client_tool_bridges


@pytest.mark.asyncio
async def test_stream_second_chat_while_channel_live_emits_typed_error(
    adapter, monkeypatch
):
    """P2 (stream parity): a second /chat/stream with a `tools` catalog on a
    session whose channel is live renders a TYPED SSE error event with the
    stable code — not the generic run failure — so the client can retry
    after the in-flight turn instead of orphaning its pending calls."""
    from unittest.mock import AsyncMock

    app = _create_session_app(adapter)
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    async with TestClient(TestServer(app)) as cli:
        with (
            patch.object(
                adapter,
                "_get_existing_session_or_404",
                return_value=({"id": "sess-live-stream"}, None),
            ),
            patch.object(
                adapter, "_conversation_history_for_session", return_value=[]
            ),
            patch.object(
                adapter, "_run_agent", new_callable=AsyncMock
            ) as mock_run,
        ):
            # The register guard lives inside _run_agent (the live-channel
            # check runs where the bridge is created); raising here models
            # exactly that outcome reaching the stream edge.
            mock_run.side_effect = ClientToolsChannelActive(
                "session sess-live-stream already has an active client-tools "
                "channel from a concurrent chat request"
            )
            resp = await cli.post(
                "/api/sessions/sess-live-stream/chat/stream",
                json={
                    "message": "second turn",
                    "tools": [
                        {
                            "name": "trama_b",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
            assert resp.status == 200
            body = await resp.text()

        assert "event: error" in body
        assert "client_tools_channel_active" in body
        # The guard fired INSIDE the run (edge rendered it), not at the
        # catalog-validation boundary.
        assert mock_run.await_count == 1
