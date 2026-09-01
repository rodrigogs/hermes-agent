"""Client tool bridge — a per-request tool catalog the CLIENT executes.

A backend that runs its own model loop (Trama) declares a tool catalog in the
chat request body; the server-side AIAgent may then call those tools, and the
call crosses back to the client as an HTTP/SSE payload.  The client executes
the tool on ITS host and posts the result back; the waiting agent thread is
released and the result re-enters the conversation loop as a normal tool
result.

Design contract: docs/design/2026-08-31-client-tools-relaying.md (D1-D7).

The mechanism mirrors the /v1/runs approval channel
(``gateway/platforms/api_server_runs.py`` + ``tools/approval.py``): the agent
thread parks on a ``threading.Event``; the client resolves the pending call
over HTTP; unregistering in the turn's ``finally`` wakes every parked thread
(fail-closed, no hang).

The bridge intercepts at ``handle_function_call``/``invoke_tool`` — BELOW the
conversation loop and the tool executor — so a relayed tool flows through the
same funnel as a native tool: pre/post tool-call hooks, guardrails, middleware
and post_tool_call observability all fire on the real tool name.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


# A relayed catalog is declarative data, not code: bound its size so a
# hostile/broken client cannot hand the model a multi-megabyte schema payload.
_MAX_CATALOG_TOOLS = 64
_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 2000
_MAX_SCHEMA_JSON_LEN = 16000
_MAX_RESULT_OUTPUT_LEN = 1_000_000

# How long a parked tool call waits for the client's result before the model
# receives a timeout error and can self-correct inside its own loop.  The
# chat handlers override this from config
# (``api_server.client_tools_timeout_seconds``).
DEFAULT_RESULT_TIMEOUT_SECONDS = 120.0


class ClientToolsError(ValueError):
    """Raised for an invalid client tool catalog (maps to HTTP 400)."""


class ClientToolsChannelActive(RuntimeError):
    """A live client-tools channel already exists for this session (409).

    Raised by the chat path when a second ``/chat`` request with a
    ``tools`` catalog arrives while a previous request's bridge is still
    registered and not closed — registering over it would orphan the first
    turn's pending calls until their timeout, so the new turn is rejected
    instead (code ``client_tools_channel_active``).
    """


def _reject(name: str, detail: str) -> None:
    raise ClientToolsError(f"client tool {name!r}: {detail}")


def _validate_one(raw: Any, seen: set) -> Tuple[str, Dict[str, Any]]:
    """Validate one catalog entry; returns the (name, schema) pair.

    Accepts both the OpenAI wire form
    ``{"type": "function", "function": {name, description, parameters}}``
    and the short form ``{"name", "description", "parameters"}``.
    """
    if not isinstance(raw, dict):
        raise ClientToolsError("each entry of 'tools' must be a JSON object")
    raw_fn = raw.get("function")
    fn = raw_fn if isinstance(raw_fn, dict) else raw
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ClientToolsError("each client tool needs a non-empty string 'name'")
    name = name.strip()
    if len(name) > _MAX_NAME_LEN:
        _reject(name, f"name longer than {_MAX_NAME_LEN} chars")
    if name in seen:
        _reject(name, "duplicate name in catalog")
    seen.add(name)
    raw_description = fn.get("description", "")
    description = raw_description if isinstance(raw_description, str) else ""
    if isinstance(raw_description, str) and len(description) > _MAX_DESCRIPTION_LEN:
        _reject(name, f"'description' longer than {_MAX_DESCRIPTION_LEN} chars")
    parameters = fn.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        _reject(name, "'parameters' must be a JSON Schema object")
    if parameters.get("type") not in (None, "object"):
        _reject(name, "'parameters.type' must be 'object'")
    try:
        schema_json = json.dumps(parameters, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        _reject(name, f"'parameters' is not JSON-serializable: {exc}")
        raise  # unreachable; narrows schema_json for the type checker
    if len(schema_json) > _MAX_SCHEMA_JSON_LEN:
        _reject(
            name,
            f"'parameters' larger than {_MAX_SCHEMA_JSON_LEN} chars",
        )
    return name, {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def validate_client_tools(raw_tools: Any) -> Dict[str, Dict[str, Any]]:
    """Validate a client tool catalog.

    Returns ``{tool_name: {"type": "function", "function": {...}}}``.

    Raises :class:`ClientToolsError` on any malformed entry — the chat
    handler maps that to a stable HTTP 400.
    """
    if raw_tools is None:
        return {}
    if not isinstance(raw_tools, list):
        raise ClientToolsError("'tools' must be a list")
    if len(raw_tools) > _MAX_CATALOG_TOOLS:
        raise ClientToolsError(
            f"'tools' accepts at most {_MAX_CATALOG_TOOLS} entries"
        )
    catalog: Dict[str, Dict[str, Any]] = {}
    seen: set = set()
    for raw in raw_tools:
        name, schema = _validate_one(raw, seen)
        catalog[name] = schema
    return catalog


class _PendingCall:
    """One relayed tool call waiting for the client's result."""

    __slots__ = ("event", "tool_call_id", "name", "arguments", "output", "is_error", "expired")

    def __init__(self, tool_call_id: str, name: str, arguments: Dict[str, Any]):
        self.event = threading.Event()
        self.tool_call_id = tool_call_id
        self.name = name
        self.arguments = arguments
        self.output: Optional[str] = None
        self.is_error = False
        self.expired = False


class ClientToolBridge:
    """Per-request channel between the server agent loop and the client host.

    One instance lives for exactly one chat request: it carries the validated
    catalog, the pending calls, and the SSE sink used to emit
    ``client_tool.call`` events.  Registered while the turn runs; the
    ``finally`` around ``run_conversation`` unregisters it.
    """

    def __init__(
        self,
        catalog: Dict[str, Dict[str, Any]],
        *,
        timeout: float = DEFAULT_RESULT_TIMEOUT_SECONDS,
    ):
        self.catalog = catalog
        self.timeout = timeout
        self.lock = threading.Lock()
        # tool_call_id → pending call
        self.pending: Dict[str, _PendingCall] = {}
        # Set by the chat handler when the turn ends / client disconnects.
        self.closed = False
        # SSE/HTTP notification for "the agent asked the client to run X".
        # Signature: emit(tool_call_id, name, arguments_json) -> None.  Runs
        # in the agent thread and must schedule any loop work itself (same
        # contract as register_gateway_notify in tools/approval.py).
        self.emit: Optional[Any] = None

    # ── catalog surface ───────────────────────────────────────────────

    def has_tool(self, name: str) -> bool:
        return name in self.catalog

    def tool_names(self) -> List[str]:
        return sorted(self.catalog)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self.catalog[name] for name in sorted(self.catalog)]

    # ── dispatch surface (called from the tool funnel) ────────────────

    def dispatch(self, name: str, args: Dict[str, Any], tool_call_id: str = "") -> str:
        """Execute *name* by handing it to the client and waiting.

        Never raises: returns a JSON string, errors included (the funnel's
        contract — ``registry.dispatch`` behaves the same way).  The wait is
        bounded (``self.timeout``) and released immediately when the bridge
        is closed (turn end / disconnect / run stop).
        """
        call_id = tool_call_id or f"ct_{uuid.uuid4().hex}"
        pending = _PendingCall(call_id, name, args if isinstance(args, dict) else {})
        with self.lock:
            if self.closed:
                return json.dumps(
                    {
                        "error": "client tools channel is closed",
                        "code": "client_tools_closed",
                        "tool": name,
                    },
                    ensure_ascii=False,
                )
            self.pending[call_id] = pending
        try:
            if self.emit is not None:
                try:
                    self.emit(call_id, name, json.dumps(args, ensure_ascii=False))
                except Exception:
                    # A broken sink must not wedge the turn: resolve the
                    # call as an error the model can react to.
                    return json.dumps(
                        {
                            "error": "client tools channel emit failed",
                            "code": "client_tools_emit_failed",
                            "tool": name,
                        },
                        ensure_ascii=False,
                    )
            # Park exactly like the approval flow's _ApprovalEntry.event.
            if not pending.event.wait(timeout=self.timeout):
                with self.lock:
                    if not pending.event.is_set():
                        pending.expired = True
                        self.pending.pop(call_id, None)
                return json.dumps(
                    {
                        "error": (
                            f"client tool '{name}' timed out after "
                            f"{int(self.timeout)}s without a result"
                        ),
                        "code": "client_tools_timeout",
                        "tool": name,
                        "tool_call_id": call_id,
                    },
                    ensure_ascii=False,
                )
        finally:
            with self.lock:
                self.pending.pop(call_id, None)
        if pending.expired or pending.output is None:
            return json.dumps(
                {
                    "error": "client tool result was never received",
                    "code": "client_tools_no_result",
                    "tool": name,
                    "tool_call_id": call_id,
                },
                ensure_ascii=False,
            )
        if pending.is_error:
            return json.dumps(
                {
                    "error": pending.output,
                    "code": "client_tools_client_error",
                    "tool": name,
                    "tool_call_id": call_id,
                },
                ensure_ascii=False,
            )
        # The output is already a string on the wire (the client serialized
        # whatever its host produced).  Return it verbatim — tool results in
        # the conversation are content strings, not re-serialized JSON.
        return pending.output

    # ── client surface (called from the HTTP handlers) ────────────────

    def resolve(self, tool_call_id: str, output: str, is_error: bool = False) -> bool:
        """Deliver the client's result for a pending call. True when delivered."""
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return False
        with self.lock:
            pending = self.pending.get(tool_call_id)
            if pending is None or pending.event.is_set():
                return False
            pending.output = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            pending.is_error = bool(is_error)
        pending.event.set()
        return True

    def close(self) -> None:
        """Turn ended / client went away: fail closed, wake every waiter."""
        with self.lock:
            self.closed = True
            waiters = list(self.pending.values())
        for pending in waiters:
            # Leave output None → dispatch returns the no-result error.
            pending.event.set()


def install_bridge(agent, bridge: ClientToolBridge) -> None:
    """Attach *bridge* to the agent instance (injection, D4).

    Mirrors the context-engine pattern in ``agent/agent_init.py``: schemas
    are appended to the agent's tool list AFTER the registry snapshot, with
    name-dedup so a catalog name shadowing a native tool can never replace
    it (the catalog was already 400-rejected on collision at the HTTP edge —
    this is defense in depth for the in-process path).
    """
    agent._client_tool_bridge = bridge
    tools = getattr(agent, "tools", None)
    if tools is None:
        return
    valid = getattr(agent, "valid_tool_names", None)
    if valid is None:
        valid = set()
        agent.valid_tool_names = valid
    existing = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
    for schema in bridge.schemas():
        name = schema["function"]["name"]
        if name in existing or name in valid:
            # Never shadow a native tool from the in-process path.
            continue
        tools.append(schema)
        valid.add(name)
        existing.add(name)


def bridge_for(agent) -> Optional[ClientToolBridge]:
    """Return the agent's bridge, or None when the request had no catalog."""
    bridge = getattr(agent, "_client_tool_bridge", None)
    return bridge if isinstance(bridge, ClientToolBridge) else None


def build_tool_result_response(adapter, session_id: str, body: Dict[str, Any], error_payload):
    """Build the response for POST /api/sessions/{id}/chat/tool-result.

    Called from the aiohttp handler in api_server.py (which has already run
    auth + JSON-body parsing).  ``adapter`` is the APIServerAdapter and
    ``error_payload`` its module-level ``_openai_error``; returns a plain
    dict: ``(payload, status)`` so this module stays aiohttp-free.
    """
    bridge = adapter._session_client_tool_bridges.get(session_id)
    if bridge is None:
        return (
            error_payload(
                f"No active client-tools channel for session: {session_id}",
                code="client_tools_not_active",
            ),
            409,
        )
    tool_call_id = body.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return (
            error_payload(
                "tool-result needs a non-empty 'tool_call_id'",
                code="invalid_tool_result",
            ),
            400,
        )
    if "output" not in body:
        return (
            error_payload(
                "tool-result needs an 'output'",
                code="invalid_tool_result",
            ),
            400,
        )
    output = body.get("output")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    if len(output) > _MAX_RESULT_OUTPUT_LEN:
        return (
            error_payload(
                "tool-result 'output' too large",
                code="invalid_tool_result",
            ),
            413,
        )
    is_error = bool(body.get("is_error", False))
    if bridge.resolve(tool_call_id, output, is_error):
        return ({"delivered": True}, 200)
    return (
        error_payload(
            f"Unknown or already-resolved tool_call_id: {tool_call_id}",
            code="tool_call_not_pending",
        ),
        409,
    )
