#!/usr/bin/env python3
"""Focused tests for profile-backed delegate_task children (issue #41889)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.delegate_tool as dt


def _parent(enabled=("file", "web", "terminal")):
    return SimpleNamespace(
        base_url="https://parent.example/v1",
        api_key="parent-key",
        provider="openrouter",
        api_mode="chat_completions",
        model="parent/model",
        platform="cli",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection="",
        request_overrides={},
        openrouter_min_coding_score=None,
        prefill_messages=None,
        reasoning_config=None,
        max_tokens=None,
        acp_command=None,
        acp_args=[],
        enabled_toolsets=list(enabled),
        disabled_toolsets=[],
        valid_tool_names=[],
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        _delegate_spinner=None,
        _memory_manager=None,
        _fallback_chain=None,
        _client_kwargs={},
        session_id="parent-session",
        _current_turn_id="",
        session_estimated_cost_usd=0.0,
        _interrupt_requested=False,
    )


def _bundle(name="reviewer"):
    return {
        "name": name,
        "soul": f"You are the {name} specialist.",
        "model": f"{name}/model",
        "provider": "openrouter",
        "base_url": "https://profile.example/v1",
        "api_key": "profile-key",
        "api_mode": "chat_completions",
        "toolsets": ["file", "web"],
    }


def _creds():
    return {
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "request_overrides": None,
        "max_output_tokens": None,
        "command": None,
        "args": None,
    }


def test_schema_exposes_top_level_and_per_task_profile():
    props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert props["profile"]["type"] == "string"
    task_props = props["tasks"]["items"]["properties"]
    assert task_props["profile"]["type"] == "string"
    assert "profile" in dt._build_top_level_description().lower()


def test_profile_soul_is_prepended_to_child_prompt():
    prompt = dt._build_child_system_prompt(
        "review the diff", profile_soul="You are a strict reviewer."
    )
    assert prompt.startswith("You are a strict reviewer.")
    assert "YOUR TASK:" in prompt


def test_unlisted_profile_is_rejected_before_resolution():
    with patch(
        "tools.delegate_tool._get_allowed_delegate_profiles",
        return_value={"reviewer"},
    ), patch("tools.delegate_tool._resolve_profile_bundle") as resolve, patch(
        "tools.delegate_tool._resolve_delegation_credentials", return_value=_creds()
    ):
        result = json.loads(
            dt.delegate_task(goal="work", profile="auditor", parent_agent=_parent())
        )
    assert "not allowed" in result["error"]
    assert "reviewer" in result["error"]
    resolve.assert_not_called()


def test_missing_profile_is_rejected_before_child_construction():
    assert hasattr(dt, "_resolve_profile_bundle")
    with patch(
        "tools.delegate_tool._get_allowed_delegate_profiles",
        return_value={"ghost"},
    ), patch(
        "tools.delegate_tool._resolve_profile_bundle",
        side_effect=ValueError("Profile 'ghost' does not exist."),
    ), patch("tools.delegate_tool._build_child_agent") as build, patch(
        "tools.delegate_tool._resolve_delegation_credentials", return_value=_creds()
    ):
        result = json.loads(
            dt.delegate_task(goal="work", profile="ghost", parent_agent=_parent())
        )
    assert "does not exist" in result["error"]
    build.assert_not_called()


def test_single_profile_routes_identity_runtime_and_tools_to_child():
    child = MagicMock()
    child.model = "reviewer/model"
    completed = {
        "task_index": 0,
        "profile": "reviewer",
        "status": "completed",
        "summary": "ok",
    }
    with patch(
        "tools.delegate_tool._get_allowed_delegate_profiles",
        return_value={"reviewer"},
    ), patch("tools.delegate_tool._resolve_profile_bundle", return_value=_bundle()), patch(
        "tools.delegate_tool._resolve_delegation_credentials", return_value=_creds()
    ), patch("tools.delegate_tool._build_child_agent", return_value=child) as build, patch(
        "tools.delegate_tool._run_single_child", return_value=completed
    ):
        result = json.loads(
            dt.delegate_task(goal="review", profile="reviewer", parent_agent=_parent())
        )

    assert result["results"][0]["profile"] == "reviewer"
    kwargs = build.call_args.kwargs
    assert kwargs["profile_name"] == "reviewer"
    assert kwargs["profile_soul"].startswith("You are the reviewer")
    assert kwargs["model"] == "reviewer/model"
    assert kwargs["override_api_key"] == "profile-key"
    assert kwargs["toolsets"] == ["file", "web"]


def test_top_level_profile_is_inherited_and_per_task_profile_wins():
    child = MagicMock()
    child.model = "specialist/model"

    def resolve(name):
        return _bundle(name)

    with patch(
        "tools.delegate_tool._get_allowed_delegate_profiles",
        return_value={"reviewer", "coder"},
    ), patch("tools.delegate_tool._resolve_profile_bundle", side_effect=resolve) as bundle, patch(
        "tools.delegate_tool._resolve_delegation_credentials", return_value=_creds()
    ), patch("tools.delegate_tool._build_child_agent", return_value=child) as build, patch(
        "tools.delegate_tool._run_single_child",
        return_value={"task_index": 0, "status": "completed", "summary": "ok"},
    ):
        dt.delegate_task(
            profile="reviewer",
            tasks=[{"goal": "a"}, {"goal": "b", "profile": "coder"}],
            parent_agent=_parent(),
        )

    assert [call.args[0] for call in bundle.call_args_list] == ["reviewer", "coder"]
    assert [call.kwargs["profile_name"] for call in build.call_args_list] == [
        "reviewer",
        "coder",
    ]


def test_profile_toolsets_are_bounded_by_parent_capabilities():
    with patch("run_agent.AIAgent", return_value=MagicMock()):
        child = dt._build_child_agent(
            task_index=0,
            goal="review",
            context=None,
            toolsets=["file", "web"],
            model="reviewer/model",
            max_iterations=3,
            task_count=1,
            parent_agent=_parent(enabled=("file", "terminal")),
            override_api_key="profile-key",
            profile_name="reviewer",
            profile_soul="Strict reviewer.",
        )
    assert child._delegate_profile == "reviewer"
    assert child._delegate_profile_dropped_toolsets == ["web"]


def test_profile_child_requires_own_runtime_secret():
    with patch("run_agent.AIAgent") as agent_cls:
        try:
            dt._build_child_agent(
                task_index=0,
                goal="review",
                context=None,
                toolsets=["file"],
                model="reviewer/model",
                max_iterations=3,
                task_count=1,
                parent_agent=_parent(),
                profile_name="reviewer",
                profile_soul="Strict reviewer.",
            )
        except ValueError as exc:
            assert "Refusing to inherit" in str(exc)
        else:
            raise AssertionError("profile child inherited the parent credential")
    agent_cls.assert_not_called()


def test_profile_child_does_not_inherit_parent_fallback_or_credential_pool():
    parent = _parent()
    parent._fallback_chain = [{"provider": "parent", "model": "fallback"}]
    fake_child = MagicMock()
    with patch("run_agent.AIAgent", return_value=fake_child) as agent_cls, patch(
        "tools.delegate_tool._resolve_child_credential_pool"
    ) as pool_resolver:
        dt._build_child_agent(
            task_index=0,
            goal="review",
            context=None,
            toolsets=["file"],
            model="reviewer/model",
            max_iterations=3,
            task_count=1,
            parent_agent=parent,
            override_provider="openrouter",
            override_base_url="https://profile.example/v1",
            override_api_key="profile-key",
            profile_name="reviewer",
            profile_soul="Strict reviewer.",
        )

    assert agent_cls.call_args.kwargs["fallback_model"] is None
    pool_resolver.assert_not_called()


def test_fabricated_result_metadata_keeps_profile_identity():
    child = SimpleNamespace(
        _delegate_profile="reviewer",
        _delegate_profile_dropped_toolsets=["web"],
    )
    assert dt._profile_fields_from_child(child) == {
        "profile": "reviewer",
        "profile_toolsets_dropped": ["web"],
    }


def test_agent_dispatch_forwards_profile_to_delegate_task():
    import run_agent

    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            object(), {"goal": "review", "profile": "reviewer"}
        )

    assert captured["profile"] == "reviewer"
    assert "parent_agent" in captured


class _ProfileWireHandler(BaseHTTPRequestHandler):
    captured = {}

    def log_message(self, format: str, *args):
        del format, args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("messages"):
            type(self).captured = {
                "authorization": self.headers.get("Authorization", ""),
                "model": body.get("model"),
                "messages": body.get("messages", []),
            }

        if body.get("stream"):
            chunks = [
                {
                    "id": "profile-e2e",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "PROFILE_DELEGATION_OK"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "profile-e2e",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body.get("model"),
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                },
            ]
            payload = "".join(
                f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
            ) + "data: [DONE]\n\n"
            raw = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            raw = json.dumps(
                {
                    "id": "profile-e2e",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "PROFILE_DELEGATION_OK",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")

        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_real_profile_delegate_reaches_provider_with_profile_identity():
    """Real delegate_task -> child AIAgent -> HTTP provider request."""
    from run_agent import AIAgent

    _ProfileWireHandler.captured = {}
    server = HTTPServer(("127.0.0.1", 0), _ProfileWireHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    parent = AIAgent(
        base_url=base_url,
        api_key="parent-test-key",
        model="parent/test-model",
        provider="openai",
        api_mode="chat_completions",
        enabled_toolsets=["file"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=3,
    )
    profile_bundle = {
        "name": "reader",
        "soul": "SOUL_MARKER_READER_PROFILE",
        "model": "reader/test-model",
        "provider": "openai",
        "base_url": base_url,
        "api_key": "profile-test-key",
        "api_mode": "chat_completions",
        "toolsets": ["file"],
    }
    try:
        with patch(
            "tools.delegate_tool._get_allowed_delegate_profiles",
            return_value={"reader"},
        ), patch(
            "tools.delegate_tool._resolve_profile_bundle",
            return_value=profile_bundle,
        ), patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value=_creds(),
        ):
            data = json.loads(
                dt.delegate_task(
                    goal="Return the canary token.",
                    profile="reader",
                    background=False,
                    parent_agent=parent,
                )
            )
    finally:
        parent.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    result = data["results"][0]
    assert result["profile"] == "reader"
    assert result["model"] == "reader/test-model"
    assert result["status"] == "completed"
    assert result["summary"] == "PROFILE_DELEGATION_OK"
    captured = _ProfileWireHandler.captured
    assert captured["authorization"] == "Bearer profile-test-key"
    assert captured["authorization"] != "Bearer parent-test-key"
    assert captured["model"] == "reader/test-model"
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in captured["messages"]
        if message.get("role") == "system"
    )
    assert "SOUL_MARKER_READER_PROFILE" in system_text
