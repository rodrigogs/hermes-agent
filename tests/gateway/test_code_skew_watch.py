"""Runner wiring tests for the code-skew watch.

These exercise the GatewayRunner methods that the per-turn and post-turn
hooks call: ``_maybe_code_skew_auto_restart`` (the idle auto-restart decision,
whose cardinal property is that it NEVER fires while a session slot is busy)
and ``_send_code_skew_notice`` (which must never raise into the turn).

The pure predicates live in ``gateway/code_skew.py`` and are covered by
``tests/test_code_skew.py``; these tests pin the wiring around them.
"""

import asyncio

from unittest.mock import MagicMock

from gateway.run import GatewayRunner
from gateway import code_skew

from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

_SKEW = ("abc1234567", "def4567890")


def _make_watch_runner(
    monkeypatch,
    *,
    watch_config: tuple[bool, bool] = (True, True),
    supervisor: bool = True,
    skew: tuple[str, str] | None = _SKEW,
) -> tuple[GatewayRunner, MagicMock]:
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._maybe_code_skew_auto_restart = (
        GatewayRunner._maybe_code_skew_auto_restart.__get__(runner, GatewayRunner)
    )
    # The legacy _running_agents property is a live SessionState view; replace
    # it with a plain dict so "idle" vs "busy" is unambiguous in this test.
    monkeypatch.setattr(GatewayRunner, "_running_agents", {})
    runner._running_agents = {}
    monkeypatch.setattr(code_skew, "watch_config", lambda agent_cfg=None: watch_config)
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: skew)
    record = MagicMock(return_value=None)
    monkeypatch.setattr(code_skew, "record_auto_restart", record)
    monkeypatch.setattr(
        "gateway.restart.is_gateway_supervisor_process", lambda: supervisor
    )
    return runner, record


class TestMaybeCodeSkewAutoRestart:
    def test_never_fires_with_busy_running_agents(self, monkeypatch):
        """THE property: a turn in flight must never be cut by this path."""
        runner, record = _make_watch_runner(monkeypatch)
        runner._running_agents = {"sess-a": object()}

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_not_called()
        record.assert_not_called()

    def test_fires_when_idle_and_supervised(self, monkeypatch):
        runner, record = _make_watch_runner(monkeypatch)

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_called_once_with(detached=False, via_service=True)
        record.assert_called_once_with("abc1234567", "def4567890")

    def test_skips_without_supervisor(self, monkeypatch):
        """exit 75 is only meaningful to a service manager — a bare process
        must not be asked to kill itself."""
        runner, record = _make_watch_runner(monkeypatch, supervisor=False)

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_not_called()
        record.assert_not_called()

    def test_skips_when_feature_disabled(self, monkeypatch):
        runner, record = _make_watch_runner(monkeypatch, watch_config=(True, False))

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_not_called()
        record.assert_not_called()

    def test_skips_without_skew(self, monkeypatch):
        runner, record = _make_watch_runner(monkeypatch, skew=None)

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_not_called()
        record.assert_not_called()

    def test_skips_when_already_restarting(self, monkeypatch):
        runner, record = _make_watch_runner(monkeypatch)
        runner._restart_requested = True

        runner._maybe_code_skew_auto_restart()

        runner.request_restart.assert_not_called()
        record.assert_not_called()

    def test_never_raises_into_the_turn_unwind(self, monkeypatch):
        runner, _record = _make_watch_runner(monkeypatch)
        runner.request_restart = MagicMock(side_effect=RuntimeError("boom"))

        # Must be swallowed — the finally-block caller cannot propagate.
        runner._maybe_code_skew_auto_restart()


class TestSendCodeSkewNotice:
    def test_delivers_standalone_message(self):
        runner, adapter = make_restart_runner()
        source = make_restart_source()
        send = MagicMock()
        adapter._send_with_retry = send
        runner._adapter_for_source = lambda src: adapter
        runner._thread_metadata_for_source = lambda src, reply_anchor=None: {"k": "v"}

        asyncio.run(runner._send_code_skew_notice(source, "⚠️ notice"))

        send.assert_called_once()
        kwargs = send.call_args.kwargs
        assert kwargs["chat_id"] == source.chat_id
        assert kwargs["content"] == "⚠️ notice"

    def test_adapter_failure_is_swallowed(self):
        runner, adapter = make_restart_runner()
        source = make_restart_source()
        adapter._send_with_retry = MagicMock(side_effect=RuntimeError("delivery boom"))
        runner._adapter_for_source = lambda src: adapter

        # The notice must never raise into the turn that follows it.
        asyncio.run(runner._send_code_skew_notice(source, "⚠️ notice"))

    def test_missing_adapter_is_a_noop(self):
        runner, _adapter = make_restart_runner()
        source = make_restart_source()
        runner._adapter_for_source = lambda src: None

        import asyncio

        asyncio.run(runner._send_code_skew_notice(source, "⚠️ notice"))
