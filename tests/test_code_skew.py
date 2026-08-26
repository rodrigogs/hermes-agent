"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it. Also covers the per-turn watch
(once-per-session-per-boot warning, never a refusal), the idle auto-restart
predicate, and the ``state/code_skew_restarts.json`` restart ledger.

The dashboard mirror (#86207) lives here too:
``web_server._dashboard_code_skew_guard`` and the ``/api/model/options`` 503
guard, which protect the Models page from the same stale-module ImportError
after ``hermes update``.
"""

import asyncio
import contextlib
import json

import pytest

from gateway import code_skew


@pytest.fixture(autouse=True)
def _reset_boot_fingerprint(monkeypatch):
    """Each test starts with no recorded boot fingerprint or warned sessions."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    monkeypatch.setattr(code_skew, "_warned_session_keys", set())


def _with_skew(monkeypatch, boot: str = "git:refs/heads/main:abc1234567890", disk: str = "git:refs/heads/main:def4567890123"):
    monkeypatch.setattr(code_skew, "_fingerprint", lambda: boot)
    code_skew.record_boot_fingerprint()
    monkeypatch.setattr(code_skew, "_fingerprint", lambda: disk)


class TestDetectCodeSkew:
    def test_no_boot_fingerprint_means_no_skew(self, monkeypatch):
        # Nothing recorded (e.g. non-git install) -> never a false positive.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def456")
        assert code_skew.detect_code_skew() is None


    def test_drift_is_detected_with_short_revs(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "def4567890")




class TestShort:
    def test_shortens_long_sha(self):
        assert code_skew._short("git:refs/heads/main:abcdef0123456789") == "abcdef0123"

    def test_keeps_unresolved_marker(self):
        assert code_skew._short("git:refs/heads/main:unresolved") == "unresolved"

    def test_passes_short_sha_through_untruncated(self):
        assert code_skew._short("git:HEAD:abc1234") == "abc1234"


class TestModelSwitchSkewGuard:
    def test_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert slash_commands._model_switch_skew_guard() is None

    def test_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "hermes gateway restart" in msg


class TestWatchConfig:
    def test_defaults_on_when_config_is_silent(self):
        assert code_skew.watch_config({}) == (True, True)

    def test_explicit_disables(self):
        cfg = {"code_skew_warning": False, "code_skew_auto_restart": False}
        assert code_skew.watch_config(cfg) == (False, False)

    def test_string_flags_accepted(self):
        cfg = {"code_skew_warning": "false", "code_skew_auto_restart": "off"}
        assert code_skew.watch_config(cfg) == (False, False)
        assert code_skew.watch_config({"code_skew_warning": "on", "code_skew_auto_restart": "yes"}) == (True, True)

    def test_unknown_keys_keep_defaults(self):
        assert code_skew.watch_config({"unrelated": 1}) == (True, True)


class TestPerTurnWarning:
    def test_skew_warns_once_per_session_per_boot(self, monkeypatch):
        _with_skew(monkeypatch)
        first = code_skew.per_turn_warning("sess-a")
        assert first is not None
        assert "abc1234567" in first
        assert "def4567890" in first
        assert "restart" in first.lower()
        # Same session is throttled (once per boot), a different session is
        # still loud — the notice is per-session, not global-once.
        assert code_skew.per_turn_warning("sess-a") is None
        assert code_skew.per_turn_warning("sess-b") is not None

    def test_no_skew_stays_silent_and_clears_warned_set(self, monkeypatch):
        _with_skew(monkeypatch)
        assert code_skew.per_turn_warning("sess-a") is not None
        assert "sess-a" in code_skew._warned_session_keys
        # Drift resolves (checkout moved back to boot rev): silent again AND
        # the warned set is cleared so the next episode re-warns.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        assert code_skew.per_turn_warning("sess-b") is None
        assert code_skew._warned_session_keys == set()

    def test_disabled_warning_never_warns(self, monkeypatch):
        monkeypatch.setattr(code_skew, "watch_config", lambda agent_cfg=None: (False, True))
        _with_skew(monkeypatch)
        assert code_skew.per_turn_warning("sess-a") is None

    def test_turn_is_never_refused_by_skew(self, monkeypatch):
        # The notice has no refusal shape: it is a string notice or None, and
        # the function never raises even when the drift is present.
        _with_skew(monkeypatch)
        result = code_skew.per_turn_warning("sess-a")
        assert result is None or isinstance(result, str)
        assert code_skew.per_turn_warning("sess-b") is not None


class TestShouldAutoRestart:
    def test_fires_when_skew_idle_and_enabled(self):
        assert code_skew.should_auto_restart(
            skew=("abc1234567", "def4567890"), running_agents_empty=True, enabled=True
        )

    def test_never_with_busy_running_agents(self):
        # THE property: a turn in flight must never be cut by this path.
        assert not code_skew.should_auto_restart(
            skew=("abc1234567", "def4567890"), running_agents_empty=False, enabled=True
        )

    def test_requires_skew(self):
        assert not code_skew.should_auto_restart(
            skew=None, running_agents_empty=True, enabled=True
        )

    def test_requires_enabled(self):
        assert not code_skew.should_auto_restart(
            skew=("abc1234567", "def4567890"), running_agents_empty=True, enabled=False
        )


class TestRecordAutoRestart:
    def test_writes_old_new_and_timestamp_that_survive(self, tmp_path):
        path = code_skew.record_auto_restart("abc1234567", "def4567890", home=tmp_path)
        assert path is not None
        assert path == tmp_path / "state" / "code_skew_restarts.json"
        # The record must outlive the process that wrote it: a fresh read
        # (simulating the next boot) parses it with old/new revs + timestamp.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 1
        rec = data[0]
        assert rec["boot_rev"] == "abc1234567"
        assert rec["disk_rev"] == "def4567890"
        assert rec["tag"] == "gateway.code_skew_auto_restart"
        assert rec["ts"]

    def test_appends_and_rings_capped(self, tmp_path):
        for i in range(code_skew._RESTART_RECORD_CAP + 5):
            code_skew.record_auto_restart(f"boot{i}", f"disk{i}", home=tmp_path)
        data = json.loads((tmp_path / "state" / "code_skew_restarts.json").read_text(encoding="utf-8"))
        assert len(data) == code_skew._RESTART_RECORD_CAP
        assert data[-1]["boot_rev"] == f"boot{code_skew._RESTART_RECORD_CAP + 4}"

    def test_best_effort_on_unwritable_home(self):
        # home points at a file, so mkdir/state dir cannot be created —
        # a failed ledger write must never block the restart it documents.
        assert code_skew.record_auto_restart("a", "b", home=__file__) is None


class TestDashboardCodeSkewGuard:
    """Dashboard mirror of the gateway's model-switch skew guard (#86207)."""

    def test_dashboard_guard_returns_none_without_skew(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert web_server._dashboard_code_skew_guard() is None

    def test_dashboard_guard_message_names_revs_and_restart(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = web_server._dashboard_code_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "restart" in msg.lower()


class TestModelOptionsSkewGuard:
    """/api/model/options must refuse with a clear 503 when the dashboard is stale.

    Regression for #86207: a dashboard kept alive across ``hermes update``
    serves stale modules, so the picker's lazy import of names the update added
    (``agent.model_metadata.is_grok_46_family``) raised ImportError and the
    handler collapsed it into a generic 500.  With the guard, the stale process
    surfaces "Restart required" (503) and never reaches the payload build.
    """

    def test_stale_dashboard_returns_503_and_skips_payload_build(self, monkeypatch):
        from fastapi import HTTPException
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        payload_calls: list = []
        monkeypatch.setattr(
            "hermes_cli.inventory.build_model_options_payload",
            lambda *a, **k: payload_calls.append(1) or {"providers": []},
        )

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(web_server.get_model_options())

        assert excinfo.value.status_code == 503
        assert "restart" in str(excinfo.value.detail).lower()
        # The stale-import crash site (the payload build) is never reached.
        assert payload_calls == []

    def test_fresh_dashboard_builds_payload_unchanged(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)

        async def _fake_run_in_threadpool(func):
            return func()

        monkeypatch.setattr(web_server, "run_in_threadpool", _fake_run_in_threadpool)
        monkeypatch.setattr(
            web_server, "_profile_scope", lambda profile: contextlib.nullcontext()
        )
        monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: {})

        payload_calls: list = []
        expected = {"providers": [], "model": {}, "provider": None}
        monkeypatch.setattr(
            "hermes_cli.inventory.build_model_options_payload",
            lambda *a, **k: payload_calls.append(1) or expected,
        )

        result = asyncio.run(web_server.get_model_options())

        assert result == expected
        assert payload_calls == [1]
