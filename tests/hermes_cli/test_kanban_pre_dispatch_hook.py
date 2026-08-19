"""Tests for the ``pre_kanban_dispatch`` model-selection hook.

Verifies the pre-dispatch transform hook fires AFTER the task is claimed
and BEFORE the worker spawns (ready and review lanes), that dict results
are applied last-writer-wins per field on the in-memory Task, and that a
human ``set_model_override`` always wins — the hook is not even consulted
while the task's ``model_override`` column is set.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _plugins():
    """Resolve ``hermes_cli.plugins`` at CALL time, never at import time.

    Several kanban test modules reload / replace
    ``sys.modules["hermes_cli.plugins"]``, so a module object captured by a
    top-level import can become a stale duplicate: its ``PluginManager`` cache
    is not the one the production path reaches, because
    ``lifecycle.has_hook`` / ``invoke_hook`` re-resolve ``from hermes_cli
    import plugins`` on every call. Registering on the stale copy made these
    tests pass alone and fail after such a module (observed: has_hook() True
    through the captured module and False through lifecycle, same hook, same
    process). Resolving late is what keeps the registration and the dispatch
    looking at ONE registry.
    """
    import hermes_cli.plugins as mod

    return mod


def _hook_manager():
    """Return the live manager with plugin discovery already done.

    Discovery must be forced BEFORE a callback is appended: the first
    production ``has_hook()`` goes through ``plugins._delivery_manager()``,
    which lazily runs ``discover_and_load()`` and rebuilds the hook registry,
    silently dropping anything registered beforehand.
    """
    plugins = _plugins()
    plugins.has_hook("pre_kanban_dispatch")  # force the lazy discovery
    return plugins.get_plugin_manager()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register a capturing callback for pre_kanban_dispatch.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    events: list[dict] = []
    saved = _register_hook(lambda **kw: events.append(kw))
    try:
        yield events
    finally:
        _restore_hooks(saved)


def _register_hook(callback):
    """Append *callback* to pre_kanban_dispatch; return the restore token.

    The token carries the manager it was taken from, so the restore cannot
    land on a different module copy than the registration did.
    """
    mgr = _hook_manager()
    saved = (mgr, {k: list(v) for k, v in mgr._hooks.items()})
    mgr._hooks.setdefault("pre_kanban_dispatch", []).append(callback)
    return saved


def _restore_hooks(saved):
    mgr, hooks = saved
    mgr._hooks = hooks


def _dispatch_spawn(conn, tid, spawn_fn, **kwargs):
    result = kb.dispatch_once(conn, spawn_fn=spawn_fn, **kwargs)
    assert any(row[0] == tid for row in result.spawned), result


# ---------------------------------------------------------------------------
# Hook surface
# ---------------------------------------------------------------------------


def test_hook_in_valid_hooks():
    assert "pre_kanban_dispatch" in _plugins().VALID_HOOKS


# ---------------------------------------------------------------------------
# Ready-lane dispatch
# ---------------------------------------------------------------------------


def test_dispatch_applies_hook_model_to_spawned_task(
    kanban_home, all_assignees_spawnable, captured_hooks,
):
    """Ready-lane dispatch applies the hook's {model, provider} to the Task handed to spawn_fn."""
    seen: list = []

    def _hook(**kw):
        return {"model": "hook-model", "provider": "hook-provider"}

    saved = _register_hook(_hook)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved)

    assert len(seen) == 1
    assert seen[0].model_override == "hook-model"
    assert seen[0].provider_override == "hook-provider"
    # Payload carries the correlation kwargs (task_id, board, assignee,
    # plus the rest of the kanban common kwargs).
    fired = [e for e in captured_hooks if e.get("task_id") == tid]
    assert len(fired) == 1
    kw = fired[0]
    assert kw["assignee"] == "alice"
    assert "board" in kw
    assert kw["run_id"] is not None
    assert "profile_name" in kw


def test_human_model_override_wins_over_hook(
    kanban_home, all_assignees_spawnable, captured_hooks,
):
    """A human set_model_override is never overwritten: the hook is not consulted."""
    seen: list = []

    def _hook(**kw):
        return {"model": "hook-model"}

    saved = _register_hook(_hook)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")
            assert kb.set_model_override(conn, tid, "human-model", "human-provider")

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved)

    assert len(seen) == 1
    assert seen[0].model_override == "human-model"
    assert seen[0].provider_override == "human-provider"
    # The hook was not even consulted: no payload was built, no callback ran.
    assert captured_hooks == []


def test_no_hook_registered_does_not_fire(
    kanban_home, all_assignees_spawnable,
):
    """Without any subscriber dispatch is unchanged and nothing is invoked."""
    seen: list = []

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")

        def _spawn(task, workspace, **kw):
            seen.append(task)
            return 4242

        _dispatch_spawn(conn, tid, _spawn)
    finally:
        conn.close()

    assert len(seen) == 1
    assert seen[0].model_override is None


# ---------------------------------------------------------------------------
# Review-lane dispatch
# ---------------------------------------------------------------------------


def test_review_lane_applies_hook(
    kanban_home, all_assignees_spawnable, captured_hooks,
):
    """Review-lane dispatch applies the hook to the reviewer's Task too."""
    seen: list = []

    def _hook(**kw):
        return {"model": "review-model"}

    saved = _register_hook(_hook)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            assert kb.request_review(
                conn, tid, summary="ready", expected_run_id=claimed.current_run_id,
            )

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved)

    assert len(seen) == 1
    assert seen[0].model_override == "review-model"
    # The sdlc-review force-load still applies alongside the hook result.
    assert "sdlc-review" in (seen[0].skills or [])


# ---------------------------------------------------------------------------
# Best-effort guarantees
# ---------------------------------------------------------------------------


def test_misbehaving_hook_does_not_break_dispatch(
    kanban_home, all_assignees_spawnable,
):
    """A hook callback that raises must not break the dispatch loop."""

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    saved = _register_hook(_boom)
    try:
        seen: list = []
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved)

    assert len(seen) == 1
    assert seen[0].model_override is None
    assert seen[0].provider_override is None


def test_invalid_hook_returns_ignored(
    kanban_home, all_assignees_spawnable,
):
    """Non-dict results, unknown fields, and non-string values are dropped."""
    seen: list = []

    def _not_a_dict(**kw):
        return "not-a-dict"

    def _mixed_dict(**kw):
        # A non-string model and an unknown field are dropped; the valid
        # provider survives.
        return {"model": 123, "bogus_field": "x", "provider": "ok-provider"}

    saved_a = _register_hook(_not_a_dict)
    saved_b = _register_hook(_mixed_dict)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved_b)

    assert len(seen) == 1
    assert seen[0].model_override is None
    assert seen[0].provider_override == "ok-provider"


def test_last_writer_wins_per_field(kanban_home, all_assignees_spawnable):
    """Multiple hooks compose in registration order, last-writer-wins per field."""
    seen: list = []

    def _hook_a(**kw):
        return {"model": "model-a", "provider": "provider-a"}

    def _hook_b(**kw):
        return {"model": "model-b"}  # provider from a survives

    saved_a = _register_hook(_hook_a)
    saved_b = _register_hook(_hook_b)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")

            def _spawn(task, workspace, **kw):
                seen.append(task)
                return 4242

            _dispatch_spawn(conn, tid, _spawn)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved_b)

    assert len(seen) == 1
    assert seen[0].model_override == "model-b"
    assert seen[0].provider_override == "provider-a"


# ---------------------------------------------------------------------------
# End-to-end: hook-selected model reaches the worker argv
# ---------------------------------------------------------------------------


def test_hook_model_reaches_worker_argv(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path,
):
    """A hook-selected model flows into the worker's ``-m``/``--provider`` args."""

    def _hook(**kw):
        return {"model": "hook-model", "provider": "hook-provider"}

    saved = _register_hook(_hook)
    try:
        captured: dict = {}

        class FakeProc:
            def __init__(self, *a, **k):
                pass

            # _default_spawn returns proc.pid (kanban_db.py:_default_spawn);
            # the dispatcher then persists int(pid).
            pid = 4242

            def poll(self):
                return None

            def __int__(self):
                return 4242

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = list(cmd)
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")
            # Real _default_spawn path: dispatch with the default spawn fn.
            _dispatch_spawn(conn, tid, None)
        finally:
            conn.close()
    finally:
        _restore_hooks(saved)

    # The launcher prefix built by ``_resolve_hermes_argv()`` can itself
    # contain a ``-m`` (the ``sys.executable -m hermes_cli.main`` fallback),
    # so scanning the whole argv would match the INTERPRETER's ``-m`` and
    # read back ``"hermes_cli.main"``. Look only at the per-task arguments:
    # ``_default_spawn`` emits the fixed prefix ``-p <profile> --cli
    # --accept-hooks`` first and appends every per-task flag after it.
    cmd = captured["cmd"]
    assert "--accept-hooks" in cmd
    args = cmd[cmd.index("--accept-hooks") + 1:]

    assert "-m" in args
    assert args[args.index("-m") + 1] == "hook-model"
    assert "--provider" in args
    assert args[args.index("--provider") + 1] == "hook-provider"
