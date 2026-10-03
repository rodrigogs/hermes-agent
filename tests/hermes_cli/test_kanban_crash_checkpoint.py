"""Crash-checkpoint tests: a retrying worker resumes instead of starting over.

2026-10-03 incident: a gateway restart killed run 71 mid-flight and the
retrying worker (run 72) saw "Prior attempts" with an EMPTY summary — every
unit of uncommitted reasoning was lost. The fix writes a run-start marker to
the worker log before each spawn, recovers the tail after the LAST marker on
crash, and stores it as the crashed run's ``summary`` (which
``build_worker_context`` already surfaces).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import (
    _RUN_START_MARKER,
    _checkpoint_from_worker_log,
    worker_logs_dir,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_log(kanban_home, task_id, *chunks):
    log_path = worker_logs_dir() / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(c.encode() if isinstance(c, str) else c for c in chunks)
    log_path.write_bytes(data)


def test_checkpoint_extracts_tail_after_last_marker(kanban_home):
    """The tail after the LAST marker is the dead run's own activity."""
    tid = "t_abc"
    _write_log(
        kanban_home, tid,
        "old run output\n", _RUN_START_MARKER,
        "reading oblique-recon.ts\n", "writing freeMprFrame.ts\n",
        _RUN_START_MARKER,
        "current run: editing freeMprPolicy.ts\n",
    )
    ck = _checkpoint_from_worker_log(tid)
    assert ck is not None
    assert "editing freeMprPolicy.ts" in ck
    assert "reading oblique-recon.ts" not in ck  # prior run's tail excluded


def test_checkpoint_none_without_marker(kanban_home):
    """Legacy logs (no marker) yield None, not a stale prior-run summary."""
    _write_log(kanban_home, "t_legacy", "old output without marker\n")
    assert _checkpoint_from_worker_log("t_legacy") is None


def test_checkpoint_none_when_log_missing(kanban_home):
    assert _checkpoint_from_worker_log("t_nolog") is None


def test_checkpoint_strips_ansi_and_is_bounded(kanban_home):
    _write_log(
        kanban_home, "t_ansi",
        _RUN_START_MARKER,
        "\x1b[32mok\x1b[0m line\n\n\n" + ("x" * 10000) + "\n",
    )
    ck = _checkpoint_from_worker_log("t_ansi", max_bytes=2048)
    assert ck is not None
    assert "\x1b[" not in ck
    assert len(ck) <= 2048
    # The tail keeps the LAST max_bytes; the leading "ok line" (written first)
    # is truncated away — only the trailing 'x' flood remains. Proves the
    # bound keeps the END (the most recent activity), not the start.
    assert set(ck) == {"x"}


def test_crash_populates_summary_from_checkpoint(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A dead-PID reclaim stores the log tail as the run's summary."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        _write_log(
            kanban_home, tid,
            _RUN_START_MARKER,
            "reading freeMprFrame.ts\n",
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        row = conn.execute(
            "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert row is not None and row["summary"] is not None
        assert "reading freeMprFrame.ts" in row["summary"]
    finally:
        conn.close()
