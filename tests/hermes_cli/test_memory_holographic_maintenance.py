"""CLI coverage for holographic memory audit and garbage collection."""

from __future__ import annotations

import json
from argparse import Namespace

from plugins.memory.holographic.store import MemoryStore


def _prepare_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "memory_store.db"
    (home / "config.yaml").write_text(
        "memory:\n"
        "  provider: holographic\n"
        "plugins:\n"
        "  hermes-memory-store:\n"
        f"    db_path: {db_path}\n"
        "    hrr_dim: 64\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return db_path


def test_cmd_memory_audit_emits_machine_readable_report(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import cmd_memory

    db_path = _prepare_home(tmp_path, monkeypatch)
    with MemoryStore(db_path, hrr_dim=64) as store:
        store.add_fact("CLI audit canary.", entities=["CliAuditCanary"])

    cmd_memory(Namespace(memory_command="audit", json=True))

    report = json.loads(capsys.readouterr().out)
    assert report["integrity_check"] == "ok"
    assert report["facts"] == 1
    assert report["healthy"] is True


def test_cmd_memory_gc_removes_orphan_entities(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import cmd_memory

    db_path = _prepare_home(tmp_path, monkeypatch)
    with MemoryStore(db_path, hrr_dim=64) as store:
        store._conn.execute("INSERT INTO entities (name) VALUES (?)", ("CliOrphan",))

    cmd_memory(Namespace(memory_command="gc", yes=True))

    assert "Removed 1 orphan" in capsys.readouterr().out
    with MemoryStore(db_path, hrr_dim=64) as store:
        assert store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
