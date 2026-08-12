"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway is a single long-lived process; its ``sys.modules`` is frozen at
boot. If the checkout is updated underneath it (a manual ``git pull``, or the
window before ``hermes update``'s graceful restart fires), a first-time lazy
import on a new code path can resolve a freshly-pulled consumer module against a
stale cached dependency -> ImportError (see
``tests/test_stale_utils_module_import.py`` for the exact failure).

We snapshot the checkout revision at gateway startup and compare on demand, so
risky callers (e.g. ``/model`` switching) can refuse with a clear "restart the
gateway" message instead of crashing on a cryptic import error.

If the revision can't be read (non-git install, IO error), the boot snapshot
stays ``None`` and skew detection no-ops — it never produces a false positive.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None


def _fingerprint() -> str | None:
    """Current checkout fingerprint, reusing the CLI's git-rev reader.

    ``hermes_cli.main`` is always already imported in a gateway process (it's
    the entry point), so this import is free and avoids duplicating the
    worktree-aware ref resolution.
    """
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout revision at gateway startup (idempotent)."""
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()


def _short(fingerprint: str) -> str:
    """Render a ``git:<ref>:<sha>`` fingerprint as a compact label."""
    sha = fingerprint.rsplit(":", 1)[-1]
    if sha and sha != "unresolved" and len(sha) > 10:
        return sha[:10]
    return sha or fingerprint


def detect_code_skew() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` short labels if the checkout drifted
    since boot, else ``None``."""
    if _boot_fingerprint is None:
        return None
    current = _fingerprint()
    if current is None or current == _boot_fingerprint:
        return None
    return _short(_boot_fingerprint), _short(current)


# ---------------------------------------------------------------------------
# Per-turn watch: warn loudly, never refuse the turn.
# ---------------------------------------------------------------------------
#
# ``_fingerprint`` is a pure file read (no git subprocess), so the runner can
# afford to check it at every agent-turn start. The returned notice is
# delivered as a standalone message; the turn itself is NEVER refused on
# skew — this is a warning channel, not a gate. ``_warned_session_keys``
# throttles the loud notice to once per session per boot; ``/status`` carries
# the persistent surface while the condition holds (so an operator who checks
# in later than the one-shot notice still sees it).

_warned_session_keys: set[str] = set()


def _watch_flag(agent_cfg: dict, key: str, default: bool) -> bool:
    value = agent_cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def watch_config(agent_cfg: dict | None = None) -> tuple[bool, bool]:
    """Return ``(warning_enabled, auto_restart_enabled)``.

    Reads ``agent.code_skew_warning`` / ``agent.code_skew_auto_restart`` from
    the user config with ``DEFAULT_CONFIG`` fallback — the same resolution
    pattern ``gateway.restart`` uses for its agent settings. ``agent_cfg`` is
    injectable for tests; when ``None`` the mtime-cached raw config is read
    (cheap per turn).
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG.get("agent", {})
    if agent_cfg is None:
        agent_cfg = {}
        try:
            from hermes_cli.config import read_raw_config

            loaded = read_raw_config() or {}
            if isinstance(loaded, dict):
                section = loaded.get("agent")
                if isinstance(section, dict):
                    agent_cfg = section
        except Exception:
            pass
    warning_default = bool(defaults.get("code_skew_warning", True))
    auto_default = bool(defaults.get("code_skew_auto_restart", True))
    return (
        _watch_flag(agent_cfg, "code_skew_warning", warning_default),
        _watch_flag(agent_cfg, "code_skew_auto_restart", auto_default),
    )


def format_warning(boot_rev: str, disk_rev: str) -> str:
    """The loud per-session notice shown when the checkout drifted."""
    return (
        "⚠️ Hermes code changed on disk since this gateway booted "
        f"(boot `{boot_rev}` → disk `{disk_rev}`). The running gateway still "
        "executes the boot code. Restart it to load the new code: `/restart`."
    )


def per_turn_warning(session_key: str | None = None) -> str | None:
    """Return the once-per-boot-per-session skew notice, else ``None``.

    A resolved drift clears the warned set so the next skew episode re-warns.
    ``session_key=None`` disables the throttle (caller-managed cadence).
    Never raises and never refuses anything — callers must treat the result
    as an optional notice, not a gate.
    """
    global _warned_session_keys
    if not watch_config()[0]:
        return None
    skew = detect_code_skew()
    if not skew:
        if _warned_session_keys:
            _warned_session_keys = set()
        return None
    if session_key is not None and session_key in _warned_session_keys:
        return None
    if session_key is not None:
        _warned_session_keys.add(session_key)
    return format_warning(*skew)


# ---------------------------------------------------------------------------
# Idle auto-restart (config-gated, supervisor-gated, never mid-turn).
# ---------------------------------------------------------------------------

def should_auto_restart(
    *,
    skew: tuple[str, str] | None,
    running_agents_empty: bool,
    enabled: bool,
) -> bool:
    """Idle-restart predicate: skew present, no messaging turn in flight,
    feature enabled.

    Deliberately does NOT consult cron/API work here: the in-band restart
    machinery (``request_restart`` → ``_await_active_work_before_restart``)
    drains ALL active work before ``stop()``, so the trigger only needs to
    guarantee that no turn is in flight at the moment it fires. The runner
    additionally requires a supervising service manager before acting —
    exit 75 is only meaningful to one.
    """
    return bool(skew and running_agents_empty and enabled)


_RESTART_RECORD_RELATIVE = ("state", "code_skew_restarts.json")
_RESTART_RECORD_CAP = 50


def get_restart_record_path(home: Path | None = None) -> Path:
    """Return ``<HERMES_HOME>/state/code_skew_restarts.json``."""
    from hermes_constants import get_hermes_home

    base = home if home is not None else get_hermes_home()
    return Path(base).joinpath(*_RESTART_RECORD_RELATIVE)


def record_auto_restart(
    boot_rev: str,
    disk_rev: str,
    *,
    home: Path | None = None,
) -> Path | None:
    """Append ``{ts, tag, boot_rev, disk_rev}`` to the restart ledger.

    Written BEFORE the exit-75 restart so the next boot can explain why the
    previous gateway exited — the record survives the restart itself. JSON
    array, newest last, ring-capped. Best-effort: a failed write must never
    block the restart it is documenting.
    """
    path = get_restart_record_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records = [r for r in data if isinstance(r, dict)]
            except (OSError, ValueError):
                records = []
        records.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": "gateway.code_skew_auto_restart",
                "boot_rev": boot_rev,
                "disk_rev": disk_rev,
            }
        )
        records = records[-_RESTART_RECORD_CAP:]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception:
        return None
