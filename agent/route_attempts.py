"""Per-hop model-attempt journal — what the executor ACTUALLY did, per backend.

The routing trace (the plugin's ``routes.jsonl``) records what the router
PLANNED: the declared tier and the head of the planned chain
(``output.attempted_model``). What it never recorded is what HAPPENED on that
plan: which backend answered, which one failed after how long, and which
planned hops were skipped without a call. "Recusou em 300 ms" and "pendurou
15 s" both arrived at the operator's screen as the same line, and they demand
opposite actions. This module is the executor-side half of closing that gap:
it journals one record per backend engagement so a decision can be replayed as
an outcome, not just as an intent.

THE CONTRACT (fixed by the capability-router board, card t_aee95351 — the
format is the contract between this repo and the hermes-smart-router plugin;
do not reshape it unilaterally):

    {"schema": "route-attempts/1", "task_id": "...", "run_id": 1, "session_id": "…",
     "n": 1, "model": "…", "provider": "…", "started_at": 0.0,
     "duration_ms": 0, "outcome": "served|failed|skipped",
     "error": {"code": "…", "message": "…"}}

  * ``n`` is 1-based, one per hop the executor walked, in walk order.
  * ``outcome: served`` — this backend completed at least one successful
    model call for the run.
  * ``outcome: failed`` — ``error.code``/``error.message`` are mandatory.
  * ``outcome: skipped`` — no ``error``; a chain entry judged without a call.
  * Records land in ``attempts.jsonl`` beside the plugin's ``routes.jsonl``;
    the plugin's reader merges them into the matching decision entry by
    ``(task_id, run_id)`` and exposes the result as the decision's optional
    ``attempts`` field.

ACTIVATION IS EXPLICIT AND DOUBLE-GATED. Both must hold:

  1. ``HERMES_ROUTE_ATTEMPTS_FILE`` — absolute path of the journal file,
     published by the hermes-smart-router plugin (it owns the state dir).
  2. ``HERMES_KANBAN_TASK`` or ``HERMES_ROUTE_ATTEMPTS_KEY`` — a correlation
     key. Without one, the process is not a routed run (an interactive CLI
     chat, a gateway conversation) and there is no decision to attach to.

With the env unset — every stock upstream install — every entry point below
is a no-op returning ``None``/``False``: no file is created, nothing is
buffered, the hot paths pay one ``os.environ.get`` each. The capability is
core; the activation is the plugin's choice.

Stdlib-only, like the plugin's durable log: a bad import here must never
brick the agent, and the writer must never raise into a recovery path — every
public function swallows its own IO errors (best-effort by contract, the same
rule ``_record_fallback_on_session`` follows).

Line size: every record is capped (``_MAX_MESSAGE_CHARS`` on the error text)
so one JSON line stays under ``PIPE_BUF`` (4096 B) and a single ``write`` is
atomic under ``O_APPEND`` even with concurrent worker processes appending to
the same journal. The plugin's routes.jsonl needed an in-process lock because
its classifier payloads exceed ``PIPE_BUF``; these records deliberately do
not.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = "route-attempts/1"

# Keep one JSON line comfortably under PIPE_BUF (4096 B) so concurrent
# appenders never interleave a torn line. The other fields are bounded by
# nature (model/provider slugs, outcome enum); the error message is the only
# open-ended one, so it is the one capped. 240 chars: measured headroom —
# the longest record with a full error block is ~500 B even with long
# provider-qualified model names, half the budget.
_MAX_MESSAGE_CHARS = 240

# Same disk ceiling the plugin's routes.jsonl advertises per file. One backup
# (not three): attempts are an adjunct to the trace, and the reader keeps the
# last complete window; two files bound the journal at 10 MiB.
_JOURNAL_MAX_BYTES = 5 * 1024 * 1024

_STATE_LOCK = threading.Lock()
# Per-agent state, keyed by id(agent). ``agent`` itself stays as a strong
# reference so Python cannot reuse an id for another live agent and make its
# sequence continue the wrong route. Each state owns one currently-open hop
# plus the next 1-based ordinal; closing a hop never resets that ordinal.
_AGENT_STATES: Dict[int, Dict[str, Any]] = {}
_FLUSH_REGISTERED = False


def attempts_path() -> Optional[Path]:
    """Journal file when the writer gate is armed, else None.

    The gate is the env var itself — a path the PLUGIN resolved. Core never
    guesses a location: an upstream install must not grow surprise files, and
    the plugin (which already anchors routes.jsonl profile-independently) is
    the one authority on where state lives.
    """
    raw = os.environ.get("HERMES_ROUTE_ATTEMPTS_FILE", "").strip()
    if not raw:
        return None
    return Path(raw)


def attempts_key() -> str:
    """Correlation key for this process's records, or "" when not a routed run.

    ``HERMES_KANBAN_TASK`` first — the dispatcher sets it on every worker
    spawn, so the kanban path needs no plugin involvement beyond publishing
    the file gate. ``HERMES_ROUTE_ATTEMPTS_KEY`` is the plugin's minted key
    for non-kanban routed spawns (the delegate tool path).
    """
    task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if task:
        return task
    return os.environ.get("HERMES_ROUTE_ATTEMPTS_KEY", "").strip()


def _active() -> bool:
    """True only when both gates hold (file + a correlation key)."""
    return attempts_path() is not None and bool(attempts_key())


def _run_id() -> Optional[int]:
    raw = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _append(record: Dict[str, Any]) -> None:
    """One best-effort append. Never raises; never blocks the caller long.

    Rotation is one shift (current -> .1, old .1 unlinked) — not the plugin's
    cascade — because a single backup already bounds the journal and the
    reader only needs the most recent window.
    """
    path = attempts_path()
    if path is None:
        return
    try:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    except (TypeError, ValueError):
        return
    try:
        with _STATE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            if size and size + len(line) > _JOURNAL_MAX_BYTES:
                backup = path.with_suffix(path.suffix + ".1")
                try:
                    backup.unlink()
                except OSError:
                    pass
                try:
                    os.replace(path, backup)
                except OSError:
                    pass  # unrotatable: append anyway, the cap is soft
            with open(path, "ab") as handle:
                handle.write(line)
                handle.flush()
    except OSError:
        pass  # full disk / permissions — an attempt record is never worth a crash


def _record(hop: Dict[str, Any], outcome: str, *, error_code: str = "", error_message: str = "") -> Dict[str, Any]:
    record = {
        "schema": _SCHEMA,
        "task_id": attempts_key(),
        "run_id": _run_id(),
        "session_id": hop["session_id"],
        "n": hop["n"],
        "model": hop["model"],
        "provider": hop["provider"],
        "started_at": hop["started_at"],
        "duration_ms": max(0, int((time.time() - hop["started_at"]) * 1000)),
        "outcome": outcome,
    }
    if outcome == "failed":
        record["error"] = {
            "code": str(error_code or "unknown")[:64],
            "message": str(error_message or "")[:_MAX_MESSAGE_CHARS],
        }
    return record


def bind_hop(agent: Any) -> None:
    """Open an ordinalled hop on the agent's current backend, if gated.

    Rebinding the same backend does not create a second record. A fallback
    closes the old hop first, then calls this after its client swap: the next
    ordinal comes from state, not from the now-closed hop, preserving 1-based
    chain order across failures and locally skipped fallbacks.
    """
    if not _active():
        return
    model = str(getattr(agent, "model", "") or "")
    provider = str(getattr(agent, "provider", "") or "")
    displaced = None
    with _STATE_LOCK:
        state = _AGENT_STATES.setdefault(id(agent), {"agent": agent, "next_n": 1, "open": None})
        current = state["open"]
        if current is not None and current["model"] == model and current["provider"] == provider:
            return
        if current is not None and current["saw_success"]:
            # Defensive only: normal fallback closes before it swaps. Defer the
            # append until after releasing _STATE_LOCK because _append owns
            # that same lock for the rotate+write critical section.
            displaced = dict(current)
        state["open"] = {
            "n": state["next_n"],
            "model": model,
            "provider": provider,
            "session_id": str(getattr(agent, "session_id", "") or ""),
            "started_at": time.time(),
            "saw_success": False,
        }
        state["next_n"] += 1
    if displaced is not None:
        _append(_record(displaced, "served"))
    _register_flush_at_exit()


def note_served(agent: Any) -> None:
    """Mark the current hop as having returned a valid model response."""
    if not _active():
        return
    with _STATE_LOCK:
        state = _AGENT_STATES.get(id(agent))
        if state and state["open"] is not None:
            state["open"]["saw_success"] = True


def close_hop(agent: Any, outcome: str, *, error_code: str = "", error_message: str = "") -> None:
    """Close and append the current hop. Recovery bookkeeping never raises."""
    if not _active():
        return
    with _STATE_LOCK:
        state = _AGENT_STATES.get(id(agent))
        hop = state["open"] if state else None
        if state:
            state["open"] = None
    if hop is None:
        return
    if outcome == "served" and not hop["saw_success"]:
        outcome = "failed"
        error_code = error_code or "closed_without_success"
    _append(_record(hop, outcome, error_code=error_code, error_message=error_message))


def note_skip(agent: Any, provider: str, model: str) -> None:
    """Append a planned chain hop the executor rejected without a call.

    The fixed public shape has no skip reason or error payload: a skip is a
    planning outcome, not an API failure. Logging the reason elsewhere is fine;
    putting it in this contract would make readers choose between two meanings
    of ``error``.
    """
    if not _active():
        return
    with _STATE_LOCK:
        state = _AGENT_STATES.setdefault(id(agent), {"agent": agent, "next_n": 1, "open": None})
        n = state["next_n"]
        state["next_n"] += 1
    now = time.time()
    _append({
        "schema": _SCHEMA, "task_id": attempts_key(), "run_id": _run_id(),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "n": n, "model": str(model or ""), "provider": str(provider or ""),
        "started_at": now, "duration_ms": 0, "outcome": "skipped",
    })


def _register_flush_at_exit() -> None:
    """Arm the atexit flush once per process (idempotent)."""
    global _FLUSH_REGISTERED
    if _FLUSH_REGISTERED:
        return
    _FLUSH_REGISTERED = True
    atexit.register(flush_all)


def flush_all() -> None:
    """Write every open hop that served. Called at process exit.

    A hop abandoned by crash, kill, or a run that never completed a model
    call writes NOTHING — an attempt with unknown duration and no outcome is
    not a fact, and absence is the honest record (the same rule the routing
    trace applies to un-instrumented history).
    """
    with _STATE_LOCK:
        open_items = []
        for state in _AGENT_STATES.values():
            hop = state.get("open")
            if hop and hop.get("saw_success"):
                open_items.append(dict(hop))
            state["open"] = None
    for hop in open_items:
        _append(_record(hop, "served"))


__all__ = [
    "attempts_path",
    "attempts_key",
    "bind_hop",
    "note_served",
    "note_skip",
    "close_hop",
    "flush_all",
]
