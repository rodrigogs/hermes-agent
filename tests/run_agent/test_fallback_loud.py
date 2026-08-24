"""Loud model fallback: an unhonored model/provider request must be visible.

Regression cover for the silent-fallback failure mode reported 2026-08-17:

    hermes -m "openrouter/zai/glm-5.3" --provider openrouter -z "ping"

answered "pong" with **glm-5.2 via zai** — a different model from a different
provider — with no error, no warning, and a session row that claimed the served
model had been the requested one. Three things conspired:

1. ``hermes -z`` redirects stdout AND stderr to devnull for the whole run and
   sets ``suppress_status_output``, which short-circuits ``_vprint`` *before*
   its ``force`` check — so even the forced fallback notice was swallowed.
2. ``sessions.model`` holds a single (model, provider) pair, and
   ``update_token_counts``' first-accounted-route reconciliation overwrites it
   with the route that actually billed. The request was destroyed on the first
   API call.
3. Nothing anywhere persisted "what was asked for".

The tests below pin each half of the fix: the state layer keeps the request
beside the delivery, and one-shot says so out loud on the real stderr.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# State layer: the request survives the served route overwriting `model`
# ---------------------------------------------------------------------------

def test_requested_route_survives_first_accounted_route_overwrite(db):
    """The exact reported case: request openrouter, get billed by zai."""
    db.create_session(
        "s_fallback",
        source="cli",
        model="openrouter/zai/glm-5.3",
        requested_model="openrouter/zai/glm-5.3",
        requested_provider="openrouter",
    )

    # First accounted API call comes from the fallback route, not the request.
    db.update_token_counts(
        "s_fallback",
        input_tokens=10,
        output_tokens=5,
        model="glm-5.2",
        billing_provider="zai",
        api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_fallback")
    # Served route won the aggregate columns — that part is intended.
    assert row["model"] == "glm-5.2"
    assert row["billing_provider"] == "zai"
    # ...and the request is still recoverable, which is the whole point.
    assert row["requested_model"] == "openrouter/zai/glm-5.3"
    assert row["requested_provider"] == "openrouter"


def test_requested_route_is_never_rewritten_by_a_later_observation(db):
    """A second writer must not be able to relabel the original request."""
    db.create_session(
        "s_keep",
        source="cli",
        model="glm-5.3",
        requested_model="glm-5.3",
        requested_provider="zai",
    )
    # A lazy writer (update_token_counts' self-healing insert) re-enters the
    # row with a different model; COALESCE must leave the request alone.
    db.update_token_counts(
        "s_keep", input_tokens=1, model="deepseek-v4-flash",
        billing_provider="deepseek", api_call_count=1,
    )
    db.record_session_fallback(
        "s_keep", requested_model="something-else", requested_provider="nowhere",
    )
    db.flush_token_counts()

    row = db.get_session("s_keep")
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"


def test_fallback_flag_defaults_off_and_is_sticky(db):
    db.create_session(
        "s_flag", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    assert db.get_session("s_flag")["fallback_activated"] == 0

    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1
    # Idempotent — a second chain hop must not corrupt the flag.
    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1


def test_record_session_fallback_backfills_a_row_created_before_the_columns(db):
    """Rows that predate the audit columns still get a usable request."""
    db.create_session("s_backfill", source="cli", model="glm-5.2")
    db.record_session_fallback(
        "s_backfill", requested_model="gpt-5.6-sol", requested_provider="openrouter",
    )
    row = db.get_session("s_backfill")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "gpt-5.6-sol"
    assert row["requested_provider"] == "openrouter"


def test_fallback_backfill_never_pairs_a_new_request_with_an_old_provider(db):
    """Backfill completes the request pair as a unit, or leaves it alone.

    A provider-less ``/model`` switch records "no provider requested" (NULL) on
    purpose. Backfilling that half independently from the process-start
    snapshot would put the switched model beside the provider of a request that
    no longer exists — the same incoherent pair, through the back door. The
    snapshot only describes the ORIGINAL request, so it may only fill a row
    that has no request recorded at all.
    """
    db.create_session(
        "s_pair", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_pair", "gpt-5.4")  # provider-less switch

    # A fallback later in the same process still carries the init snapshot.
    db.record_session_fallback(
        "s_pair", requested_model="glm-5.3", requested_provider="zai",
    )

    row = db.get_session("s_pair")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "gpt-5.4"
    assert row["requested_provider"] is None


def test_record_session_fallback_tolerates_a_missing_row(db):
    """Never raise on the recovery path — the row is created lazily."""
    db.record_session_fallback("s_does_not_exist", requested_model="x")
    assert db.get_session("s_does_not_exist") is None


def test_explicit_model_switch_resets_the_request_audit(db):
    """A /model switch is a NEW request, so the stale flag must clear."""
    db.create_session(
        "s_switch", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.record_session_fallback("s_switch")
    db.update_session_model("s_switch", "deepseek-v4-flash", provider="deepseek")

    row = db.get_session("s_switch")
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] == "deepseek"
    assert row["fallback_activated"] == 0


def test_provider_less_switch_does_not_keep_the_previous_request_provider(db):
    """The audit pair must describe ONE request, not halves of two.

    ``/model deepseek-v4-flash`` without a provider asks for a model and
    nothing else. COALESCE-ing the provider half of the audit left the PREVIOUS
    request's provider standing beside the NEW model, so the row described a
    route nobody had ever asked for ("requested deepseek-v4-flash via zai") —
    and the `hermes sessions list` warning would have printed exactly that.
    "No provider requested" is NULL, not the last one.
    """
    db.create_session(
        "s_noprov", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.record_session_fallback("s_noprov")

    db.update_session_model("s_noprov", "deepseek-v4-flash")

    row = db.get_session("s_noprov")
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] is None
    assert row["fallback_activated"] == 0


def test_switch_audit_and_resume_route_have_separate_provider_semantics(db):
    """The audit column and ``model_config.$.provider`` are different things.

    The stored ``$.provider`` exists so a later resume recombines the model
    with the provider that serves it (#79536) and must survive a provider-less
    switch; the audit column must state what THIS request asked for. Lineage
    markers survive both, since the switch goes through the shared merge.
    """
    db.create_session(
        "s_cfg", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
        model_config={
            "provider": "custom:feather",
            "_branched_from": "parent-session",
            "_delegate_from": "boss-session",
        },
    )
    db.record_session_fallback("s_cfg")

    # Provider-less switch: the audit forgets the provider, the resume route
    # keeps it (the caller made no statement about routing).
    db.update_session_model("s_cfg", "deepseek-v4-flash")
    row = db.get_session("s_cfg")
    config = json.loads(row["model_config"])
    assert config["provider"] == "custom:feather"
    assert config["_branched_from"] == "parent-session"
    assert config["_delegate_from"] == "boss-session"
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] is None
    assert row["fallback_activated"] == 0

    # Provider-bearing switch: both halves move to the new request together.
    db.record_session_fallback("s_cfg")
    db.update_session_model("s_cfg", "glm-5.4", provider="zai")
    row = db.get_session("s_cfg")
    config = json.loads(row["model_config"])
    assert config["provider"] == "zai"
    assert config["_branched_from"] == "parent-session"
    assert config["_delegate_from"] == "boss-session"
    assert row["requested_model"] == "glm-5.4"
    assert row["requested_provider"] == "zai"
    assert row["fallback_activated"] == 0


def test_fallback_backfill_completes_the_provider_of_the_same_request(db):
    """Backfilling the pair as a unit is not the same as refusing to backfill.

    The rule is "one route, adopted whole" — so a snapshot naming the model the
    row already records is not a foreign route, it is the SAME route with the
    provider half known. Refusing it (because ``requested_model`` is merely
    non-NULL) drops the provider from the loud warning for every resumed
    session whose recorded model was genuinely re-requested via a real
    provider: `requested glm-5.3 → served grok-4 (xai)` instead of
    `requested glm-5.3 (zai) → ...`.
    """
    db.create_session(
        "s_same", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    # Provider-less /model switch back to the same model: "no provider
    # requested" is recorded, on purpose.
    db.update_session_model("s_same", "glm-5.3")
    assert db.get_session("s_same")["requested_provider"] is None

    # A later process re-requests that very model, this time knowing the
    # provider; the fallback snapshot may complete the pair it already names.
    db.record_session_fallback(
        "s_same", requested_model="glm-5.3", requested_provider="zai",
    )
    row = db.get_session("s_same")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"


def test_session_upsert_adopts_the_request_pair_only_as_a_unit(db):
    """``create_session``'s upsert is the third writer of the audit pair.

    Every process's first turn re-runs ``create_session`` for an existing
    session id — that is what the ``ON CONFLICT`` upsert is for — carrying THAT
    process's immutable start-of-run snapshot. COALESCE-ing the two halves
    independently lets the snapshot's provider land beside a model some earlier
    ``/model`` switch requested, describing a route nobody ever asked for.
    """
    # Row already records a provider-less request (a `/model` switch, or an
    # ad-hoc --base-url endpoint whose provider key cannot be recovered).
    db.create_session(
        "s_upsert", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_upsert", "gpt-5.4")

    # A later process's first turn, whose own request differs.
    db.create_session(
        "s_upsert", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    row = db.get_session("s_upsert")
    assert row["requested_model"] == "gpt-5.4"
    assert row["requested_provider"] is None, (
        "the upsert must not pair a foreign provider with the recorded model"
    )

    # A row with NEITHER half still adopts the snapshot's whole pair.
    db.create_session("s_bare", source="cli", model="glm-5.2")
    db.create_session(
        "s_bare", source="cli", model="glm-5.2",
        requested_model="glm-5.3", requested_provider="zai",
    )
    bare = db.get_session("s_bare")
    assert bare["requested_model"] == "glm-5.3"
    assert bare["requested_provider"] == "zai"

    # A row with BOTH halves is never rewritten by a later snapshot.
    db.create_session(
        "s_bare", source="cli", model="glm-5.2",
        requested_model="gpt-5.4", requested_provider="minimax",
    )
    kept = db.get_session("s_bare")
    assert kept["requested_model"] == "glm-5.3"
    assert kept["requested_provider"] == "zai"


# ---------------------------------------------------------------------------
# The whole state machine: 8 row states (4 pairs x 2 flag values) x 3 writers
# ---------------------------------------------------------------------------

#: ``(row state, row pair, snapshot pair, expected pair after the write)`` for
#: a row whose ``fallback_activated`` is DOWN.
#: Both snapshot-carrying writers (``create_session``'s ON CONFLICT upsert and
#: ``record_session_fallback``'s backfill) share one gate, so one table pins
#: both. The three earlier rounds of this fix each enumerated one axis short:
#: two enumerated three of the four pair states and shipped the bug living in
#: the fourth, and the third left the flag axis (below) unswept entirely.
_AUDIT_PAIR_TABLE = [
    # Nothing recorded: the snapshot's request is adopted as a whole pair,
    # whatever shape it has.
    ("neither", (None, None), ("glm-5.4", "minimax"), ("glm-5.4", "minimax")),
    ("neither", (None, None), (None, "minimax"), (None, "minimax")),
    ("neither", (None, None), (None, None), (None, None)),
    # Model only: the recorded model can be newer than any snapshot (a /model
    # switch writes it mid-run), so it wins — and its NULL provider means "no
    # provider requested", which only a snapshot naming that SAME model may
    # fill in.
    ("model only", ("gpt-5.4", None), ("glm-5.4", "minimax"), ("gpt-5.4", None)),
    ("model only", ("gpt-5.4", None), ("gpt-5.4", "minimax"), ("gpt-5.4", "minimax")),
    ("model only", ("gpt-5.4", None), (None, "minimax"), ("gpt-5.4", None)),
    # Provider only (`hermes --provider vllm`, no model.default): a request
    # that never named a model, and one no switch can have written. A snapshot
    # that names a model supersedes it as a whole pair — the model may not be
    # stitched onto the stored provider, and refusing the model would strand
    # the warning on a bare provider from an abandoned start.
    ("provider only", (None, "vllm"), ("glm-5.4", "minimax"), ("glm-5.4", "minimax")),
    ("provider only", (None, "vllm"), ("glm-5.4", "vllm"), ("glm-5.4", "vllm")),
    # A snapshot naming no model adds nothing, so the row stands.
    ("provider only", (None, "vllm"), (None, "minimax"), (None, "vllm")),
    ("provider only", (None, "vllm"), (None, None), (None, "vllm")),
    # Both halves: a complete request is never rewritten by a later snapshot.
    ("both", ("gpt-5.4", "vllm"), ("glm-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), ("gpt-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), (None, None), ("gpt-5.4", "vllm")),
]


#: Row states are the TRIPLE, so every case above is played twice: once against
#: a row carrying no verdict and once against a row whose ``fallback_activated``
#: is already up. The flag is not a fourth independent column — it is the
#: verdict on the pair beside it ("the request these two name was abandoned") —
#: so what a writer may do to the pair depends on it, and vice versa.
_ROW_FLAGS = [0, 1]


def _setup_row(db, session_id, row, flag):
    """Put the row into one of the eight ``(pair, flag)`` states, and prove it.

    The flag is raised with a snapshot-less ``record_session_fallback``, which
    is a pure flag raise from every pair state (pinned by
    ``test_a_bare_record_session_fallback_raises_only_the_flag``) — so the row
    really is the state the parametrization claims, with no pair write smuggled
    into the setup.
    """
    db.create_session(
        session_id, source="cli", model="glm-5.2",
        requested_model=row[0], requested_provider=row[1],
    )
    if flag:
        db.record_session_fallback(session_id)
    stored = db.get_session(session_id)
    assert (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    ) == (row[0], row[1], flag), "could not set up the row state"


def _assert_one_whole_request(result, *candidates):
    """The stored TRIPLE must be one whole record, never a mix of two.

    The pair rule ("one of the two requests in play, never the model of one
    beside the provider of another") extended to the verdict: a request and the
    verdict reached about it are one record, so ``result`` must equal one of the
    whole records the writer had to choose between — pair and flag together.

    Completing a half is not an exception: a half may only be taken from a
    snapshot that agrees with the row about the half they share, so every legal
    completion equals the snapshot's own pair. And splicing is not a harmless
    approximation in either direction — a flag from record A over pair B cries
    wolf about a request that was honored, while a pair from B over a flag
    belonging to A drops the provider of the request that really was abandoned.
    """
    assert result in candidates, (
        f"{result} is not one of the whole records in play: {candidates}"
    )


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row,snapshot,expected",
    _AUDIT_PAIR_TABLE,
    ids=[
        f"{s}-{'x'.join(str(v) for v in snap)}"
        for s, _row, snap, _exp in _AUDIT_PAIR_TABLE
    ],
)
def test_upsert_audit_pair_table(db, state, row, snapshot, expected, flag):
    """``create_session``'s ON CONFLICT upsert, over the whole state machine.

    The flag axis splits every case in two. ``create_session``'s snapshot is a
    process START — a request that has not been answered yet, let alone
    abandoned — so this writer may never move a raised verdict:

    * flag DOWN: no verdict to contradict, the ordinary pair gate applies, and
      an adopted snapshot arrives with the only verdict a just-made request can
      carry, ``0``. This writer never touches the flag, so that is automatic.
    * flag UP: the stored pair is what the verdict is ABOUT, so it is frozen
      whole. Freezing loses nothing — should this snapshot's own request also be
      abandoned, ``record_session_fallback`` restates pair and flag together at
      the moment the new pair becomes a true statement.
    """
    _setup_row(db, "s_tbl", row, flag)

    # The next process's first turn, carrying its own start-of-run snapshot.
    db.create_session(
        "s_tbl", source="cli", model="glm-5.2",
        requested_model=snapshot[0], requested_provider=snapshot[1],
    )
    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    assert result == (
        (row[0], row[1], 1) if flag else (expected[0], expected[1], 0)
    )
    # The verdict is never invented and never discarded by this writer: it may
    # keep the row's whole record, or take a snapshot's pair with the flag down.
    _assert_one_whole_request(result, (row[0], row[1], flag), (*snapshot, 0))


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row,snapshot,expected",
    _AUDIT_PAIR_TABLE,
    ids=[
        f"{s}-{'x'.join(str(v) for v in snap)}"
        for s, _row, snap, _exp in _AUDIT_PAIR_TABLE
    ],
)
def test_fallback_backfill_audit_pair_table(db, state, row, snapshot, expected, flag):
    """``record_session_fallback``'s backfill, over the whole state machine.

    Same pair expectations as the upsert — both writers carry a process-start
    snapshot into a row they did not write, so they answer to one gate — but
    deliberately NO flag guard, and the incoming flag changes nothing. This
    writer's call site is ``try_activate_fallback`` swapping away from the live
    route, so it is the one writer that may move pair and raised flag together:
    whichever pair an arm takes, that pair is being abandoned as we write, and
    the ``= 1`` in the same statement is the verdict on it. Superseding an
    already-flagged pair swaps one truthfully flagged request for another, which
    is the best a single-pair row can do; the arms that keep a recorded model
    already cover the case where the live route is not this snapshot.
    """
    _setup_row(db, "s_tbl", row, flag)
    db.record_session_fallback(
        "s_tbl", requested_model=snapshot[0], requested_provider=snapshot[1],
    )
    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    assert result == (expected[0], expected[1], 1), (
        "the incoming flag must not change the backfill's pair gate"
    )
    assert stored["fallback_activated"] == 1, "the flag is the point of the call"
    # Either request may end up stored, but the verdict this call asserts
    # applies to whichever one does.
    _assert_one_whole_request(result, (row[0], row[1], 1), (*snapshot, 1))


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row",
    [(state, row) for state, row, _snap, _exp in _AUDIT_PAIR_TABLE],
    ids=[
        f"{state}-{'x'.join(str(v) for v in row)}"
        for state, row, _snap, _exp in _AUDIT_PAIR_TABLE
    ],
)
@pytest.mark.parametrize(
    "switch", [("glm-5.4", "minimax"), ("glm-5.4", None), ("", "minimax")]
)
def test_update_session_model_audit_pair_table(db, state, row, switch, flag):
    """``update_session_model``: the third writer, from all eight row states.

    A /model switch is a new explicit request, so it writes all THREE columns
    from THIS call — which makes it coherent from every prior state by
    construction, including the provider-only one and including either incoming
    flag. Nothing is coalesced in, so nothing can be mixed in. Clearing the flag
    is legitimate here precisely because the request it judged is being
    discarded in the same statement; the upsert, which discards nothing, must
    freeze a flagged pair instead. The empty-model case pins the ``or None``
    normalization: '' and None both mean "no model requested" and must be stored
    identically, or the NULL gates in the other two writers would read '' as a
    recorded name.
    """
    _setup_row(db, "s_tbl", row, flag)
    db.update_session_model("s_tbl", switch[0], provider=switch[1])

    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    assert result == (switch[0] or None, switch[1], 0)
    # This writer has exactly one legal outcome: its own request, unjudged.
    _assert_one_whole_request(result, (switch[0] or None, switch[1], 0))


def test_audit_columns_are_declared_so_existing_dbs_reconcile(tmp_path):
    """The columns are declarative: an older DB gains them on next open."""
    import sqlite3

    from hermes_state_common import SCHEMA_SQL

    declared = SessionDB._parse_schema_columns(SCHEMA_SQL)["sessions"]
    for column in ("requested_model", "requested_provider", "fallback_activated"):
        assert column in declared, column

    # Build a DB, drop the columns out of the picture by recreating the table
    # without them, then reopen: _reconcile_columns must ADD them back.
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
        "model TEXT, started_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = SessionDB(path)
    try:
        live = {
            row[1]
            for row in db._conn.execute('PRAGMA table_info("sessions")').fetchall()
        }
        assert {"requested_model", "requested_provider", "fallback_activated"} <= live
    finally:
        db.close()


# ---------------------------------------------------------------------------
# One-shot: the warning reaches the real stderr, stdout stays clean
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, requested_model, requested_provider, fallback):
        self.origin_requested_model = requested_model
        self.origin_requested_provider = requested_provider
        self._fallback_activated = fallback


def test_annotate_requested_route_reads_the_immutable_snapshot():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    # try_activate_fallback reassigns requested_provider to the fallback; the
    # audit must not read that attribute.
    agent.requested_provider = "zai"
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    assert result["requested_model"] == "openrouter/zai/glm-5.3"
    assert result["requested_provider"] == "openrouter"
    assert result["fallback_activated"] is True


def test_no_warning_when_the_requested_model_answered():
    from hermes_cli import oneshot

    agent = _FakeAgent("glm-5.3", "zai", False)
    result = {"model": "glm-5.3", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)
    assert oneshot._fallback_warning_line(result) is None


def test_warning_names_both_routes():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    line = oneshot._fallback_warning_line(result)
    assert line is not None
    assert line.endswith("\n")
    assert "openrouter/zai/glm-5.3 via openrouter" in line
    assert "glm-5.2 via zai" in line
    # The whole failure mode was that a wrong-model answer looked normal.
    assert "SERVED" in line


def test_warning_survives_a_missing_request_half():
    from hermes_cli import oneshot

    line = oneshot._fallback_warning_line(
        {"model": "glm-5.2", "provider": "zai", "fallback_activated": True}
    )
    assert line is not None
    assert "an unknown model" in line


def test_oneshot_writes_the_warning_to_the_real_stderr(monkeypatch, capsys):
    """The end-to-end guarantee: -z can no longer answer 200-quiet.

    ``run_oneshot`` swallows every byte the agent writes; only the final
    response reaches stdout. This asserts the fallback notice takes the real
    stderr path instead, and that stdout stays exactly the response.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="openrouter/zai/glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == "pong\n"
    assert "openrouter/zai/glm-5.3" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_nonexistent_requested_model_still_names_it_in_the_warning(
    monkeypatch, capsys
):
    """Requesting a model that does not exist may not answer quietly either.

    The 2026-08-17 reproduction included ``openai/gpt-5.6-sol`` — a model id
    that does not exist — which was answered by glm-5.2/zai with no signal.
    Whatever the reason a request cannot be honored (unreachable provider,
    bad id, exhausted quota), the fallback notice must name the id exactly as
    typed, so a pipeline pinning a model can see the pin was not honored.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openai/gpt-5.6-sol", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot(
        "ping", model="openai/gpt-5.6-sol", provider="openrouter"
    )
    captured = capsys.readouterr()

    assert rc == 0  # a fallback that answered is not an error...
    assert captured.out == "pong\n"  # ...but stdout stays machine-readable
    # and the nonexistent id is named verbatim on the real stderr.
    assert "openai/gpt-5.6-sol via openrouter" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_provider_without_credentials_fails_loudly(monkeypatch, capsys):
    """No credentials must never become a quiet answer from an unrelated model.

    When there is no fallback chain to absorb the failure, the run must exit
    non-zero with an explicit error — the same loud contract as the fallback
    warning, on the branch where nothing answered at all.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        raise RuntimeError(
            "openrouter: AuthenticationError: no API key configured"
        )

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""  # no answer at all — never an unrelated model
    assert "agent failed" in captured.err
    assert "AuthenticationError" in captured.err


def test_oneshot_usage_file_records_request_and_delivery(tmp_path, monkeypatch):
    """Pipelines get the audit in machine-readable form."""
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot(
        "ping",
        model="openrouter/zai/glm-5.3",
        provider="openrouter",
        usage_file=str(usage),
    )

    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["requested_model"] == "openrouter/zai/glm-5.3"
    assert report["requested_provider"] == "openrouter"
    assert report["fallback_activated"] is True
    assert report["model"] == "glm-5.2"


def test_usage_file_marks_an_honored_request_as_not_fallen_back(tmp_path, monkeypatch):
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.3",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(_FakeAgent("glm-5.3", "zai", False), result)
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot("ping", model="glm-5.3", provider="zai", usage_file=str(usage))
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["fallback_activated"] is False


# ---------------------------------------------------------------------------
# Agent init: the snapshot is taken, and fallback cannot overwrite it
# ---------------------------------------------------------------------------

def test_fallback_swap_leaves_the_origin_snapshot_intact():
    """try_activate_fallback rewrites requested_provider; not the audit."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record_session_fallback(self, session_id, **kwargs):
            self.calls.append((session_id, kwargs))

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    agent.session_id = "s1"
    agent._session_db = _Recorder()
    _record_fallback_on_session(agent)

    assert agent._session_db.calls == [
        (
            "s1",
            {
                "requested_model": "openrouter/zai/glm-5.3",
                "requested_provider": "openrouter",
            },
        )
    ]


def test_record_fallback_on_session_never_raises():
    """A bookkeeping failure must not abort provider recovery."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Exploding:
        def record_session_fallback(self, *a, **k):
            raise RuntimeError("db is locked")

    agent = _FakeAgent("glm-5.3", "zai", True)
    agent.session_id = "s1"
    agent._session_db = _Exploding()
    _record_fallback_on_session(agent)  # must not raise

    # No session_db / no session_id are also non-events.
    bare = _FakeAgent("glm-5.3", "zai", True)
    bare.session_id = None
    bare._session_db = None
    _record_fallback_on_session(bare)


# ---------------------------------------------------------------------------
# `hermes sessions list` names the divergence
# ---------------------------------------------------------------------------

def test_sessions_list_reports_flagged_rows(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings([
        {
            "id": "20260817_191805_ec4afa",
            "model": "glm-5.2",
            "billing_provider": "zai",
            "requested_model": "openrouter/zai/glm-5.3",
            "requested_provider": "openrouter",
            "fallback_activated": 1,
        },
        {"id": "ok", "model": "glm-5.3", "fallback_activated": 0},
    ])
    out = capsys.readouterr().out
    assert "20260817_191805_ec4afa" in out
    assert "openrouter/zai/glm-5.3 (openrouter)" in out
    assert "glm-5.2 (zai)" in out
    assert "ok" not in out.replace("20260817_191805_ec4afa", "")


def test_sessions_list_stays_quiet_when_nothing_fell_back(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings(
        [{"id": "s1", "model": "glm-5.3", "fallback_activated": 0}]
    )
    assert capsys.readouterr().out == ""


def test_sessions_list_warns_off_real_listing_rows(db, capsys):
    """Same warning, driven by the real listing query instead of hand-made dicts.

    The tests above feed ``_print_fallback_warnings`` literal dicts, so they
    would keep passing if the listing SELECT stopped carrying the audit
    columns. This walks the whole path: request persisted at creation, served
    route overwriting ``model``, listing row read back, warning printed — and
    then a ``/model`` switch making the warning stop, since it is a new request.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_listed", source="cli", model="openrouter/zai/glm-5.3",
        requested_model="openrouter/zai/glm-5.3", requested_provider="openrouter",
    )
    db.record_session_fallback("s_listed")
    db.update_token_counts(
        "s_listed", input_tokens=10, output_tokens=5,
        model="glm-5.2", billing_provider="zai", api_call_count=1,
    )
    db.flush_token_counts()

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_listed"]
    assert rows, "the flagged session must be listable"
    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "openrouter/zai/glm-5.3 (openrouter)" in out
    assert "glm-5.2 (zai)" in out

    # A provider-less /model switch is a new request: the warning stops, and
    # nothing may reintroduce the abandoned provider as the requested one.
    db.update_session_model("s_listed", "glm-5.4")
    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_listed"]
    assert rows[0]["requested_model"] == "glm-5.4"
    assert rows[0]["requested_provider"] is None
    sessions_cmd._print_fallback_warnings(rows)
    assert capsys.readouterr().out == ""


def test_next_process_first_turn_cannot_make_the_warning_lie(db, capsys):
    """The printed warning must never name a route nobody asked for.

    Walks the three writes a real session takes, off a real DB, the real
    listing SELECT and the real printer:

    1. request glm-5.3 via zai, then a provider-less ``/model gpt-5.4`` —
       the audit pair becomes ``gpt-5.4`` / NULL ("no provider requested").
    2. the served route (grok-4 via xai) is accounted and the fallback flag is
       raised, carrying the process's original glm-5.3/zai snapshot.
    3. the NEXT process's first turn re-runs ``create_session`` for the same id
       with its own snapshot (``hermes --resume -m glm-5.4 --provider
       minimax`` skips the model restore, so the snapshot need not match the
       row).

    The upsert used to complete the pair's provider half from step 3's
    snapshot, and `hermes sessions list` printed
    ``requested gpt-5.4 (minimax) → served grok-4 (xai)``. Nobody ever
    requested gpt-5.4 via minimax.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_third_writer", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_third_writer", "gpt-5.4")

    db.update_token_counts(
        "s_third_writer", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_third_writer", requested_model="glm-5.3", requested_provider="zai",
    )

    db.create_session(
        "s_third_writer", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [
        s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_third_writer"
    ]
    assert rows, "the flagged session must be listable"
    assert rows[0]["requested_model"] == "gpt-5.4"
    assert rows[0]["requested_provider"] is None

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested gpt-5.4 → served grok-4 (xai)" in out
    assert "minimax" not in out


def test_provider_only_row_cannot_lend_its_provider_to_a_foreign_model(db, capsys):
    """The mirror of the above: a stored PROVIDER must not adopt a foreign model.

    A provider-only row (``requested_provider`` set, ``requested_model`` NULL)
    is ordinary production state, not a test poke: ``hermes --provider vllm``
    with no ``model.default`` in config leaves ``self.model == ""``, so
    ``agent_init`` snapshots an empty requested model while
    ``requested_provider`` has an ``"auto"`` floor and is effectively never
    empty. The row is written before the first API call (the titler forces
    ``_ensure_db_session``), so it survives even when that model-less request
    400s.

    The next process on that session id then arrives with its own complete
    request. Independent ``COALESCE``s stitched THAT process's model onto the
    stored provider and `hermes sessions list` printed

        s1  requested glm-5.4 (vllm) → served grok-4 (xai)

    Nobody ever requested that: the first process asked for vllm with no model,
    the second asked for glm-5.4 via minimax. Double harm — the stale ``vllm``
    also suppressed the correct ``minimax`` the backfill would have supplied.
    """
    from hermes_cli import sessions_cmd

    # First process: `hermes --provider vllm` with no default model.
    db.create_session(
        "s_provider_only", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    row = db.get_session("s_provider_only")
    assert row["requested_model"] is None
    assert row["requested_provider"] == "vllm"

    # Next process's first turn: a whole request of its own.
    db.create_session(
        "s_provider_only", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    # ...which is not honored: another route serves and bills the turn.
    db.update_token_counts(
        "s_provider_only", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_provider_only", requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [
        s for s in db.list_sessions_rich(limit=10)
        if s["id"] == "s_provider_only"
    ]
    assert rows, "the flagged session must be listable"
    assert rows[0]["requested_model"] == "glm-5.4"
    assert rows[0]["requested_provider"] == "minimax", (
        "the model came from this snapshot, so the provider must too"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested glm-5.4 (minimax) → served grok-4 (xai)" in out
    assert "vllm" not in out


def test_fallback_backfill_alone_cannot_mispair_a_provider_only_row(db):
    """The same mis-pairing arises purely inside ``record_session_fallback``.

    No second ``create_session`` needed: the process that starts provider-only
    and falls back on its very first turn backfills through this writer, whose
    model half was an ungated ``COALESCE`` too.
    """
    db.create_session(
        "s_backfill_only", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_backfill_only", requested_model="glm-5.4", requested_provider="minimax",
    )
    row = db.get_session("s_backfill_only")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "glm-5.4"
    assert row["requested_provider"] == "minimax"


# ---------------------------------------------------------------------------
# The flag is part of the record: a raised flag is a verdict ON the stored pair
# ---------------------------------------------------------------------------

def _provider_only_flagged_then_honored_run(db, *, account_the_abandoned_route):
    """Set up the two-process history both flag tests share.

    P1 = ``hermes --provider vllm`` with no ``model.default``: a provider-only
    row is written before the first API call, that model-less request is
    abandoned by ``try_activate_fallback`` (flag up), and the fallback route
    either does or does not manage to bill a turn.

    P2 = ``hermes -c -m glm-5.4 --provider minimax`` on the same session id,
    honored end to end: its own request is what serves and bills.
    """
    db.create_session(
        "s_flagged", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_flagged", requested_model=None, requested_provider="vllm",
    )
    flagged = db.get_session("s_flagged")
    assert (
        flagged["requested_model"],
        flagged["requested_provider"],
        flagged["fallback_activated"],
    ) == (None, "vllm", 1), "P1 must leave a flagged provider-only row"

    if account_the_abandoned_route:
        db.update_token_counts(
            "s_flagged", input_tokens=10, output_tokens=5,
            model="grok-4", billing_provider="xai", api_call_count=1,
        )
        db.flush_token_counts()

    # P2's first turn re-runs create_session with its own start-of-run snapshot.
    db.create_session(
        "s_flagged", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    db.update_token_counts(
        "s_flagged", input_tokens=10, output_tokens=5,
        model="glm-5.4", billing_provider="minimax", api_call_count=1,
    )
    db.flush_token_counts()

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_flagged"]
    assert rows, "the flagged session must be listable"
    return rows


def test_flagged_row_does_not_cry_wolf_about_a_request_that_was_honored(db, capsys):
    """A raised flag may not be handed to a request nobody abandoned.

    The upsert learned (round 3) to let a snapshot's whole pair supersede a
    provider-only row's request — correct while the row carries no verdict, but
    that row is exactly the shape ``hermes --provider vllm`` leaves behind when
    its model-less request is the one that got abandoned, and the proof sits in
    the same row: ``fallback_activated = 1``. Superseding the pair without
    reading the flag left P1's verdict standing over P2's request, and
    `hermes sessions list` printed

        s1  requested glm-5.4 (minimax) → served glm-5.4 (minimax)

    announcing that a session "ran a model other than the one requested" about
    a request whose requested and served routes are character-for-character
    identical — the wolf-cry the sticky flag exists to prevent. It also erased
    ``vllm``, the provider of the request that actually WAS abandoned, from the
    row and from the output; before round 3 it was at least still printed.
    """
    from hermes_cli import sessions_cmd

    rows = _provider_only_flagged_then_honored_run(
        db, account_the_abandoned_route=False
    )
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (None, "vllm", 1), (
        "the flag is a verdict on P1's request, so P1's request must stay"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    # The truth: the vllm request that named no model was abandoned, and the
    # session went on to run glm-5.4 via minimax.
    assert "requested vllm → served glm-5.4 (minimax)" in out
    assert "requested glm-5.4 (minimax) → served glm-5.4 (minimax)" not in out


def test_flagged_row_keeps_the_provider_of_the_request_that_was_abandoned(db, capsys):
    """Same history, but the abandoned route billed a turn before P2 arrived.

    The wolf-cry above needs the served columns to coincide with P2's request;
    with P1's fallback route accounted, the served half is grok-4 via xai and
    the printed line stopped looking self-contradictory while saying something
    worse: ``requested glm-5.4 (minimax) → served grok-4 (xai)`` asserts that
    P2's request — honored end to end — was not honored. The flag belongs to
    P1's vllm request, and so must the requested half.
    """
    from hermes_cli import sessions_cmd

    rows = _provider_only_flagged_then_honored_run(
        db, account_the_abandoned_route=True
    )
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (None, "vllm", 1)

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested vllm → served grok-4 (xai)" in out
    assert "requested glm-5.4 (minimax)" not in out


def test_the_guard_does_not_silence_the_next_requests_own_fallback(db, capsys):
    """Freezing a flagged pair against snapshots must not hide a real fallback.

    The guard above refuses ``create_session``'s snapshot because a process
    START is a request that has not been answered yet, let alone abandoned. If
    that request IS later abandoned, the writer whose call site asserts the
    abandonment — ``record_session_fallback`` — restates the pair and the flag
    together, so the warning still names P2's route, provider included. The
    verdict and the request it judges always move as one record.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_two_falls", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_two_falls", requested_model=None, requested_provider="vllm",
    )
    db.create_session(
        "s_two_falls", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    # P2's own request is not honored either: it falls back to grok-4 via xai.
    db.update_token_counts(
        "s_two_falls", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_two_falls", requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_two_falls"]
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == ("glm-5.4", "minimax", 1)
    sessions_cmd._print_fallback_warnings(rows)
    assert (
        "requested glm-5.4 (minimax) → served grok-4 (xai)"
        in capsys.readouterr().out
    )


def test_the_guard_does_not_drop_the_provider_from_a_reasserted_request(db):
    """The round-2 completion still reaches the warning, via the right writer.

    A flagged ``model only`` row plus a snapshot naming that very model is the
    case round 2 added the completion arm for: dropping the provider would print
    ``requested glm-5.3 → served ...`` for a session that genuinely re-requested
    glm-5.3 through a known provider. The upsert now declines it — the stored
    verdict is about the provider-less request the row records — but the
    fallback backfill supplies it the moment the re-request is itself abandoned,
    which is the only moment at which the completed pair is a true statement.
    """
    db.create_session(
        "s_complete", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider=None,
    )
    db.record_session_fallback("s_complete")  # flag only, pair untouched

    # Next process's first turn re-requests glm-5.3, this time naming zai.
    db.create_session(
        "s_complete", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    frozen = db.get_session("s_complete")
    assert frozen["requested_provider"] is None, (
        "while the flag is up, the verdict's own request may not be re-labelled"
    )

    # ...and that re-request is abandoned too, so the pair is restated whole.
    db.record_session_fallback(
        "s_complete", requested_model="glm-5.3", requested_provider="zai",
    )
    row = db.get_session("s_complete")
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"
    assert row["fallback_activated"] == 1


def test_a_bare_record_session_fallback_raises_only_the_flag(db):
    """The flag-only setup the table tests rely on must really be flag-only.

    ``record_session_fallback(sid)`` with no snapshot is how a caller with no
    request knowledge flags a row (and how ``_record_fallback_on_session``
    behaves when the origin snapshot is empty). It must leave the pair exactly
    as it stands from every row state — otherwise the flagged rows in the table
    below would not be the row states they claim to be.
    """
    for name, pair in (
        ("neither", (None, None)),
        ("model only", ("gpt-5.4", None)),
        ("provider only", (None, "vllm")),
        ("both", ("gpt-5.4", "vllm")),
    ):
        sid = f"s_bare_{name.replace(' ', '_')}"
        db.create_session(
            sid, source="cli", model="glm-5.2",
            requested_model=pair[0], requested_provider=pair[1],
        )
        db.record_session_fallback(sid)
        row = db.get_session(sid)
        assert (row["requested_model"], row["requested_provider"]) == pair, name
        assert row["fallback_activated"] == 1, name
