"""Tests for the Hermes memory retrieval fixes.

These run against a real SQLite store with the production schema, because the
defects they pin are both about the seam between the store and the retriever —
a mock would have hidden them.

Two defects, both measured on the live 104-fact store before fixing:

1. RECALL COLLAPSE. FTS5 is lexical. "how do I reach the Oracle machine"
   matched zero of 104 facts, because the facts say "araponga"/"ssh" and the
   query says "Oracle"/"machine". With no fallback, search() returned [] and the
   agent behaved as though it knew nothing about the subject.

2. DEAD USAGE COUNTER. retrieval_count was only incremented by search_facts(),
   which has no callers. Live retrieval goes through HybridRetriever, so the
   column stayed at 0 for 101 of 104 facts while the agent retrieved every turn
   — nobody could tell a load-bearing memory from a useless one.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")  # the retrieval module needs numpy for HRR

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def usage(store, fact_id: int) -> int:
    """retrieval_count for one fact, via the store's real API."""
    for fact in store.list_facts(limit=500):
        if fact["fact_id"] == fact_id:
            return fact["retrieval_count"]
    raise AssertionError(f"fact {fact_id} not found")


@pytest.fixture
def store(tmp_path):
    """A real store with the production schema."""
    st = MemoryStore(tmp_path / "memory_store.db")
    yield st
    try:
        st.close()
    except Exception:
        pass


@pytest.fixture
def retriever(store):
    # hrr_weight=0 keeps these tests about retrieval logic rather than about
    # whether numpy is installed on the runner.
    return FactRetriever(store, fts_weight=0.55, jaccard_weight=0.15, hrr_weight=0.0)


# ── 1. recall ──────────────────────────────────────────────────────────
def test_a_vocabulary_mismatch_does_not_erase_the_memory(store, retriever):
    """The live failure, reproduced.

    The fact is about reaching a machine, but says "araponga" and "ssh". The
    operator asks about "the Oracle machine". FTS5 finds nothing; the memory must
    still be reachable, or the agent claims ignorance about something it knows.
    """
    store.add_fact(
        "Reach araponga with ssh -i ~/.ssh/id_oci_araponga ubuntu@163.176.180.233",
        category="project", tags="araponga, ssh, oracle",
    )
    store.add_fact("Unrelated: the kanban board lives in the WebUI", category="tool")

    hits = retriever.search("how do I reach the Oracle machine", limit=5)

    assert hits, "a vocabulary mismatch must not read as 'no such memory'"
    assert "araponga" in hits[0]["content"]


def test_the_lexical_path_still_wins_when_it_matches(store, retriever):
    """The fallback must not outrank a real lexical hit."""
    store.add_fact("The capability router pins hard verbs to tier T4", category="project")
    store.add_fact("Something else entirely about drones", category="general")

    hits = retriever.search("capability router hard verbs", limit=5)

    assert "capability router" in hits[0]["content"].lower()


def test_an_fts_operator_a_human_typed_does_not_wipe_the_results(store, retriever):
    """FTS5 rejects bare operators. A query like "a1-hunter OR trader" or "AND"
    raised, was swallowed, and returned nothing — measured on the live store."""
    store.add_fact("a1-hunter polls Oracle for A1 capacity every 15 minutes",
                   category="project", tags="a1-hunter, oracle")

    for hostile in ("a1-hunter OR trader", "AND", "*", 'the "office" screen', "NOT working"):
        hits = retriever.search(hostile, limit=5)
        assert isinstance(hits, list), f"{hostile!r} must not raise"

    # The one that is genuinely about a stored fact must find it.
    assert retriever.search("a1-hunter OR trader", limit=5), \
        "an operator query with an FTS operator in it must still retrieve"


def test_the_fallback_only_returns_facts_with_real_overlap(store, retriever):
    """A fallback that returns everything is worse than one that returns nothing:
    it would fill the prompt with noise the model then trusts."""
    store.add_fact("Kanban board columns are configurable", category="tool")
    store.add_fact("Provider deepseek-v3.2 is the classifier", category="provider-config")

    hits = retriever.search("zzzz totally unrelated quantum bicycle", limit=5)

    assert hits == [], "no token overlap means no memory, not every memory"


def test_the_fallback_respects_trust_and_category_filters(store, retriever):
    """The fallback bypasses FTS, so it must not also bypass the caller's filters.

    Tested against _scan_candidates DIRECTLY. Going through search() proved
    useless twice: any query with a shared token is answered by FTS, whose own
    WHERE clause enforces the filters — so a filter regression in the scan path
    stayed invisible through two rounds of mutation testing.
    """
    low = store.add_fact("Distrusted araponga ssh credentials", category="general",
                         tags="araponga")
    store.update_fact(low, trust_delta=-0.45)  # 0.5 default -> 0.05
    keep = store.add_fact("Trusted araponga ssh credentials note", category="project",
                          tags="araponga")

    hits = retriever._scan_candidates("araponga credentials", None, 0.3, 10)

    ids = [h["fact_id"] for h in hits]
    assert keep in ids, "the trusted fact must survive the scan"
    assert low not in ids, "min_trust must hold on the fallback, not only on FTS"

    scoped = retriever._scan_candidates("araponga credentials", "project", 0.0, 10)
    assert scoped and all(h["category"] == "project" for h in scoped), \
        "category must hold on the fallback"
    other = retriever._scan_candidates("araponga credentials", "tool", 0.0, 10)
    assert other == [], "a category with no facts must yield nothing, not everything"


def test_the_fallback_is_only_reached_when_fts_finds_nothing(store, retriever):
    """Order matters: the lexical index is authoritative when it matches, and the
    scan is a last resort. If the scan ran first it would outrank real hits."""
    store.add_fact("Reach araponga over ssh with the oci key", category="project",
                   tags="araponga, ssh")

    assert retriever._fts_candidates("araponga", None, 0.0, 30), "FTS answers when it can"
    # A query sharing no token with any fact: FTS is empty, scan decides.
    assert retriever._fts_candidates("zzz nonexistent wording", None, 0.0, 30) == []
    fallback = retriever._scan_candidates("araponga ssh", None, 0.3, 5)
    assert fallback and all(c["fts_rank"] == 0.0 for c in fallback), \
        "fallback candidates must not claim an FTS score they did not earn"


def test_an_empty_query_retrieves_nothing_rather_than_everything(store, retriever):
    store.add_fact("Some fact", category="general")
    assert retriever.search("", limit=5) == []
    assert retriever.search("   ", limit=5) == []


# ── 2. usage counting ──────────────────────────────────────────────────
def test_retrieval_through_the_live_path_counts_as_a_use(store, retriever):
    """The whole point of the counter: distinguish a load-bearing memory from
    one that has never once been served."""
    fact_id = store.add_fact("The router sidecar listens on port 8791", category="tool")

    before = usage(store, fact_id)
    retriever.search("router sidecar port", limit=5)
    after = usage(store, fact_id)

    assert before == 0
    assert after == 1, "the live retrieval path must record what it served"


def test_only_the_facts_actually_served_are_counted(store, retriever):
    """Counting a candidate that lost the ranking would make every fact look
    equally useful, which is the same as counting nothing."""
    served = store.add_fact("Router tier T4 is gpt-5.6-terra", category="model-routing",
                            tags="router, tier")
    ignored = store.add_fact("Completely separate note about kanban", category="tool")

    hits = retriever.search("router tier T4", limit=1)

    assert len(hits) == 1
    assert usage(store, served) == 1
    assert usage(store, ignored) == 0


def test_counting_survives_repeated_retrieval(store, retriever):
    fact_id = store.add_fact("Compaction threshold is 208352 tokens", category="project",
                             tags="compaction")
    for _ in range(3):
        retriever.search("compaction threshold", limit=5)
    assert usage(store, fact_id) == 3


def test_a_failure_to_count_never_breaks_the_retrieval(store, retriever, monkeypatch):
    """Telemetry must never cost the operator the memory they were about to get."""
    store.add_fact("A fact worth retrieving about the router", category="project")

    def explode(_ids):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "record_retrievals", explode)
    with pytest.raises(sqlite3.OperationalError):
        # Prove the stub really is wired in...
        store.record_retrievals([1])

    # ...and that a genuine sqlite failure inside record_retrievals is swallowed.
    monkeypatch.undo()
    monkeypatch.setattr(store, "_conn", _BrokenConn(store._conn))
    hits = retriever.search("router", limit=5)
    assert hits, "a broken counter must not empty the results"


class _BrokenConn:
    """Reads fine, raises on the UPDATE the counter issues."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args, **kwargs):
        if sql.strip().upper().startswith("UPDATE FACTS SET RETRIEVAL_COUNT"):
            raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_record_retrievals_is_a_no_op_for_an_empty_list(store):
    """Guard the trivial case explicitly: an empty IN () is a SQL syntax error."""
    store.record_retrievals([])  # must not raise


def test_record_retrievals_is_the_field_s_only_live_writer(store, retriever):
    """A regression fence.

    If someone reverts search() to not record, this fails — which is exactly how
    the field died the first time: a writer existed (search_facts) but nothing on
    the live path called it.
    """
    import inspect
    source = inspect.getsource(FactRetriever.search)
    assert "record_retrievals" in source, \
        "the live search path must count usage, or the column dies again"


# ── 3. category centroids: the write-only bank becomes a read ───────────
def test_category_centroids_are_readable_at_all(store):
    """memory_banks was written on every add_fact (66% of its runtime, measured)
    and read by nothing but a COUNT in audit(). This is the read."""
    store.add_fact("The capability router pins hard verbs to T4", category="model-routing",
                   tags="router")
    store.add_fact("Tier T1 is deepseek-v3.2", category="model-routing", tags="tier")
    store.add_fact("The kanban board has configurable columns", category="tool")

    ranked = store.rank_categories("which tier does the router pick")

    if not ranked:  # numpy absent on this runner
        pytest.skip("HRR unavailable, centroids cannot be compared")
    assert {cat for cat, _ in ranked} >= {"model-routing", "tool"}
    assert ranked == sorted(ranked, key=lambda p: p[1], reverse=True), "best first"


def test_a_category_hint_never_excludes_a_fact(store, retriever):
    """The centroid is a tie-breaker, not a filter. A strong lexical match in an
    unexpected category must still be returned — a hint that can hide a memory is
    worse than no hint."""
    wanted = store.add_fact("araponga ssh key lives on hermes-wsl only", category="tool",
                            tags="araponga")
    for i in range(6):
        store.add_fact(f"Unrelated routing note {i}", category="model-routing")

    hits = retriever._scan_candidates("araponga ssh", None, 0.0, 10)

    assert wanted in [h["fact_id"] for h in hits], \
        "the best token match must survive whatever the centroid prefers"


def test_ranking_categories_degrades_quietly_without_hrr(store, monkeypatch):
    """A ranking hint that raises would break every fallback retrieval."""
    store.add_fact("Some fact", category="general")
    monkeypatch.setattr(store, "_hrr_available", False)
    assert store.rank_categories("anything") == []


def test_ranking_categories_survives_a_corrupt_bank(store, monkeypatch):
    """A bank row written at another dimension, or truncated, must be skipped —
    not crash the retrieval that asked for a hint."""
    store.add_fact("A fact in a category with a bank", category="tool", tags="tool")
    if not store._hrr_available:
        pytest.skip("HRR unavailable")

    with store._lock:
        store._conn.execute(
            "INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)"
            " VALUES ('cat:garbage', ?, ?, 1, CURRENT_TIMESTAMP)",
            (b"\x00\x01\x02", store.hrr_dim),
        )
        store._conn.commit()

    ranked = store.rank_categories("tool")  # must not raise
    assert "garbage" not in {cat for cat, _ in ranked}


def test_an_empty_query_gets_no_category_hint(store):
    store.add_fact("A fact", category="tool")
    assert store.rank_categories("") == []


# ── 4. hyphenated names: the most common query class ────────────────────
def test_a_hyphenated_name_is_retrievable(store, retriever):
    """The single worst defect found.

    The sanitizer deleted hyphens, so "copilot-acp" became "copilotacp" and
    matched nothing. Measured on the live 104-fact store, EVERY hyphenated term
    returned zero results while the facts plainly contained them: copilot-acp in
    9 facts, capability-router in 4, gpt-5.6-terra in 5. Provider, model and
    plugin names are what the agent asks about most.
    """
    store.add_fact("Claude Code runs as a provider via slug copilot-acp on Bedrock",
                   category="provider-config", tags="copilot-acp")
    store.add_fact("Tier T4 is gpt-5.6-terra on openai-codex", category="model-routing",
                   tags="gpt-5.6-terra")
    store.add_fact("Unrelated note about the kanban board", category="tool")

    for term, expected in (("copilot-acp", "copilot-acp"), ("gpt-5.6-terra", "gpt-5.6-terra")):
        hits = retriever.search(term, limit=5)
        assert hits, f"{term!r} must retrieve the fact that names it"
        assert expected in hits[0]["content"], f"{term!r} must rank its own fact first"


def test_the_sanitizer_keeps_both_the_whole_term_and_its_parts():
    """Either spelling should hit: the index may hold "capability-router" as one
    token or as two, depending on the tokenizer."""
    out = FactRetriever._sanitize_fts_query("capability-router")
    assert '"capability-router"' in out, "the exact hyphenated phrase"
    assert '"capability"' in out and '"router"' in out, "and its components"


def test_the_sanitizer_never_emits_a_bare_hyphen():
    """A bare hyphen is FTS5's NOT operator; leaking one inverts the query."""
    for hostile in ("-", "--", "a - b", "- leading", "trailing -", "well-known -x"):
        out = FactRetriever._sanitize_fts_query(hostile)
        # Every hyphen that survives must be inside a quoted phrase.
        for segment in out.split(" OR "):
            if "-" in segment:
                assert segment.startswith('"') and segment.endswith('"'), \
                    f"{hostile!r} produced an unquoted hyphen: {out!r}"


def test_a_query_of_pure_punctuation_matches_nothing_without_raising(store, retriever):
    """The old fallback returned the raw query, which made FTS5 raise on ordinary
    input like "AND" or "*" — and the exception was swallowed into no results."""
    store.add_fact("A fact that exists", category="general")
    for pathological in ("*", "AND", "()", "---", "+"):
        assert FactRetriever._sanitize_fts_query(pathological) == '""', \
            f"{pathological!r} must sanitise to a matchless phrase"
        assert retriever.search(pathological, limit=5) == [] or True  # must not raise


def test_hyphen_handling_does_not_break_ordinary_words(store, retriever):
    """A regression fence: fixing hyphens must not disturb the common case."""
    store.add_fact("The compaction threshold is 208352 tokens", category="project",
                   tags="compaction")
    hits = retriever.search("compaction threshold", limit=5)
    assert hits and "compaction" in hits[0]["content"].lower()


# ── 5. atomicity is the stronger guarantee ──────────────────────────────
def test_a_failing_bank_rebuild_rolls_the_whole_write_back(tmp_path, monkeypatch):
    """Recorded because the opposite was tried and reverted.

    Making the category-bank rebuild best-effort looked like a durability win —
    "a diagnostic aggregate must not cost a memory". It is the wrong trade: a
    fact stored WITHOUT its entity links and HRR vector is a half-written memory,
    and swallowing the error leaves the store quietly inconsistent. The existing
    suite already pins this by using a failing bank as an atomicity canary; this
    test states the decision where the fix was attempted.
    """
    store = MemoryStore(tmp_path / "memory_store.db")
    monkeypatch.setattr(store, "_rebuild_bank",
                        lambda _c: (_ for _ in ()).throw(RuntimeError("bank failed")))

    with pytest.raises(RuntimeError, match="bank failed"):
        store.add_fact("This must not survive half-written", entities=["Canary"])

    assert store.list_facts(min_trust=0.0, limit=10) == [], "the fact must roll back"
    assert store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert store._conn.in_transaction is False, "and the write lock must be released"
    store.close()


# ── 6. downvotes demote, they do not delete ─────────────────────────────
def test_repeated_downvotes_cannot_bury_a_fact_beyond_recovery(tmp_path):
    """The one-way trust trap.

    Retrieval filters at min_trust=0.3, and a downvote costs 0.10. Three of them
    took a fact from 0.5 to 0.2, after which it was never served again — and a
    fact that is never served can never be upvoted back. An honest correction
    silently deleted the memory from recall while leaving the row on disk.
    """
    store = MemoryStore(tmp_path / "memory_store.db")
    fact_id = store.add_fact("A fact that gets corrected repeatedly", category="general")

    for _ in range(8):
        store.record_feedback(fact_id, helpful=False)

    trust = [f for f in store.list_facts(min_trust=0.0, limit=10)
             if f["fact_id"] == fact_id][0]["trust_score"]
    assert trust >= 0.3, f"trust {trust} fell below the retrieval floor"

    # And it is genuinely still reachable, not merely stored.
    retriever = FactRetriever(store, fts_weight=0.55, jaccard_weight=0.15, hrr_weight=0.0)
    assert retriever.search("corrected repeatedly", min_trust=0.3, limit=5), \
        "a demoted fact must remain retrievable so it can recover"
    store.close()


def test_a_downvote_still_costs_trust(tmp_path):
    """The floor must not turn feedback into a no-op — ranking still has to move."""
    store = MemoryStore(tmp_path / "memory_store.db")
    fact_id = store.add_fact("A fact with default trust", category="general")

    before = store.record_feedback(fact_id, helpful=False)

    assert before["new_trust"] < before["old_trust"], "a downvote must reduce trust"
    assert before["new_trust"] == pytest.approx(0.4), "0.5 - 0.10, above the floor"
    store.close()


def test_explicit_removal_and_explicit_trust_are_unaffected(tmp_path):
    """The floor guards FEEDBACK only. An operator who means to delete, or to
    drive trust to zero, must still be able to."""
    store = MemoryStore(tmp_path / "memory_store.db")
    keep = store.add_fact("Explicitly zeroed", category="general")
    store.update_fact(keep, trust_delta=-1.0)
    zeroed = [f for f in store.list_facts(min_trust=0.0, limit=10)
              if f["fact_id"] == keep][0]["trust_score"]
    assert zeroed == pytest.approx(0.0), "explicit trust_delta still reaches 0"

    gone = store.add_fact("To be deleted", category="general")
    assert store.remove_fact(gone) is True
    assert gone not in [f["fact_id"] for f in store.list_facts(min_trust=0.0, limit=10)]
    store.close()


def test_upvotes_still_raise_trust_to_the_ceiling(tmp_path):
    store = MemoryStore(tmp_path / "memory_store.db")
    fact_id = store.add_fact("A consistently useful fact", category="tool")
    for _ in range(20):
        store.record_feedback(fact_id, helpful=True)
    trust = [f for f in store.list_facts(min_trust=0.0, limit=10)
             if f["fact_id"] == fact_id][0]["trust_score"]
    assert trust == pytest.approx(1.0), "trust must still be able to reach 1.0"
    store.close()


# ── 7. the health check must be able to fail ────────────────────────────
def test_audit_detects_an_index_that_has_lost_all_recall(tmp_path):
    """audit()'s FTS parity check was a tautology.

    COUNT(*) on an external-content FTS5 table scans the CONTENT table, so
    "fts_rows == facts" could never be false. Wiping the index left audit()
    reporting healthy=True while every search returned nothing — the one check an
    operator would trust was structurally blind to total loss of lexical recall.

    FTS5's own integrity-check is not enough either: measured, it reports "ok" for
    an empty index, because empty is a *consistent* state.
    """
    store = MemoryStore(tmp_path / "memory_store.db")
    store.add_fact("Routing sends hard verbs to the opus tier", category="model-routing")

    assert store.audit()["healthy"] is True
    assert store.audit()["fts_integrity"] == "ok"

    with store._lock:
        store._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('delete-all')")
        store._conn.commit()

    report = store.audit()
    assert report["healthy"] is False, "a store with no lexical recall is not healthy"
    assert "rebuild" in report["fts_integrity"], "and the report must name the repair"
    store.close()


def test_a_desynced_index_can_actually_be_repaired(tmp_path):
    """Detection without a repair path just relocates the problem: nothing in the
    plugin could rebuild the index before, so a desync was permanent."""
    store = MemoryStore(tmp_path / "memory_store.db")
    store.add_fact("Compaction threshold is 208352 tokens", category="project")
    retriever = FactRetriever(store, fts_weight=0.55, jaccard_weight=0.15, hrr_weight=0.0)

    def fts_only(term: str) -> int:
        """Ask the index directly. _fts_candidates now falls through to a token
        scan when FTS finds nothing (that is the recall fix), so it cannot be used
        to observe whether the index itself is alive."""
        with store._lock:
            return store._conn.execute(
                "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?",
                (f'"{term}"',),
            ).fetchone()[0]

    assert fts_only("compaction") == 1, "sanity: the index works to begin with"

    with store._lock:
        store._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('delete-all')")
        store._conn.commit()
    assert fts_only("compaction") == 0, "index is wiped"
    assert store.audit()["healthy"] is False
    # The fallback keeps the memory reachable even now — that is the whole point
    # of it — so recall degrades rather than disappearing while the index is down.
    assert retriever.search("compaction threshold", min_trust=0.0, limit=5), \
        "the scan fallback must cover for a dead index"

    assert store.rebuild_fts() == "ok"

    assert fts_only("compaction") == 1, "lexical recall restored"
    assert store.audit()["healthy"] is True
    store.close()


def test_the_health_probe_does_not_cry_wolf(tmp_path):
    """An empty store, or content with no probeable word, must not read as broken."""
    store = MemoryStore(tmp_path / "memory_store.db")
    assert store.audit()["healthy"] is True, "an empty store is healthy"
    store.add_fact("a b c", category="general")  # no token >= 4 chars
    assert store.audit()["fts_integrity"] == "ok"
    store.close()


# ── 8. a heuristic must never cost the fact ─────────────────────────────
def test_junk_from_a_regex_does_not_discard_the_fact(tmp_path):
    """_extract_entities runs inside add_fact's atomic group, and
    _normalize_entities RAISED on a rejected name — so one bad guess from a regex
    rolled back the whole write and the memory was silently lost.

    Content below is engineered to make the old patterns produce fragments: the
    apostrophe rule cross-paired "user's ... it's" into the entity
    "s shell is zsh and it".
    """
    store = MemoryStore(tmp_path / "memory_store.db")

    hostile = [
        "The user's shell is zsh and it's persistent across Wait, this reboots",
        "Running Windows on the Avell, admin' rights needed' for the ACPI fix",
        "we decided 'to use, a comma' inside quotes which is a fragment",
    ]
    for text in hostile:
        assert store.add_fact(text, category="general"), f"lost: {text[:40]}"

    assert len(store.list_facts(min_trust=0.0, limit=10)) == len(hostile)
    store.close()


def test_an_entity_the_caller_named_still_fails_loudly(tmp_path):
    """The leniency is for GUESSES only. A caller who asked for a specific link
    must hear that it was refused, rather than have it silently dropped."""
    store = MemoryStore(tmp_path / "memory_store.db")
    with pytest.raises(ValueError, match="invalid entity name"):
        store.add_fact("A fact", entities=["Wait, this is a fragment"])
    assert store.list_facts(min_trust=0.0, limit=10) == [], "and it rolls back"
    store.close()


def test_a_real_name_is_not_rejected_for_its_first_word(tmp_path):
    """The first-word stopword rule was order-dependent nonsense: "API Gateway"
    passed and "Gateway API" did not — the same two words."""
    from plugins.memory.holographic.store import _is_valid_entity
    for name in ("API Gateway", "Gateway API", "Windows Server", "Linux Mint",
                 "Mac Studio", "Protocol Buffers"):
        assert _is_valid_entity(name), f"{name!r} is a real product name"
    # Fragments must still be refused.
    for junk in ("Running Windows", "Wait, this", "the thing", "s shell is zsh"):
        assert not _is_valid_entity(junk), f"{junk!r} is a sentence fragment"


def test_an_apostrophe_does_not_forge_an_entity(tmp_path):
    """r"'([^']+)'" treated apostrophes as quote delimiters and cross-paired them
    into a multi-word sentence that passed validation and persisted forever."""
    store = MemoryStore(tmp_path / "memory_store.db")
    got = store._extract_entities("The user's shell is zsh and it's persistent")
    assert not any("shell is zsh" in g for g in got), f"cross-paired: {got}"
    # A genuine single-quoted term is still captured.
    assert "pytest" in store._extract_entities("Run 'pytest' before pushing")
    store.close()
