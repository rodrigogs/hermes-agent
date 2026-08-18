"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
# Feedback-driven trust bottoms out here rather than at 0.0.
#
# Retrieval filters at min_trust=0.3 by default, so three downvotes took a fact
# from 0.5 to 0.2 and it was never served again — and a fact that is never served
# can never be upvoted back. That is a one-way trap: an honest correction of a
# once-wrong memory permanently deleted it from recall while leaving the row on
# disk, so nobody could see what had been lost. The floor keeps a distrusted fact
# reachable (it still ranks last, since score scales with trust) so the trap
# becomes a demotion.
#
# Deliberate removal is unaffected: remove_fact() deletes, and update_fact() can
# still drive trust to 0.0 explicitly via trust_delta.
_TRUST_FEEDBACK_MIN = 0.3
# How many facts audit() probes for lexical recall. One was not enough: with a
# single probe, destroying 19 of 20 facts' index rows still reported healthy.
_FTS_PROBE_SAMPLES = 8
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# A single capitalised word, but NOT at the start of a sentence or line — that
# position is grammatical capitalisation, not a name, and accepting it produced
# 446 one-off junk entities ("Full", "Keep", "Please", "Todos") on the live
# store. Requiring a preceding word means "on Bedrock" is captured while
# "Bedrock (Mac, ...)" at position 0 is left to the other patterns.
_RE_PROPER_NOUN  = re.compile(r'(?<=[a-z,;)]\s)([A-Z][a-z]{2,})\b')
# Technical identifiers the agent is asked about by name: copilot-acp,
# gpt-5.6-terra, us-west-2, glm-4.7-flash, hermes-delegate-profile, z.ai.
# Captures the WHOLE dotted/hyphenated run — an earlier attempt let \b match
# after a hyphen and truncated "us-west-2" to "west-2", "glm-4.7-flash" to
# "glm-4.7".
_RE_SLUG = re.compile(r'(?<![\w.-])([a-z][a-z0-9]*(?:[-.][a-z0-9]+)+)(?![\w-])')

# Hyphenated English compounds share the slug SHAPE but name no entity. A
# heuristic (digit present, part count, dot count) was tried and misjudged in
# both directions — it rejected "openai-codex" and accepted "read-only" — so the
# distinction is drawn explicitly. Anything unlisted is treated as a name, which
# is the right default for a store whose subject IS technical identifiers.
_SLUG_STOPWORDS = frozenset({
    "self-hosted", "self-host", "zero-key", "end-to-end", "access-verified",
    "high-volume", "family-estimated", "best-value", "not-trust",
    "do-not-trust", "evidence-based", "always-in-context", "last-resort",
    "read-only", "write-only", "host-key", "ff-only", "up-to-date",
    "out-of-date", "brave-free", "well-known", "free-models-per-day",
    "case-insensitive", "cross-platform", "open-source", "real-time",
    "per-day", "per-turn", "per-call", "one-way", "two-way", "built-in",
    "opt-in", "opt-out", "fail-closed", "fail-open", "read-write",
    "auto-extraction", "in-context", "non-claude", "e2e",
})
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
# Require a word boundary outside both quotes, so an apostrophe inside a word
# ("user's", "it's") cannot pair with the next one. Verified on the live shape:
# "The user's shell is zsh and it's persistent" previously yielded the entity
# "s shell is zsh and it", which validated and persisted permanently.
_RE_SINGLE_QUOTE = re.compile(r"(?<![\w'])'([^'\n]{2,40})'(?![\w'])")
# Bounded on both sides. The original `(\w+(?:\s+\w+)*)` backtracks quadratically:
# measured at 1873ms on a 2000-word fact and 730ms on a 5000-char run of one
# letter, all of it inside add_fact's transaction, holding the write lock. An alias
# is a name — a few short words — never a paragraph, so capping the run at 4 words
# of <=30 chars loses nothing real and brings both cases under 10ms.
_RE_AKA          = re.compile(
    r'((?:\w{1,30}\s+){0,3}\w{1,30})\s+(?:aka|also known as)\s+'
    r'((?:\w{1,30}\s+){0,3}\w{1,30})',
    re.IGNORECASE,
)

# Common sentence-initial / auxiliary words that _RE_CAPITALIZED wrongly
# promotes to entities when a sentence starts with them (e.g. "Running
# Windows ...", "Admin Windows ..."). Also generic status/verb words that
# are never real entities on their own.
_ENTITY_STOPWORDS = frozenset({
    "running", "admin", "created", "wait", "waiting", "note", "status",
    "the", "this", "that", "when", "where", "while", "after", "before",
    "then", "now", "also", "using", "used", "set", "get", "run", "ok",
    "pass", "fail", "done", "todo", "fixme", "warning", "error", "info",
    "started", "stopped", "enabled", "disabled", "blocked", "suspicious",
    "protocol", "gateway", "gate", "unknown", "windows", "linux", "mac",
    # Fragments of multi-word names the single-word pattern also matches:
    # "Claude Code" yields "Code", "Hermes One" yields "One". Both are captured
    # whole by _RE_CAPITALIZED, so the fragment is a duplicate that splits one
    # entity's facts across two nodes. Measured on the live store: "Code" 14
    # facts and "one" 11, shadowing "Claude Code" and "Hermes One".
    "code", "one", "two", "web", "full", "keep", "final", "other", "only",
    "canonical", "highest", "cost", "please", "commit", "patches", "discovery",
    "config", "detalhes", "busca", "todos", "zerar", "teste", "fluxo", "limite",
})
# Only these may not START an entity. Verbs, auxiliaries and sentence glue — a
# name beginning with one is a fragment. Product-name heads (windows, linux, mac,
# gateway, protocol, status) stay OUT of this set and remain in the all-words
# check below, so "Windows Server" is a name while "Windows" alone is not.
_ENTITY_HEAD_STOPWORDS = frozenset({
    "running", "admin", "created", "wait", "waiting", "note", "using", "used",
    "set", "get", "run", "started", "stopped", "enabled", "disabled", "blocked",
    "the", "this", "that", "when", "where", "while", "after", "before", "then",
    "now", "also", "ok", "pass", "fail", "done", "todo", "fixme",
    # Possessive fragments left by an apostrophe split.
    "s", "t", "re", "ll", "ve", "d", "m",
})

# Shell / command noise: if a quoted term looks like a command line or
# contains shell metacharacters, it's not an entity. NOTE: does NOT reject
# a plain '.' — identifiers like "glm-5.2" and "Z.AI" are valid entities.
# Rejects real shell noise: pipes, redirects, subshells, flags (--foo),
# absolute paths (/usr/bin/x), env assignment.
_RE_SHELL_NOISE = re.compile(r'[|&;$<>(){}\[\]\\]|(?:^|\s)--?\w|/\w+/|\s=\s')


def _is_valid_entity(name: str) -> bool:
    """Reject junk entity candidates (sentence fragments, commands, noise).

    Keeps real multi-word proper nouns ("Parrot Mambo", "Gaming Center") and
    identifiers ("glm-5.2", "obdive") while dropping the garbage the old
    unfiltered regexes produced (46% of entities were noise pre-2026-07).
    """
    n = name.strip()
    if not (2 <= len(n) <= 40):
        return False
    low = n.lower()
    # a comma anywhere -> sentence fragment, not an entity ("Wait, thats ...")
    if "," in n:
        return False
    # Hyphenated English compounds look like slugs but name nothing.
    if low in _SLUG_STOPWORDS:
        return False
    # strip surrounding punctuation from each word before stopword checks
    words = [re.sub(r"^[\W_]+|[\W_]+$", "", w) for w in low.split()]
    words = [w for w in words if w]
    # A leading VERB means a sentence fragment ("Running Windows ..."). A leading
    # noun does not: rejecting the whole set made "Gateway API", "Windows Server",
    # "Linux Mint" and "Mac Studio" invalid while "API Gateway" passed — the same
    # two words, accepted or refused by order alone.
    if words and words[0] in _ENTITY_HEAD_STOPWORDS:
        return False
    # every word is a stopword (e.g. "Running Windows") -> junk
    if words and all(w in _ENTITY_STOPWORDS for w in words):
        return False
    # punctuation-only or ellipsis
    if re.fullmatch(r"[\W_]+", n) or n.endswith("..."):
        return False
    # shell/command noise (quoted command lines, flags, paths)
    if _RE_SHELL_NOISE.search(n):
        return False
    # must contain at least one letter
    if not re.search(r"[A-Za-z]", n):
        return False
    return True


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # Exposed so a test can assert it is NOT relying on the stopword list when it
    # means to test the sentence-position guard. Those two rejections overlap, and
    # a test that confuses them passes against a broken pattern.
    _ENTITY_STOPWORDS_FOR_TEST = _ENTITY_STOPWORDS

    # --- Process-wide shared connection registry -------------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist in one process (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All instances for the same database now share ONE connection and
    # ONE re-entrant lock, so access is fully serialized and cross-connection
    # contention is impossible. The shared connection is refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY

        # Acquire (or open) the process-wide shared connection for this DB.
        # resolve() (not just expanduser) so symlinked/relative paths to the
        # same file share ONE connection instead of silently reintroducing
        # the multi-writer contention this registry exists to prevent.
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Autocommit: every statement is its own transaction, so a
                    # write that raises mid-method can never leave a dangling
                    # transaction (and its write lock) open. The explicit
                    # commit() calls below become harmless no-ops.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                entry = {
                    "conn": conn,
                    "lock": threading.RLock(),
                    "refs": 0,
                    "ready": False,
                    "atomic_depth": 0,
                }
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._entry.setdefault("atomic_depth", 0)
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        # Initialise the schema once per shared connection.
        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        # Temporal anchoring (SOTA: Mem0 self-editing pattern)
        if "superseded_by" not in columns:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN superseded_by INTEGER REFERENCES facts(fact_id)"
            )
        if "superseded_at" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN superseded_at TIMESTAMP")
        self._commit_if_needed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def _atomic(self):
        """Run a write group as one SQLite transaction under the shared lock."""
        with self._lock:
            outermost = self._entry["atomic_depth"] == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._entry["atomic_depth"] += 1
            try:
                yield
            except Exception:
                self._entry["atomic_depth"] -= 1
                if outermost:
                    self._conn.rollback()
                raise
            else:
                self._entry["atomic_depth"] -= 1
                if outermost:
                    self._conn.commit()

    def _commit_if_needed(self) -> None:
        """Commit standalone writes, but never split an active atomic group."""
        if self._entry["atomic_depth"] == 0:
            self._conn.commit()

    def _get_fact_content(self, fact_id: int) -> str | None:
        """Return the content string for a fact, or None if not found."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()
            return row["content"] if row else None
        except Exception:
            return None

    def mark_superseded(self, fact_id: int, by_fact_id: int) -> bool:
        """Mark a fact as superseded by a newer version (temporal anchoring).

        The old fact is kept for audit/history but excluded from retrieval
        by default. This is Mem0's UPDATE pattern: the old fact persists
        with a pointer to its replacement.
        """
        with self._atomic():
            row = self._conn.execute(
                "SELECT fact_id FROM facts WHERE fact_id = ? AND superseded_by IS NULL",
                (fact_id,),
            ).fetchone()
            if row is None:
                return False  # already superseded or doesn't exist
            self._conn.execute(
                "UPDATE facts SET superseded_by = ?, superseded_at = CURRENT_TIMESTAMP"
                " WHERE fact_id = ?",
                (by_fact_id, fact_id),
            )
            self._commit_if_needed()
            return True

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        entities: list[str] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._atomic():
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                self._commit_if_needed()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Explicit entities are authoritative. Heuristic extraction remains
            # the fallback for callers that do not provide them.
            #
            # A name the CALLER chose is validated strictly — a refused link is
            # something they must hear about. A name a REGEX guessed is not: this
            # runs inside the atomic group, so raising on one junk candidate rolled
            # back the fact itself. Losing a memory because a heuristic misfired is
            # the worst possible trade.
            explicit = entities is not None
            entity_names = entities if explicit else self._extract_entities(content)
            for name in self._normalize_entities(entity_names, strict=explicit):
                entity_id = self._resolve_entity(name)
                self._set_entity_aliases(entity_id, self._aliases_for(name, aliases))
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            # HRR bank rebuild removed (2026-08-04): measured 66% of add_fact wall
            # time with 3.8% category-identification accuracy — worse than uniform
            # guess. The dead weight is noted in rank_categories() docstring.

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._commit_if_needed()

            return results

    def rank_categories(self, query: str, dim: int | None = None) -> list[tuple[str, float]]:
        """Rank categories by how much a query resembles each one's centroid.

        ``memory_banks`` holds one bundled HRR vector per category, rebuilt on
        every write — 66% of add_fact's time, measured.

        MEASURED QUALITY, so nobody has to guess: asking each of 104 real facts
        to identify its own category from its own content, this ranked the true
        category first 3.8% of the time, against 14.3% for a uniform guess over
        7 categories. Bundling hundreds of HRR vectors into a single superposition
        destroys the signal. It is therefore NOT used to influence retrieval; it
        is kept as a diagnostic, and the write cost it imposes is a standing
        argument for either raising hrr_dim or dropping the banks entirely.

        Returns ``(category, similarity)`` best-first, or [] when HRR is
        unavailable or no bank exists. Never raises: a ranking hint that fails
        must degrade to "no hint", not break retrieval.
        """
        if not self._hrr_available or not query:
            return []
        dim = dim or self.hrr_dim
        try:
            query_vec = hrr.encode_text(query, dim)
        except Exception:
            return []

        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT bank_name, vector, dim FROM memory_banks"
                ).fetchall()
            except sqlite3.Error:
                return []

        ranked: list[tuple[str, float]] = []
        for row in rows:
            # A bank built at a different dimension cannot be compared; skip it
            # rather than producing a meaningless number.
            if row["dim"] != dim:
                continue
            name = str(row["bank_name"] or "")
            category = name[4:] if name.startswith("cat:") else name
            try:
                sim = hrr.similarity(query_vec, hrr.bytes_to_phases(row["vector"]))
            except Exception:
                continue
            ranked.append((category, float(sim)))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def record_retrievals(self, fact_ids: list[int]) -> None:
        """Count that these facts were actually served to the model.

        ``search_facts`` already did this, but nothing calls it: live retrieval
        goes through ``HybridRetriever``, which issues its own SELECT. The result
        was a usage column that stayed at zero (3 of 104 facts on the live store)
        while the agent retrieved on every turn — so nobody could tell a
        load-bearing memory from one that has never once been useful.

        Best-effort by design, and NON-BLOCKING. This runs on the prefetch path,
        which used to be a pure read — and in WAL mode a reader is never blocked by
        a writer, but a WRITER is. Measured: with one concurrent add_fact process,
        a prefetch-shaped search went from 2.7ms to 10020ms, the entire
        busy_timeout, and then swallowed the error so the counter it waited for was
        never even recorded. Waiting ten seconds to write telemetry, and losing the
        telemetry anyway, is the worst of both.

        So the counter takes the lock only if it is free, and abandons the count
        otherwise. A usage statistic is allowed to be approximate; a turn is not
        allowed to hang.
        """
        if not fact_ids:
            return
        # Coercion is INSIDE the guard: int(None) and int("x") raise TypeError and
        # ValueError, neither of which is a sqlite3.Error, so a malformed id would
        # have propagated out of search() and killed the turn that was about to use
        # the memory — the exact opposite of what this method promises.
        try:
            ids = [int(f) for f in fact_ids]
        except (TypeError, ValueError):
            return
        # Never queue behind another writer: give up immediately if the in-process
        # lock is held, and tell SQLite not to wait either.
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._conn.execute("PRAGMA busy_timeout = 0")
            # Chunked: SQLite's variable limit is build-dependent (999 on older
            # builds, 32766 on 3.53) and a single IN (...) over a large result set
            # trips "too many SQL variables".
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                placeholders = ",".join("?" * len(chunk))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1"
                    f" WHERE fact_id IN ({placeholders})",
                    chunk,
                )
            self._commit_if_needed()
        except sqlite3.Error:
            # SQLITE_BUSY lands here now instead of after a ten-second wait. The
            # count is telemetry, not the memory: losing one is invisible, and
            # stalling the turn that was about to use the memory is not.
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass
        finally:
            # Restore the timeout every other caller depends on.
            try:
                self._conn.execute("PRAGMA busy_timeout = 10000")
            except sqlite3.Error:
                pass
            self._lock.release()

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
        entities: list[str] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._atomic():
            row = self._conn.execute(
                "SELECT fact_id, trust_score, category FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._commit_if_needed()

            # Entity bindings are authoritative persistent data. Replace them
            # only when the caller explicitly supplies ``entities``; changing
            # content/tags/trust must never silently re-run heuristics and erase
            # prior explicit links.
            if entities is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._normalize_entities(entities):
                    entity_id = self._resolve_entity(name)
                    self._set_entity_aliases(entity_id, self._aliases_for(name, aliases))
                    self._link_fact_entity(fact_id, entity_id)
                self._garbage_collect_entities()
                self._commit_if_needed()
            elif aliases is not None:
                # Alias-only updates enrich canonical entities already linked to
                # this fact. They do not mutate its entity membership.
                for canonical, alias_names in aliases.items():
                    entity_id = self._find_entity_id(canonical)
                    if entity_id is None:
                        raise ValueError(f"unknown entity for alias update: {canonical}")
                    linked = self._conn.execute(
                        """
                        SELECT 1 FROM fact_entities
                        WHERE fact_id = ? AND entity_id = ?
                        """,
                        (fact_id, entity_id),
                    ).fetchone()
                    if linked is None:
                        raise ValueError(
                            f"entity is not linked to fact {fact_id}: {canonical}"
                        )
                    self._set_entity_aliases(entity_id, alias_names)
                self._commit_if_needed()

            # Recompute HRR whenever text or entity bindings changed.
            if content is not None or entities is not None:
                vector_content = content
                if vector_content is None:
                    vector_content = self._conn.execute(
                        "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
                    ).fetchone()["content"]
                self._compute_hrr_vector(fact_id, vector_content)
            # Bank rebuilds removed (2026-08-04) — dead weight, see add_fact.

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact, its links, and entities left with no facts."""
        with self._atomic():
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._commit_if_needed()
            self._garbage_collect_entities()
            self._rebuild_bank(row["category"])
            return True

    def garbage_collect_entities(self) -> int:
        """Remove every entity no longer linked to a fact."""
        with self._atomic():
            return self._garbage_collect_entities()

    def _garbage_collect_entities(self) -> int:
        cur = self._conn.execute(
            """
            DELETE FROM entities
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_entities fe
                WHERE fe.entity_id = entities.entity_id
            )
            """
        )
        return cur.rowcount

    def audit(self) -> dict:
        """Return a read-only integrity and index-parity report."""
        with self._lock:
            one = lambda sql: self._conn.execute(sql).fetchone()[0]
            facts = one("SELECT COUNT(*) FROM facts")
            # COUNT(*) on an external-content FTS5 table scans the CONTENT table
            # (facts), not the inverted index, so "fts_rows == facts" was a
            # tautology that could never fail. Verified: wiping the index with
            # delete-all left audit() reporting healthy=True while every search
            # returned nothing — the one check an operator would trust was
            # structurally blind to total loss of lexical recall.
            fts_rows = one("SELECT COUNT(*) FROM facts_fts")
            # FTS5's own integrity-check is NOT enough: measured, it reports "ok"
            # for a wiped index, because empty is a *consistent* state. The only
            # signal that recall actually works is that a term known to be in the
            # corpus still matches, so probe one.
            fts_integrity = "ok"
            try:
                self._conn.execute(
                    "INSERT INTO facts_fts(facts_fts) VALUES('integrity-check')"
                )
            except sqlite3.Error as exc:
                fts_integrity = f"integrity-check failed: {exc}"
            else:
                fts_integrity = self._probe_fts_recall(facts)
            facts_with_hrr = one(
                "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL"
            )
            facts_without_hrr = facts - facts_with_hrr
            facts_without_entities = one(
                """
                SELECT COUNT(*) FROM facts f
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_entities fe WHERE fe.fact_id = f.fact_id
                )
                """
            )
            orphan_entities = one(
                """
                SELECT COUNT(*) FROM entities e
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_entities fe WHERE fe.entity_id = e.entity_id
                )
                """
            )
            orphan_links = one(
                """
                SELECT COUNT(*) FROM fact_entities fe
                LEFT JOIN facts f ON f.fact_id = fe.fact_id
                LEFT JOIN entities e ON e.entity_id = fe.entity_id
                WHERE f.fact_id IS NULL OR e.entity_id IS NULL
                """
            )
            banks = one("SELECT COUNT(*) FROM memory_banks")
            bank_fact_count = one(
                "SELECT COALESCE(SUM(fact_count), 0) FROM memory_banks"
            )
            integrity_check = one("PRAGMA integrity_check")
            foreign_keys = bool(one("PRAGMA foreign_keys"))
            healthy = all(
                (
                    integrity_check == "ok",
                    foreign_keys,
                    fts_integrity == "ok",
                    facts_without_hrr == 0 if self._hrr_available else True,
                    orphan_entities == 0,
                    orphan_links == 0,
                )
            )
            return {
                "path": str(self.db_path),
                "integrity_check": integrity_check,
                "fts_integrity": fts_integrity,
                "foreign_keys": foreign_keys,
                "hrr_available": self._hrr_available,
                "facts": facts,
                "fts_rows": fts_rows,
                "facts_with_hrr": facts_with_hrr,
                "facts_without_hrr": facts_without_hrr,
                "facts_without_entities": facts_without_entities,
                "entities": one("SELECT COUNT(*) FROM entities"),
                "orphan_entities": orphan_entities,
                "orphan_links": orphan_links,
                "banks": banks,
                "bank_fact_count": bank_fact_count,
                "healthy": healthy,
            }

    def _probe_fts_recall(self, fact_count: int) -> str:
        """Confirm the index can still find terms the corpus definitely contains.

        Returns "ok", or a description of the failure. Caller holds the lock.

        An empty external-content index is internally consistent, so FTS5's
        integrity-check passes while every search silently returns nothing — the
        exact failure that made audit() report healthy=True on a store that had
        lost all lexical recall.

        SAMPLES SEVERAL FACTS, not one. A single-fact probe was blind to partial
        desync: deleting 19 of 20 facts' index rows while leaving the probe target
        intact still reported healthy=True. Spread the sample across the id range,
        because rows are usually lost in blocks (a failed batch, an interrupted
        migration), not at random.
        """
        if fact_count == 0:
            return "ok"  # nothing to find; not a fault
        rows = self._conn.execute(
            "SELECT fact_id, content FROM facts WHERE content <> ''"
            # Spread over the id range: first, last and a few in between.
            " ORDER BY fact_id"
        ).fetchall()
        if not rows:
            return "ok"
        step = max(1, len(rows) // _FTS_PROBE_SAMPLES)
        sample = rows[::step][:_FTS_PROBE_SAMPLES]
        # Always include the newest fact: an interrupted write leaves the tail
        # unindexed, and that is the most recent thing the agent learned.
        if rows[-1] not in sample:
            sample.append(rows[-1])

        checked = 0
        for row in sample:
            # \w with re.UNICODE, NOT [A-Za-z0-9]: the ASCII class truncated at
            # the first accent, so "Coração" yielded the fragment "Cora", which is
            # not a token in the index — and audit() then reported a perfectly
            # healthy Portuguese store as desynced and demanded a rebuild that
            # could not clear it. This store is full of Portuguese.
            words = [w for w in re.findall(r"[^\W\d_]{4,}|\w{4,}", row["content"] or "",
                                           re.UNICODE)]
            if not words:
                continue  # cannot form a probe from this fact; do not cry wolf
            checked += 1
            try:
                # Constrain to THIS fact's rowid. Matching the term alone was
                # useless: the first long word is usually shared across facts, so
                # one surviving index row satisfied every probe — 19 of 20 facts
                # could be missing and the audit still passed.
                hits = self._conn.execute(
                    "SELECT COUNT(*) FROM facts_fts"
                    " WHERE facts_fts MATCH ? AND rowid = ?",
                    (f'"{words[0].lower()}"', row["fact_id"]),
                ).fetchone()[0]
            except sqlite3.Error as exc:
                return f"probe query failed: {exc}"
            if hits == 0:
                return (
                    f"fact {row['fact_id']} is not findable by {words[0]!r}, a term"
                    f" in its own content — the index is out of sync with"
                    f" {len(rows)} facts; run rebuild_fts()"
                )
        if checked == 0:
            return "ok"  # no probeable content anywhere
        return "ok"

    def rebuild_fts(self) -> str:
        """Rebuild the FTS5 index from the facts table.

        The repair for what _probe_fts_recall detects. Nothing in the plugin could
        do this before, so a desynced index was permanent.
        """
        with self._lock:
            self._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
            self._commit_if_needed()
        return self._probe_fts_recall(
            self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        )

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)
            # Downvotes demote, they do not delete. Never push a fact below the
            # retrieval floor, or it can never be served again and so can never
            # earn its trust back. A fact already below the floor (set explicitly,
            # or by an older build) is left where it is rather than promoted.
            if not helpful and old_trust >= _TRUST_FEEDBACK_MIN:
                new_trust = max(new_trust, _TRUST_FEEDBACK_MIN)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._commit_if_needed()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _normalize_entities(
        self, entities: list[str], *, strict: bool = True
    ) -> list[str]:
        """Validate and case-insensitively deduplicate entity names.

        ``strict=True`` (the default, for entities a caller named explicitly)
        raises on a rejected name: the caller asked for a specific link and
        deserves to hear that it was refused.

        ``strict=False`` skips rejects instead. That is for HEURISTIC candidates,
        where raising is catastrophic: _extract_entities runs inside add_fact's
        atomic group, so one junk candidate from a regex — an apostrophe
        cross-pair such as "s shell is zsh and it" — rolled the whole write back
        and silently discarded the fact the agent was told to remember.
        """
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in entities:
            if not isinstance(raw, str):
                if strict:
                    raise ValueError("entities must contain only strings")
                continue
            name = raw.strip()
            if not _is_valid_entity(name):
                if strict:
                    raise ValueError(f"invalid entity name: {raw!r}")
                continue
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                normalized.append(name)
        return normalized

    def facts_for_entity(self, name: str) -> list[dict]:
        """Return facts explicitly linked to an entity or alias."""
        with self._lock:
            entity_id = self._find_entity_id(name)
            if entity_id is None:
                return []
            rows = self._conn.execute(
                """
                SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.retrieval_count, f.helpful_count, f.created_at, f.updated_at
                FROM facts f
                JOIN fact_entities fe ON fe.fact_id = f.fact_id
                WHERE fe.entity_id = ?
                ORDER BY f.trust_score DESC, f.updated_at DESC
                """,
                (entity_id,),
            ).fetchall()
            return [self._row_to_dict(fact) for fact in rows]

    def facts_for_entities_intersection(self, names: list[str]) -> list[dict]:
        """Return facts linked to every requested entity, using exact SQL."""
        with self._lock:
            normalized = [name.strip() for name in names if name.strip()]
            if not normalized:
                return []
            entity_ids = [self._find_entity_id(name) for name in normalized]
            if any(entity_id is None for entity_id in entity_ids):
                return []
            unique_ids = list(dict.fromkeys(entity_ids))
            placeholders = ",".join("?" for _ in unique_ids)
            rows = self._conn.execute(
                f"""
                SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.retrieval_count, f.helpful_count, f.created_at, f.updated_at
                FROM facts f
                JOIN fact_entities fe ON fe.fact_id = f.fact_id
                WHERE fe.entity_id IN ({placeholders})
                GROUP BY f.fact_id
                HAVING COUNT(DISTINCT fe.entity_id) = ?
                ORDER BY f.trust_score DESC, f.updated_at DESC
                """,
                [*unique_ids, len(unique_ids)],
            ).fetchall()
            return [self._row_to_dict(fact) for fact in rows]

    def _find_entity_id(self, name: str) -> int | None:
        """Resolve an existing entity/alias without creating a new row."""
        normalized = name.strip()
        if not normalized:
            return None
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE",
            (normalized,),
        ).fetchone()
        if row is None:
            escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            row = self._conn.execute(
                """
                SELECT entity_id FROM entities
                WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%' ESCAPE '\\' COLLATE NOCASE
                """,
                (escaped,),
            ).fetchone()
        return int(row["entity_id"]) if row is not None else None

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if not _is_valid_entity(stripped):
                return
            if stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        # A one-word proper noun is still a proper noun. _RE_CAPITALIZED demands
        # TWO consecutive capitalised words, so "Bedrock", "Claude", "Avell",
        # "Hermes" — the entities this store is mostly ABOUT — were never
        # extracted. Measured: 20 of 104 facts had no entity at all, and every
        # one of them named a single-word product or vendor. _is_valid_entity
        # already rejects sentence-initial verbs and status words, so this is
        # generous at the pattern and strict at the filter.
        for m in _RE_PROPER_NOUN.finditer(text):
            _add(m.group(1))

        # Technical slugs are the other half of the gap: copilot-acp, us-west-2,
        # glm-4.7-flash, gpt-5.6-terra. These are the names the agent is asked
        # about most, and no pattern saw them.
        for m in _RE_SLUG.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    @staticmethod
    def _aliases_for(
        entity_name: str,
        aliases: dict[str, list[str]] | None,
    ) -> list[str]:
        if not aliases:
            return []
        for canonical, values in aliases.items():
            if str(canonical).strip().casefold() == entity_name.casefold():
                return values if isinstance(values, list) else []
        return []

    def _set_entity_aliases(self, entity_id: int, aliases: list[str]) -> None:
        if not aliases:
            return
        row = self._conn.execute(
            "SELECT name, aliases FROM entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown entity_id {entity_id}")
        values = [value.strip() for value in str(row["aliases"] or "").split(",") if value.strip()]
        seen = {value.casefold() for value in values}
        canonical = str(row["name"]).casefold()
        for alias in aliases:
            alias = str(alias).strip()
            if not alias or alias.casefold() == canonical or alias.casefold() in seen:
                continue
            if "," in alias:
                raise ValueError("entity aliases must not contain commas")
            existing_id = self._find_entity_id(alias)
            if existing_id is not None and existing_id != entity_id:
                raise ValueError(f"entity alias already belongs to another entity: {alias}")
            values.append(alias)
            seen.add(alias.casefold())
        self._conn.execute(
            "UPDATE entities SET aliases = ? WHERE entity_id = ?",
            (",".join(values), entity_id),
        )
        self._commit_if_needed()

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match (case-insensitive). Use = COLLATE NOCASE, NOT LIKE:
        # LIKE treats '_' and '%' in the incoming name as wildcards, which
        # silently false-merges distinct entities (e.g. 'anthropic_messages'
        # would match 'anthropicXmessages'). COLLATE NOCASE keeps the
        # case-insensitivity without the wildcard footgun.
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated. Escape LIKE
        # wildcards in the incoming name so '_'/'%' can't over-match, and match
        # case-insensitively via COLLATE NOCASE.
        escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%' ESCAPE '\\' COLLATE NOCASE
            """,
            (escaped,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._commit_if_needed()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._commit_if_needed()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._commit_if_needed()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._commit_if_needed()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._commit_if_needed()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
            # Bank rebuilds removed (2026-08-04) — dead weight, see add_fact.

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``.

        ``close()`` is refcount-driven, so a live holder (e.g. an agent's
        memory provider) keeps a profile's SQLite handle open indefinitely.
        That is exactly what a profile delete must break on Windows: the
        desktop's main ``serve`` process opens ``memory_store.db`` for every
        known profile, and ``rmtree`` of the profile directory fails with
        ``WinError 32`` while any of those handles is open (#88347). This
        closes the matching connections unconditionally — the directory is
        going away, so later use by a stale holder is expected to fail — and
        returns how many were closed. In a process that holds none (e.g. the
        CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            # Snapshot the keys first so the registry stays stable while
            # connections are closed inside their per-database locks (closing
            # can run no user code, but this keeps the invariant obvious).
            doomed = [
                key
                for key in cls._shared
                if os.path.normcase(key).startswith(root)
            ]
            for key in doomed:
                entry = cls._shared.pop(key)
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    # A connection that is already closed or broken must not
                    # abort releasing its siblings.
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry. After release_all_under() force-
                    # closed this entry (profile delete, #88347) a same-path
                    # store may have re-registered a FRESH entry under the
                    # same key; a stale holder's late close() must not evict
                    # it — that would silently reintroduce the multi-writer
                    # contention this registry exists to prevent.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
