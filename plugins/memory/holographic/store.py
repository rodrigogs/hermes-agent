"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

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
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
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
    # strip surrounding punctuation from each word before stopword checks
    words = [re.sub(r"^[\W_]+|[\W_]+$", "", w) for w in low.split()]
    words = [w for w in words if w]
    # first word is a stopword/verb -> sentence fragment, not an entity
    if words and words[0] in _ENTITY_STOPWORDS:
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
            entity_names = entities if entities is not None else self._extract_entities(content)
            for name in self._normalize_entities(entity_names):
                entity_id = self._resolve_entity(name)
                self._set_entity_aliases(entity_id, self._aliases_for(name, aliases))
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
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
            # A category move changes two aggregates: remove from the old bank
            # and add to the new one, atomically with the fact update.
            old_category = row["category"]
            new_category = category or old_category
            if new_category != old_category:
                self._rebuild_bank(old_category)
            self._rebuild_bank(new_category)

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
            fts_rows = one("SELECT COUNT(*) FROM facts_fts")
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
                    fts_rows == facts,
                    facts_without_hrr == 0 if self._hrr_available else True,
                    orphan_entities == 0,
                    orphan_links == 0,
                    bank_fact_count == facts_with_hrr,
                )
            )
            return {
                "path": str(self.db_path),
                "integrity_check": integrity_check,
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

    def _normalize_entities(self, entities: list[str]) -> list[str]:
        """Validate and case-insensitively deduplicate entity names."""
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in entities:
            if not isinstance(raw, str):
                raise ValueError("entities must contain only strings")
            name = raw.strip()
            if not _is_valid_entity(name):
                raise ValueError(f"invalid entity name: {raw!r}")
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

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
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
        """Recompute all HRR vectors + banks from text. For recovery/migration.

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

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

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
                    MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
