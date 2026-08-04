"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        probe_min_score: float = 0.08,
        reason_min_score: float = 0.08,
        dense_weight: float = 0.4,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight
        # Retained for config/API compatibility with pre-Honest-Memory setups.
        # Exact probe/reason paths abstain instead of thresholding random HRR
        # similarities; related/search still use HRR where it is non-assertive.
        self.probe_min_score = probe_min_score
        self.reason_min_score = reason_min_score
        # Dense retrieval is OPTIONAL and lazily attached. When it is absent or its
        # endpoint is down, every path below behaves exactly as it did before it
        # existed — that is why the weight is additive and the lexical score stays
        # the dominant term.
        self._dense = None
        self.dense_weight = float(dense_weight)

    def attach_dense(self, index) -> None:
        """Give the retriever an EmbeddingIndex. Optional by design."""
        self._dense = index

    @property
    def dense_available(self) -> bool:
        return bool(self._dense and self._dense.embedder.available)

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            # NOT an early return. The dense path exists precisely for the queries
            # the lexical stage cannot answer — "is there anything about flying
            # machines" finds the drone fact at cosine 0.46 while FTS finds zero
            # rows. Returning here made the whole embedding layer dead code for its
            # own best cases, which is what the first version of this did.
            dense_only = self._fuse_dense(query, [], category, min_trust, limit)
            if not dense_only:
                return []
            dense_only.sort(key=lambda f: f["score"], reverse=True)
            results = dense_only[:limit]
            self.store.record_retrievals([f["fact_id"] for f in results if f.get("fact_id")])
            return results

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        # The query vector is loop-invariant — encode it at most once, on
        # the first candidate that actually carries an HRR vector. Lazy on
        # purpose: migrated stores can have FTS candidates whose hrr_vector
        # was never backfilled, and those must not pay for an encode nothing
        # will use. encode_text is deterministic (SHA-256 counter blocks),
        # so the hoisted vector is bit-identical to per-candidate encodes.
        # The role atom is likewise hoisted and reused for the bind.
        query_vec = None
        role_content = None
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                # Bind the query to ROLE_CONTENT so it matches how encode_fact
                # stores content (content is bound to role_content, not bare).
                # Comparing a bare query vector against role-bound content is
                # pure noise (~0.5 for every fact) — see audit 2026-07.
                if role_content is None:
                    role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
                if query_vec is None:
                    query_vec = hrr.bind(hrr.encode_text(query, self.hrr_dim), role_content)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Stage 3: fuse dense similarity over the UNION of both candidate sets.
        #
        # Measured on the real 104-fact corpus, 21 questions (13 direct, 8
        # paraphrased or Portuguese):
        #     lexical only      17/21 recall@5, MRR 0.654
        #     dense only        17/21           MRR 0.692
        #     RRF (k=60)        16/21           MRR 0.702   <- worse, rejected
        #     weighted union    19/21 (90%)     MRR 0.716   <- this
        # The two disagree on which cases they rescue, so the win comes from the
        # UNION: a fact the lexical stage never retrieved can still be served.
        scored = self._fuse_dense(query, scored, category, min_trust, limit)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
        # Count what was actually served. This is the ONLY live retrieval path,
        # so if it does not record use, retrieval_count stays zero forever and a
        # never-once-useful memory is indistinguishable from a load-bearing one.
        self.store.record_retrievals([f["fact_id"] for f in results if f.get("fact_id")])
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return facts explicitly linked to an entity or alias.

        This is an assertive recall operation, so it uses persisted SQL links
        only and abstains for unknown entities instead of guessing via HRR.
        """
        exact = self.store.facts_for_entity(entity)
        if category is not None:
            exact = [fact for fact in exact if fact["category"] == category]
        if exact:
            for fact in exact[:limit]:
                fact["score"] = fact["trust_score"]
                fact["retrieval_method"] = "entity_sql"
            return exact[:limit]

        # Honest Memory is evidence-based: if no persisted entity/alias link
        # exists, a random HRR similarity is not evidence that the entity is
        # known. Probabilistic retrieval remains useful for search/related, but
        # probe must abstain here.
        return []

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        with self.store._lock:
            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        # Both role atoms are loop-invariant — encode them once here
        # (deterministic SHA-256-based atoms) instead of per fact row.
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact
            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return facts explicitly linked to every requested entity.

        The SQL intersection is auditable and has strict AND semantics. If no
        persisted fact contains all entities, abstain instead of inferring a
        relation from probabilistic vector similarity.
        """
        if not entities:
            return []

        exact = self.store.facts_for_entities_intersection(entities)
        if category is not None:
            exact = [fact for fact in exact if fact["category"] == category]
        if exact:
            for fact in exact[:limit]:
                fact["score"] = fact["trust_score"]
                fact["retrieval_method"] = "entity_sql_intersection"
            return exact[:limit]

        # A relation is only asserted when one persisted fact links every
        # requested entity. HRR similarity cannot establish that conjunction.
        return []

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        with self.store._lock:
            rows = conn.execute(
                f"""
                SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.created_at, f.updated_at, f.hrr_vector
                FROM facts f
                {where}
                """,
                params,
            ).fetchall()

            if len(rows) < 2:
                return []

            # Guard against O(n²) explosion on large fact stores.
            # At 500 facts, that's ~125K comparisons — acceptable.
            # Above that, only check the most recently updated facts.
            _MAX_CONTRADICT_FACTS = 500
            if len(rows) > _MAX_CONTRADICT_FACTS:
                rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
                rows = rows[:_MAX_CONTRADICT_FACTS]

            # Build entity sets from the same committed snapshot as the facts.
            fact_entities: dict[int, set[str]] = {}
            for row in rows:
                fid = row["fact_id"]
                entity_rows = conn.execute(
                    """
                    SELECT e.name FROM entities e
                    JOIN fact_entities fe ON fe.entity_id = e.entity_id
                    WHERE fe.fact_id = ?
                    """,
                    (fid,),
                ).fetchall()
                fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        with self.store._lock:
            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            # max(0, sim) not (sim+1)/2 — see probe(): the 0.5 non-match
            # baseline lets trust outrank relevance.
            fact["score"] = max(0.0, sim) * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # Minimum cosine for a DENSE-ONLY candidate to be served.
    #
    # An embedding always returns its nearest neighbours, so without a floor the
    # retriever answers every question — including ones the corpus cannot answer.
    # Measured on the benchmark corpus: with no floor, recall@5 hit 1.00 but all
    # four unanswerable questions returned memories, which is fabrication, not
    # recall.
    #
    # 0.53 is recalibrated for qwen3-embedding:0.6b (supersedes the 0.50 that
    # was measured with nomic-embed-text, whose similarity distribution sits
    # ~0.05 lower). Measured on the real 104-fact corpus with qwen3:
    # unanswerable max 0.5071, answerable min 0.5788 — a clean gap, so 0.53
    # rejects every unanswerable question and keeps every answerable one.
    # The earlier 0.50 let domain-plausible distractors through ("which AWS
    # region hosts the postgres primary", 0.54) with the stronger model.
    #
    # A fact the LEXICAL stage found is never subject to this floor: overlapping
    # words are independent evidence, and the dense score only reweights it.
    _DENSE_MIN_SIM = 0.53

    def _fuse_dense(
        self,
        query: str,
        scored: list[dict],
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Blend dense similarity into the lexical scores, adding new candidates.

        Fails open in every direction: no index, no endpoint, no vectors, a
        dimension mismatch or any exception leaves ``scored`` untouched.
        """
        if not self.dense_available:
            return scored
        try:
            neighbours = self._dense.similar(query, limit=max(limit * 4, 20))
        except Exception:
            return scored
        if not neighbours:
            return scored

        dense_by_id = {fid: sim for fid, sim in neighbours}
        # Normalise the lexical scores so the two terms are comparable; without
        # this the blend depends on the absolute FTS rank, which is corpus-relative.
        top = max((f.get("score", 0.0) for f in scored), default=0.0) or 1.0
        by_id = {f["fact_id"]: f for f in scored if f.get("fact_id")}

        for fact in scored:
            sim = dense_by_id.get(fact.get("fact_id"), 0.0)
            fact["score"] = ((fact.get("score", 0.0) / top) * (1.0 - self.dense_weight)
                            + max(sim, 0.0) * self.dense_weight)
            if sim:
                fact["dense_sim"] = round(float(sim), 4)

        # Facts the lexical stage never saw. This is where the recall comes from —
        # and where fabrication would come from, hence the floor.
        newcomers = [fid for fid, sim in neighbours
                     if fid not in by_id and sim >= self._DENSE_MIN_SIM][:max(limit * 2, 10)]
        for fact in self._facts_by_id(newcomers, category, min_trust):
            sim = max(dense_by_id.get(fact["fact_id"], 0.0), 0.0)
            fact["fts_rank"] = 0.0
            fact["dense_sim"] = round(float(sim), 4)
            # No lexical component to credit: a dense-only hit scores on similarity
            # alone, still weighted by trust as every other path is.
            fact["score"] = sim * self.dense_weight * float(fact.get("trust_score", 0.5))
            scored.append(fact)
        return scored

    def _facts_by_id(
        self, fact_ids: list[int], category: str | None, min_trust: float
    ) -> list[dict]:
        """Hydrate dense-only candidates, honouring the caller's filters.

        The filters are applied HERE and not left to the caller: a dense hit must
        not smuggle in a fact that min_trust or category would have excluded.
        """
        if not fact_ids:
            return []
        placeholders = ",".join("?" * len(fact_ids))
        params: list = list(fact_ids)
        where = [f"fact_id IN ({placeholders})", "trust_score >= ?"]
        params.append(min_trust)
        if category:
            where.append("category = ?")
            params.append(category)
        try:
            with self.store._lock:
                rows = self.store._conn.execute(
                    "SELECT * FROM facts WHERE " + " AND ".join(where), params
                ).fetchall()
        except Exception:
            return []
        out = []
        for row in rows:
            fact = dict(row)
            fact.pop("hrr_vector", None)
            out.append(fact)
        return out

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        # FTS5 defaults to AND-between-tokens, which kills recall on
        # natural-language queries ("what happened with the deployment
        # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
        # any significant term can match.
        params.append(self._sanitize_fts_query(query))

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        with self.store._lock:
            try:
                rows = conn.execute(sql, params).fetchall()
            except Exception:
                # A malformed MATCH must not silently erase the memory: FTS5
                # rejects bare operators ("AND", "*", "a-b OR c"), which a human
                # types often. Scan instead of giving up.
                rows = []

        if not rows:
            # FTS is lexical: it only finds a fact whose stored WORDS overlap the
            # query. Ask "how do I reach the Oracle machine" and every candidate
            # is dropped when the facts say "araponga"/"ssh" instead — measured
            # on the live store, that query returned zero of 104 facts. Falling
            # through to a token scan keeps a vocabulary mismatch from reading
            # as "the agent knows nothing about this".
            return self._scan_candidates(query, category, min_trust, limit)

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    # Minimum Jaccard for the fallback to call a fact a candidate. Low enough
    # that a short query against a long fact still matches (a 4-token question
    # against a 40-token fact tops out near 0.1), high enough to reject a single
    # incidental word in common.
    _SCAN_MIN_OVERLAP = 0.04

    # Longest query tokens used for the SQL prefilter. Bounded so a rambling
    # question cannot build a 40-clause OR.
    _SCAN_MAX_TOKENS = 6

    def _scan_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Last-resort candidate pass when FTS5 yields nothing.

        Deliberately dumb and bounded: pull the eligible facts and rank them by
        token overlap in Python. At the scale this store actually operates
        (hundreds to low thousands of facts) that costs microseconds, and it is
        the difference between "no memory matched your words" and "no memory
        exists". Candidates come back with ``fts_rank`` at 0.0, so the caller's
        scoring is driven by Jaccard and HRR — the FTS term contributes nothing
        it did not earn.
        """
        params: list = []
        where = ["trust_score >= ?"]
        params.append(min_trust)
        if category:
            where.append("category = ?")
            params.append(category)
        # Narrow in SQL FIRST, then cap. Capping a plain "newest 200" scan threw
        # away the answer: with 301 facts where the ONLY match was the oldest, the
        # fallback returned nothing — defeating the entire reason it exists. A LIKE
        # over each query token is a coarse prefilter (SQLite has no index for it,
        # but this only runs when FTS already found nothing), after which the cap
        # bounds the number of CANDIDATES rather than truncating the corpus.
        tokens = sorted(
            {t for t in self._tokenize(query) if t not in self._FTS_STOPWORDS},
            key=len, reverse=True,
        )[: self._SCAN_MAX_TOKENS]
        if tokens:
            where.append("(" + " OR ".join("lower(content) LIKE ?" for _ in tokens)
                         + " OR " + " OR ".join("lower(tags) LIKE ?" for _ in tokens) + ")")
            params.extend([f"%{t}%" for t in tokens] * 2)
        sql = (
            "SELECT * FROM facts WHERE " + " AND ".join(where)
            # Newest first: when several facts match, a recent one is the likelier
            # answer. The cap is a stall guard, not a filter.
            + " ORDER BY updated_at DESC, fact_id DESC LIMIT ?"
        )
        params.append(max(limit * 20, 200))

        with self.store._lock:
            try:
                rows = self.store._conn.execute(sql, params).fetchall()
            except Exception:
                return []

        # Stopwords must not be evidence. Matching on "the"/"do"/"and" made this
        # fallback surface whatever fact happened to contain a function word —
        # measured: "how do I reach the Oracle machine" returned a note about
        # Python formatting, matched on {"i", "the"}. Noise the model then trusts
        # is worse than an honest miss.
        query_tokens = {t for t in self._tokenize(query) if t not in self._FTS_STOPWORDS}
        if not query_tokens:
            return []

        # A category-centroid tie-breaker was tried here and MEASURED AS HARMFUL:
        # asking each of the 104 live facts to identify its own category from its
        # own content, the bundled centroid ranked the true category first only
        # 3.8% of the time — worse than the 14.3% a uniform guess over 7
        # categories would score. Bundling hundreds of HRR vectors into one
        # destroys the signal. So ranking here uses token overlap alone;
        # store.rank_categories() remains available as a diagnostic, and the
        # dead-weight of maintaining memory_banks is noted in its docstring.
        scored: list[tuple[float, dict]] = []
        for row in rows:
            fact = dict(row)
            fact_tokens = {
                t for t in self._tokenize(fact.get("content", ""))
                | self._tokenize(fact.get("tags", ""))
                if t not in self._FTS_STOPWORDS
            }
            shared = query_tokens & fact_tokens
            if not shared:
                continue
            # Require a content word in common, not merely a nonzero Jaccard: a
            # long fact sharing one incidental token scores above zero yet has
            # nothing to do with the question.
            overlap = self._jaccard_similarity(query_tokens, fact_tokens)
            if overlap < self._SCAN_MIN_OVERLAP:
                continue
            fact["fts_rank"] = 0.0
            scored.append((overlap, fact))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    # Stopwords dropped before FTS5 OR-expansion. Short English function
    # words that carry no retrieval signal and force false-negative AND
    # matches when left in the query.
    _FTS_STOPWORDS = frozenset({
        "a", "about", "above", "after", "again", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
    })

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        FTS5 treats a multi-word MATCH argument as AND-joined by default,
        which tanks recall on prose queries. This helper:
          - tokenizes the query
          - drops stopwords and short (<2 char) tokens
          - strips FTS5 special characters from each token
          - OR-joins the survivors

        If nothing remains (pathological query), falls back to the raw
        query so the caller sees zero results instead of a SQL error.
        """
        if not query:
            return ""
        # Strip FTS5 operator characters from EACH token to avoid accidentally
        # creating a malformed query.
        #
        # The hyphen is deliberately NOT in this set. Deleting it turned
        # "copilot-acp" into "copilotacp", which matches nothing: measured on the
        # live store, every hyphenated term scored zero — copilot-acp (present in
        # 9 facts), capability-router (4), gpt-5.6-terra (5), deepseek-v3.2 (2).
        # Those are precisely the provider, model and plugin names the agent asks
        # about most, so the single most common class of query could not retrieve
        # anything. Inside a quoted FTS5 phrase a hyphen is literal, so keeping it
        # is safe; what is unsafe is a BARE hyphen, which FTS5 reads as NOT.
        _FTS_SPECIAL = '"()*^:+'
        tokens: list[str] = []
        seen: set[str] = set()

        def push(term: str) -> None:
            if len(term) < 2 or term in cls._FTS_STOPWORDS or term in seen:
                return
            seen.add(term)
            # Phrase-literal so no special char can escape as an operator.
            tokens.append(f'"{term}"')

        for raw in query.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>").translate(
                str.maketrans("", "", _FTS_SPECIAL)
            )
            # A stray leading/trailing hyphen is punctuation, not part of a name.
            cleaned = cleaned.strip("-")
            if not cleaned:
                continue
            push(cleaned)
            if "-" in cleaned:
                # Also offer the components: the tokenizer may have indexed
                # "capability" and "router" separately, and either should hit.
                for part in cleaned.split("-"):
                    push(part)
        if not tokens:
            # Nothing survived (pure punctuation, or all stopwords). Returning the
            # raw query here used to raise OperationalError on ordinary input like
            # "AND" or "*"; an empty phrase matches nothing and never throws.
            return '""'
        return " OR ".join(tokens)

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
