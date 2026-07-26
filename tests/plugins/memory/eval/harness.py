"""Retrieval evaluation harness for the holographic memory store.

Pure library: no pytest import, no network, no LLM. Rebuilds a frozen corpus
snapshot into a throwaway SQLite file, runs the real FactRetriever against it,
and returns metrics. Imported by both the pytest gate and the CLI.

Design constraints, and why:

* The corpus is a JSONL SNAPSHOT, not the live database. The live store is
  mutated every turn (retrieval_count, trust_score, new facts), so measuring
  against it makes yesterday's number unreproducible. The snapshot is
  human-diffable, so a corpus change shows up in code review.
* fact_id is carried in the snapshot and inserted EXPLICITLY. Question
  expectations are fact_ids; letting AUTOINCREMENT renumber them would silently
  re-point every expectation at the wrong fact.
* search() writes retrieval_count. It runs against the temp copy only, so an
  eval run never perturbs the production store's usage statistics.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
CORPUS = EVAL_DIR / "corpus.jsonl"
BASELINE = EVAL_DIR / "baseline.json"

# The prefetch() hook injects exactly 5 facts into the system prompt every turn.
# K is that number and nothing else: measuring recall@10 would score facts the
# agent never sees.
K = 5


def _repo_root() -> Path:
    # tests/plugins/memory/eval/harness.py -> repo root
    return EVAL_DIR.parents[3]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def build_corpus(db_path: Path, corpus: list[dict] | None = None):
    """Materialise the snapshot into a fresh store at db_path.

    Returns (MemoryStore, FactRetriever) built with the SAME weights the live
    plugin uses, so a measured delta is attributable to the retriever and not to
    a config difference between test and production.
    """
    sys.path.insert(0, str(_repo_root()))
    from plugins.memory.holographic.store import MemoryStore

    records = corpus if corpus is not None else load_jsonl(CORPUS)

    store = MemoryStore(db_path=str(db_path), default_trust=0.5, hrr_dim=1024)
    conn = store._conn
    with store._lock:
        for rec in records:
            # Direct INSERT, not add_fact(): add_fact cannot pin fact_id, and the
            # expectations in questions.jsonl are fact_ids. The FTS5 triggers
            # (facts_ai) fire on this INSERT, so the external-content index is
            # populated exactly as it is in production.
            conn.execute(
                "INSERT INTO facts (fact_id, content, category, tags, trust_score,"
                " retrieval_count, helpful_count, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    rec["fact_id"], rec["content"], rec["category"], rec.get("tags", ""),
                    rec["trust_score"], 0, 0, rec["created_at"], rec["updated_at"],
                ),
            )
        for rec in records:
            for name in rec.get("entities", []):
                eid = conn.execute(
                    "SELECT entity_id FROM entities WHERE name = ?", (name,)
                ).fetchone()
                if eid is None:
                    cur = conn.execute(
                        "INSERT INTO entities (name, entity_type) VALUES (?, 'unknown')",
                        (name,),
                    )
                    entity_id = cur.lastrowid
                else:
                    entity_id = eid[0]
                conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?,?)",
                    (rec["fact_id"], entity_id),
                )
        conn.commit()

    # HRR vectors are derived, never snapshotted: they are a deterministic
    # function of content + entities (SHA-256 atoms), so recomputing them keeps
    # the fixture small and catches an accidental change to the encoder.
    store.rebuild_all_vectors()
    return store


def make_retriever(store, **overrides):
    from plugins.memory.holographic.retrieval import FactRetriever

    kwargs = dict(
        fts_weight=0.55,      # live config.yaml plugins.hermes-memory-store
        jaccard_weight=0.15,
        hrr_weight=0.3,
        hrr_dim=1024,
        temporal_decay_half_life=0,
    )
    kwargs.update(overrides)
    return FactRetriever(store, **kwargs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    qid: str
    kind: str
    query: str
    expected: list[int]
    returned: list[int]
    first_hit_rank: int | None
    top_score: float
    abs_bm25: float          # un-normalised FTS5 bm25 of the best candidate
    scores: list[float] = field(default_factory=list)


@dataclass
class Report:
    n_positive: int
    n_negative: int
    recall_at_k: float
    mrr_at_k: float
    precision_at_1: float
    negative_precision: float      # fraction of negatives that returned NOTHING
    per_query: list[QueryResult]
    misses: list[str]
    negative_leaks: list[str]

    def as_baseline(self) -> dict:
        return {
            "k": K,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr_at_k": round(self.mrr_at_k, 4),
            "precision_at_1": round(self.precision_at_1, 4),
            "negative_precision": round(self.negative_precision, 4),
            "known_misses": sorted(self.misses),
            "known_negative_leaks": sorted(self.negative_leaks),
        }


def _abs_bm25(store, retriever, query: str, min_trust: float) -> float:
    """Absolute bm25 of the best FTS5 candidate.

    Reported because _fts_candidates() divides every rank by the maximum rank in
    the candidate set, so the top candidate ALWAYS scores 1.0 regardless of how
    weakly it matched. That normalisation is why a nonsense query still produces
    a confident-looking score. The absolute value is the honest signal and is
    recorded here so a future abstain gate can be evaluated on this same set.
    """
    match = retriever._sanitize_fts_query(query)
    try:
        with store._lock:
            row = store._conn.execute(
                "SELECT facts_fts.rank AS rk FROM facts_fts"
                " JOIN facts f ON f.fact_id = facts_fts.rowid"
                " WHERE facts_fts MATCH ? AND f.trust_score >= ?"
                " ORDER BY facts_fts.rank LIMIT 1",
                (match, min_trust),
            ).fetchone()
    except sqlite3.Error:
        return 0.0
    return abs(row["rk"]) if row else 0.0


def evaluate(store, retriever, questions: list[dict], k: int = K,
             min_trust: float = 0.3) -> Report:
    per_query: list[QueryResult] = []
    hits = at1 = 0
    rr = 0.0
    n_pos = n_neg = 0
    neg_clean = 0
    misses: list[str] = []
    leaks: list[str] = []

    for q in questions:
        results = retriever.search(q["query"], min_trust=min_trust, limit=k)
        ids = [int(f["fact_id"]) for f in results]
        scores = [float(f["score"]) for f in results]
        expected = [int(x) for x in q.get("expect", [])]
        rank = next((i + 1 for i, fid in enumerate(ids) if fid in expected), None)
        per_query.append(QueryResult(
            qid=q["id"], kind=q["kind"], query=q["query"], expected=expected,
            returned=ids, first_hit_rank=rank,
            top_score=scores[0] if scores else 0.0,
            abs_bm25=_abs_bm25(store, retriever, q["query"], min_trust),
            scores=scores,
        ))

        if q["kind"] == "negative":
            n_neg += 1
            if not ids:
                neg_clean += 1
            else:
                leaks.append(q["id"])
            continue

        n_pos += 1
        if rank:
            hits += 1
            rr += 1.0 / rank
            if rank == 1:
                at1 += 1
        else:
            misses.append(q["id"])

    return Report(
        n_positive=n_pos,
        n_negative=n_neg,
        recall_at_k=hits / n_pos if n_pos else 0.0,
        mrr_at_k=rr / n_pos if n_pos else 0.0,
        precision_at_1=at1 / n_pos if n_pos else 0.0,
        negative_precision=neg_clean / n_neg if n_neg else 1.0,
        per_query=per_query,
        misses=misses,
        negative_leaks=leaks,
    )


def run(tmpdir: Path, questions: list[dict] | None = None, **retriever_kw) -> Report:
    """One-call entry point: build corpus in tmpdir, evaluate, return Report."""
    store = build_corpus(tmpdir / "eval_corpus.db")
    try:
        retriever = make_retriever(store, **retriever_kw)
        qs = questions if questions is not None else load_jsonl(QUESTIONS)
        return evaluate(store, retriever, qs)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Honesty audit: how much of the score is free lexical overlap?
# ---------------------------------------------------------------------------

def lexical_leakage(store, retriever, questions: list[dict]) -> dict:
    """Fraction of positive questions winnable by rare-token exact match alone.

    A question set written by looking at the facts drifts toward reusing the
    facts' own words, and then the benchmark measures string matching rather
    than retrieval. This quantifies that drift so it can be watched over time
    instead of assumed away. A rising number means the set is going stale.
    """
    with store._lock:
        rows = store._conn.execute("SELECT fact_id, content FROM facts").fetchall()
    content = {r["fact_id"]: r["content"] for r in rows}
    n = len(content)
    df: dict[str, int] = {}
    for text in content.values():
        for tok in retriever._tokenize(text):
            df[tok] = df.get(tok, 0) + 1

    free = total = 0
    detail = {}
    for q in questions:
        if q["kind"] == "negative":
            continue
        total += 1
        qt = {t for t in retriever._tokenize(q["query"])
              if t not in retriever._FTS_STOPWORDS}
        tgt: set[str] = set()
        for fid in q["expect"]:
            tgt |= {t for t in retriever._tokenize(content.get(fid, ""))
                    if t not in retriever._FTS_STOPWORDS}
        shared = qt & tgt
        rarest = min([df.get(t, n) for t in shared], default=n)
        verbatim = len(shared) / max(len(qt), 1)
        is_free = rarest <= 3
        free += int(is_free)
        detail[q["id"]] = {"verbatim_share": round(verbatim, 3),
                           "rarest_shared_df": rarest, "free": is_free}
    return {"free": free, "total": total,
            "free_share": round(free / total, 3) if total else 0.0,
            "detail": detail}


# ---------------------------------------------------------------------------
# CLI: report, and the DELIBERATE baseline update
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(prog="tests.plugins.memory.eval.harness")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite baseline.json from the current run (review the diff!)")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--verbose", "-v", action="store_true", help="per-query table")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="memeval-"))
    report = run(tmp)

    store = build_corpus(tmp / "leak.db")
    try:
        leak = lexical_leakage(store, make_retriever(store), load_jsonl(QUESTIONS))
    finally:
        store.close()

    payload = report.as_baseline()
    payload["lexical_free_share"] = leak["free_share"]

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"corpus      : {report.n_positive} positive / {report.n_negative} negative questions")
        print(f"recall@{K}    : {report.recall_at_k:.3f}")
        print(f"MRR@{K}       : {report.mrr_at_k:.3f}")
        print(f"P@1         : {report.precision_at_1:.3f}")
        print(f"neg. precis.: {report.negative_precision:.3f}  "
              f"({report.n_negative - len(report.negative_leaks)}/{report.n_negative} "
              f"correctly returned nothing)")
        print(f"misses      : {sorted(report.misses)}")
        print(f"neg. leaks  : {sorted(report.negative_leaks)}")
        print(f"lexical free: {leak['free']}/{leak['total']} questions winnable by "
              f"rare-token exact match")

    if args.verbose:
        print(f"\n{'id':<6} {'kind':<11} {'rank':>4} {'top':>6} {'bm25':>6}  returned")
        for q in report.per_query:
            print(f"{q.qid:<6} {q.kind:<11} {str(q.first_hit_rank):>4} "
                  f"{q.top_score:>6.3f} {q.abs_bm25:>6.2f}  {q.returned}")

    if args.update_baseline:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {BASELINE}")
        print("COMMIT THIS SEPARATELY, with the reason in the message. A baseline "
              "that moves in the same commit as a retrieval change hides which "
              "one caused which.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
