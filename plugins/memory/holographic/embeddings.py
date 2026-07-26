"""Local dense embeddings for the fact store.

WHY THIS EXISTS, and why it is shaped the way it is.

Retrieval here was purely lexical, so a question phrased differently from the
stored fact found nothing. Measured on the real 104-fact corpus with 21 questions
(13 direct, 8 paraphrased or in Portuguese):

    lexical only        17/21 recall@5, MRR 0.654
    dense only          17/21           MRR 0.692
    RRF fusion (k=60)   16/21           MRR 0.702   <- worse recall, rejected
    weighted fusion     19/21 (90%)     MRR 0.716   <- shipped

Lexical and dense each rescue cases the other misses — "is there anything about
flying things" and "quanto tempo leva pra compactar" are invisible to FTS, while
"o que quebra quando eu atualizo" is invisible to the embedding. Fusion is
therefore additive, not a replacement, and the lexical score stays the dominant
term.

DESIGN CONSTRAINTS, all deliberate:

* LOCAL ONLY. The model runs in ollama on this box (nomic-embed-text, 768 dims,
  ~45ms per fact). Memory retrieval must not depend on a paid API or the internet
  being up, and a fact must never fail to be stored because an embedding provider
  was down.
* FAILS OPEN. Every entry point returns None or [] on any failure. If the embedder
  is missing, slow, or broken, retrieval degrades to exactly the lexical behaviour
  that exists today. It never raises into a turn.
* NEVER ON THE WRITE PATH SYNCHRONOUSLY BLOCKING A FACT. add_fact stores the fact
  first; the vector is computed after, best-effort, and a missing vector is a
  normal state that backfill() repairs later.
* CACHED BY CONTENT HASH. The same text is never embedded twice, so a backfill or
  a repeated query costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import threading
import urllib.error
import urllib.request
from typing import Iterable, Optional, Sequence

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is present in this deployment
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Defaults chosen for this box: ollama is already installed and running, and
# nomic-embed-text is small enough to keep resident on CPU.
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/embed"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768
# A turn must not wait on the embedder. 8s is generous for a warm local model
# (measured 45ms/fact) and short enough that a hung ollama cannot hold a turn.
DEFAULT_TIMEOUT = 8.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id      INTEGER PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
    -- The hash of the exact text that was embedded. A fact whose content changes
    -- gets a new hash, which is how staleness is detected without a trigger.
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fact_embeddings_hash ON fact_embeddings(content_hash);
"""


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def pack(vector: "np.ndarray") -> bytes:
    """float32 little-endian, so the blob is portable and half the size of f64."""
    return np.asarray(vector, dtype="<f4").tobytes()


def unpack(blob: bytes) -> "np.ndarray":
    return np.frombuffer(blob, dtype="<f4")


def _normalise(vectors: "np.ndarray") -> "np.ndarray":
    """Unit-length rows, so cosine similarity is a dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


class Embedder:
    """Text -> unit vectors, via a local ollama endpoint.

    Every method fails soft: on any error the caller gets None/[] and retrieval
    carries on lexically. That is the whole contract.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("HERMES_EMBED_ENDPOINT", DEFAULT_ENDPOINT)
        self.model = model or os.environ.get("HERMES_EMBED_MODEL", DEFAULT_MODEL)
        self.timeout = float(timeout if timeout is not None else DEFAULT_TIMEOUT)
        self.dim = dim
        self._lock = threading.Lock()
        self._cache: dict[str, "np.ndarray"] = {}
        # Latched after a failure so a dead endpoint is not retried on every
        # single query, turning one outage into a per-turn stall.
        self._unavailable = False

    @property
    def available(self) -> bool:
        return _HAS_NUMPY and not self._unavailable

    def reset(self) -> None:
        """Clear the unavailable latch (the endpoint may have come back)."""
        with self._lock:
            self._unavailable = False

    def embed(self, texts: Sequence[str]) -> Optional["np.ndarray"]:
        """Embed a batch. Returns an (n, dim) unit-norm array, or None on failure."""
        if not _HAS_NUMPY or not texts:
            return None
        cleaned = [" ".join(str(t or "").split())[:8000] for t in texts]
        keys = [content_hash(f"{self.model}:{t}") for t in cleaned]

        with self._lock:
            if self._unavailable:
                return None
            missing = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing:
            fresh = self._call([cleaned[i] for i in missing])
            if fresh is None:
                return None
            with self._lock:
                for slot, vector in zip(missing, fresh):
                    self._cache[keys[slot]] = vector

        with self._lock:
            try:
                return np.vstack([self._cache[k] for k in keys])
            except KeyError:
                return None

    def embed_one(self, text: str) -> Optional["np.ndarray"]:
        batch = self.embed([text])
        return None if batch is None else batch[0]

    def _call(self, texts: list[str]) -> Optional["np.ndarray"]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            # Latch: a missing or hung endpoint must cost one timeout, not one per
            # query for the rest of the process's life.
            with self._lock:
                self._unavailable = True
            return None

        raw = body.get("embeddings")
        if raw is None and body.get("embedding") is not None:
            raw = [body["embedding"]]
        if not raw or len(raw) != len(texts):
            with self._lock:
                self._unavailable = True
            return None
        try:
            vectors = np.array(raw, dtype="<f4")
        except (TypeError, ValueError):
            with self._lock:
                self._unavailable = True
            return None
        if vectors.ndim != 2 or vectors.shape[1] < 8:
            with self._lock:
                self._unavailable = True
            return None
        self.dim = int(vectors.shape[1])
        return _normalise(vectors)


class EmbeddingIndex:
    """Persistence and search for fact vectors.

    Small-corpus by design: the whole matrix is loaded and multiplied in numpy. At
    104 facts x 768 dims that is 320 kB and a dot product measured in microseconds,
    so an ANN index (sqlite-vec, faiss) would be more moving parts for no gain. The
    honest ceiling is on the order of 100k facts; past that this needs replacing,
    and the docstring should be the thing that tells the next reader so.
    """

    def __init__(self, store, embedder: Embedder | None = None) -> None:
        self._store = store
        self.embedder = embedder or Embedder()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            with self._store._lock:
                self._store._conn.executescript(_SCHEMA)
                self._store._conn.commit()
        except sqlite3.Error:
            # A read-only handle, or a store this build cannot migrate. Dense
            # retrieval simply stays off.
            pass

    # ── writing ────────────────────────────────────────────────────────
    def embed_fact(self, fact_id: int, text: str) -> bool:
        """Compute and store one fact's vector. False on any failure."""
        if not self.embedder.available:
            return False
        vector = self.embedder.embed_one(text)
        if vector is None:
            return False
        return self._save(fact_id, text, vector)

    def _save(self, fact_id: int, text: str, vector: "np.ndarray") -> bool:
        try:
            with self._store._lock:
                self._store._conn.execute(
                    "INSERT INTO fact_embeddings (fact_id, content_hash, model, dim, vector)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(fact_id) DO UPDATE SET content_hash=excluded.content_hash,"
                    " model=excluded.model, dim=excluded.dim, vector=excluded.vector,"
                    " created_at=CURRENT_TIMESTAMP",
                    (int(fact_id), content_hash(text), self.embedder.model,
                     int(vector.shape[0]), pack(vector)),
                )
                self._store._conn.commit()
            return True
        except sqlite3.Error:
            return False

    def stale_fact_ids(self) -> list[tuple[int, str]]:
        """Facts with no vector, or a vector for different text or another model.

        Returns ``(fact_id, embed_text)``. This is what makes a missing embedding a
        normal, repairable state rather than a failure.
        """
        try:
            with self._store._lock:
                rows = self._store._conn.execute(
                    "SELECT f.fact_id, f.content, f.tags, e.content_hash, e.model"
                    " FROM facts f LEFT JOIN fact_embeddings e ON e.fact_id = f.fact_id"
                ).fetchall()
        except sqlite3.Error:
            return []
        stale = []
        for row in rows:
            text = embed_text(row["content"], row["tags"])
            if (row["content_hash"] != content_hash(text)
                    or row["model"] != self.embedder.model):
                stale.append((int(row["fact_id"]), text))
        return stale

    def backfill(self, limit: int | None = None, batch: int = 32) -> int:
        """Embed everything stale. Returns how many vectors were written."""
        pending = self.stale_fact_ids()
        if limit is not None:
            pending = pending[:limit]
        if not pending or not self.embedder.available:
            return 0
        written = 0
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            vectors = self.embedder.embed([text for _, text in chunk])
            if vectors is None:
                break  # endpoint died mid-run; keep what we have
            for (fact_id, text), vector in zip(chunk, vectors):
                if self._save(fact_id, text, vector):
                    written += 1
        return written

    # ── reading ────────────────────────────────────────────────────────
    def load_matrix(self) -> tuple[list[int], Optional["np.ndarray"]]:
        """All stored vectors as (fact_ids, matrix). Rows of the wrong dim are
        skipped rather than crashing the search — a model change leaves a mixed
        table until backfill catches up."""
        if not _HAS_NUMPY:
            return [], None
        try:
            with self._store._lock:
                rows = self._store._conn.execute(
                    "SELECT fact_id, dim, vector FROM fact_embeddings ORDER BY fact_id"
                ).fetchall()
        except sqlite3.Error:
            return [], None
        if not rows:
            return [], None
        width = max(int(r["dim"]) for r in rows)
        ids, vectors = [], []
        for row in rows:
            vector = unpack(row["vector"])
            if vector.shape[0] != width:
                continue
            ids.append(int(row["fact_id"]))
            vectors.append(vector)
        if not ids:
            return [], None
        return ids, np.vstack(vectors)

    def similar(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """``(fact_id, cosine)`` best-first. Empty on any failure — the caller then
        behaves exactly as it did before dense retrieval existed."""
        if not self.embedder.available:
            return []
        query_vector = self.embedder.embed_one(query)
        if query_vector is None:
            return []
        ids, matrix = self.load_matrix()
        if not ids or matrix is None or matrix.shape[1] != query_vector.shape[0]:
            return []
        scores = matrix @ query_vector
        order = np.argsort(-scores)[:max(1, limit)]
        return [(ids[i], float(scores[i])) for i in order]


def embed_text(content: str, tags: str | None) -> str:
    """The exact text a fact is embedded from.

    Tags are included because they carry the operator's own keywords, which are
    often the words a later question uses. Defined in one place so the hash used
    for staleness always matches the text actually sent to the model.
    """
    return " ".join(f"{content or ''} {tags or ''}".split())
