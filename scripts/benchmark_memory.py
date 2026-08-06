#!/usr/bin/env python3
"""Hermes Memory Benchmark — systematic recall evaluation.

Evaluates the holographic fact_store against a curated test set of
queries with known expected fact_ids. Measures recall@k, MRR, and
per-signal breakdown (dense_sim, entity_boost, fts_rank).

Usage:
    python3 scripts/benchmark_memory.py [--verbose]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Test set: (query, expected_fact_id, description)
# Each query has ONE expected fact that SHOULD rank #1.
# Queries cover: EN + PT-BR, literal + paraphrase, short + long.
# ---------------------------------------------------------------------------
QUERIES = [
    # ── EN, literal ──
    (
        "dense retrieval qwen3 embedding benchmark recall",
        182,
        "EN literal: dense retrieval upgrade",
    ),
    (
        "gateway restart wsl execute_code systemctl",
        76,
        "EN literal: gateway restart procedure",
    ),
    (
        "default model provider hermes gpt-5.6-terra",
        132,
        "EN literal: default model config",
    ),
    (
        "how to update hermes without losing local patches merge upstream",
        159,
        "EN query: update procedure preserving fork commits",
    ),
    (
        "compaction threshold track B dynamic formula",
        149,
        "EN literal: compaction thresholds",
    ),
    # ── PT-BR, literal ──
    (
        "como reiniciar o gateway hermes",
        76,
        "PT-BR literal: gateway restart",
    ),
    (
        "qual o modelo padrão do hermes",
        132,
        "PT-BR literal: default model",
    ),
    (
        "como atualizar o hermes sem perder alterações",
        159,
        "PT-BR literal: update procedure",
    ),
    # ── PT-BR, paraphrase ──
    (
        "qual é o jeito certo de reiniciar o serviço do gateway",
        76,
        "PT-BR paraphrase: gateway restart",
    ),
    (
        "recuperação de memória melhorada com embeddings",
        182,
        "PT-BR paraphrase: dense retrieval",
    ),
    # ── Short queries ──
    (
        "hrr retrieval fix",
        78,
        "Short EN: HRR bug fix",
    ),
    (
        "copilot bridge claude mac",
        66,
        "Short EN: copilot-acp bridge",
    ),
    # ── Edge cases ──
    (
        "kanban authority policy profiles role guard",
        68,
        "EN: kanban authority policy",
    ),
    (
        "NoNewPrivileges sandbox webui sudo block",
        76,
        "EN variant: sandbox security",
    ),
    (
        "auto_extract regex PT-BR limitation extraction multilíngue",
        93,
        "EN: auto_extract design note",
    ),
]

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def load_provider():
    """Initialize a fresh holographic provider against the real DB."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "memory" / "holographic"))
    from plugins.memory.holographic import HolographicMemoryProvider

    from hermes_constants import get_hermes_home
    db_path = str(get_hermes_home() / "memory_store.db")

    config = {"db_path": db_path, "dense_retrieval": True}
    provider = HolographicMemoryProvider(config=config)
    provider.initialize("benchmark-session")
    return provider


def run_query(provider, query: str, k: int = 5) -> dict:
    """Execute one query and return the top-k results."""
    result = json.loads(
        provider._handle_fact_store({
            "action": "search",
            "query": query,
            "limit": k,
        })
    )
    return result


def benchmark(verbose: bool = False):
    provider = load_provider()
    try:
        total = len(QUERIES)
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        mrr_sum = 0.0
        dense_count = 0
        entity_boost_count = 0
        errors = 0
        total_time = 0.0

        print(f"Benchmark: {total} queries against live fact_store\n")

        for i, (query, expected_id, desc) in enumerate(QUERIES):
            t0 = time.monotonic()
            try:
                result = run_query(provider, query, k=5)
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [{i+1:2d}] ERROR: {desc} — {exc}")
                continue
            elapsed = time.monotonic() - t0
            total_time += elapsed

            results = result.get("results", [])
            found_ids = [r["fact_id"] for r in results]
            rank = found_ids.index(expected_id) + 1 if expected_id in found_ids else None

            if rank == 1:
                hits_at_1 += 1
            if rank is not None and rank <= 3:
                hits_at_3 += 1
            if rank is not None and rank <= 5:
                hits_at_5 += 1
            if rank is not None:
                mrr_sum += 1.0 / rank

            # Signal breakdown
            top = results[0] if results else {}
            has_dense = "dense_sim" in top
            has_eb = top.get("entity_boost", False)
            if has_dense:
                dense_count += 1
            if has_eb:
                entity_boost_count += 1

            status = "✅" if rank == 1 else ("⚠️ #" + str(rank) if rank else "❌")
            if verbose or rank != 1:
                print(
                    f"  [{i+1:2d}] {status} {desc} "
                    f"(#{expected_id} rank={rank}, score={top.get('score', 0):.3f}, "
                    f"dense={has_dense}, eb={has_eb})"
                )

        # Summary
        print(f"\n{'='*60}")
        print(f"Results: {total} queries, {errors} errors")
        print(f"{'='*60}")
        print(f"  Recall@1:  {hits_at_1}/{total} = {hits_at_1/total*100:.1f}%")
        print(f"  Recall@3:  {hits_at_3}/{total} = {hits_at_3/total*100:.1f}%")
        print(f"  Recall@5:  {hits_at_5}/{total} = {hits_at_5/total*100:.1f}%")
        print(f"  MRR:       {mrr_sum/total:.4f}")
        print(f"  Avg time:  {total_time/total*1000:.1f}ms/query")
        print(f"  Dense active:  {dense_count}/{total}")
        print(f"  Entity boost:  {entity_boost_count}/{total}")

        return {
            "total": total,
            "errors": errors,
            "recall_at_1": hits_at_1 / total,
            "recall_at_3": hits_at_3 / total,
            "recall_at_5": hits_at_5 / total,
            "mrr": mrr_sum / total,
            "avg_ms": total_time / total * 1000,
            "dense_ratio": dense_count / total,
            "entity_boost_ratio": entity_boost_count / total,
        }
    finally:
        provider.shutdown()


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    benchmark(verbose=verbose)
