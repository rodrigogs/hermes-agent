"""CI gate: retrieval quality must not silently regress.

Runs the real FactRetriever over a frozen 104-fact snapshot of the live store
and compares against tests/plugins/memory/eval/baseline.json.

The gate is RATCHETED and ASYMMETRIC:
  * A drop below baseline minus tolerance FAILS. That is the whole point.
  * An improvement PASSES but does not auto-update the baseline. A number that
    updates itself measures nothing — the next regression just moves the bar
    down with it. Updating is a deliberate human act:
        python -m tests.plugins.memory.eval.harness --update-baseline
    which rewrites baseline.json so the improvement shows up in the diff.

Tolerance is one question's worth of recall (1/n_positive) plus epsilon, so
float noise and a single genuinely-ambiguous question cannot redden CI, while a
systematic loss of two or more questions does.

Marked `not integration` compatible: no network, no LLM, no API key. Runtime is
dominated by rebuilding 104 HRR vectors (~0.4 s) plus ~23 ms per query.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")  # HRR term needs numpy; skip rather than fail

from tests.plugins.memory.eval import harness


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memeval")
    return harness.run(tmp)


@pytest.fixture(scope="module")
def baseline():
    with open(harness.BASELINE, encoding="utf-8") as fh:
        return json.load(fh)


def _tol(baseline) -> float:
    """One question's worth of recall, plus float slack."""
    return 1.0 / baseline["n_positive"] + 1e-9


def test_corpus_snapshot_is_intact(report, baseline):
    """The 104 facts must all be there. A shrinking corpus inflates recall."""
    assert report.n_positive == baseline["n_positive"]
    assert report.n_negative == baseline["n_negative"]


def test_recall_at_5_not_below_baseline(report, baseline):
    """recall@5 — 'would the agent have the fact it needs in its prompt?'

    K=5 because prefetch() injects exactly 5 facts. This is the metric that
    tracks the capability; MRR below tracks the ranking within it.
    """
    got, want = report.recall_at_k, baseline["recall_at_k"]
    assert got >= want - _tol(baseline), (
        f"recall@{harness.K} regressed: {got:.3f} < baseline {want:.3f}\n"
        f"newly missing: {sorted(set(report.misses) - set(baseline['known_misses']))}\n"
        "If this is an intentional trade, run:\n"
        "  python -m tests.plugins.memory.eval.harness --update-baseline"
    )


def test_mrr_at_5_not_below_baseline(report, baseline):
    """MRR guards RANK. Recall can hold while the right fact slides to slot 5,
    which in a 5-slot prompt is the last position before it disappears."""
    got, want = report.mrr_at_k, baseline["mrr_at_k"]
    assert got >= want - _tol(baseline), (
        f"MRR@{harness.K} regressed: {got:.3f} < baseline {want:.3f}"
    )


def test_no_new_hard_misses(report, baseline):
    """Named regression list. Aggregate metrics can stay flat while the SET of
    answered questions rotates — trading a fixed question for a broken one. This
    catches that; the known_misses list is the honest record of what is broken."""
    new = sorted(set(report.misses) - set(baseline["known_misses"]))
    assert not new, f"questions that used to pass now miss: {new}"


def test_negative_set_does_not_regress(report, baseline):
    """A retriever that answers everything is indistinguishable from one that
    answers nothing. Negatives are questions with NO supporting fact; anything
    returned is fabricated relevance injected into the system prompt."""
    got, want = report.negative_precision, baseline["negative_precision"]
    assert got >= want - 1e-9, (
        f"negative precision regressed: {got:.3f} < {want:.3f}\n"
        f"newly leaking: {sorted(set(report.negative_leaks) - set(baseline['known_negative_leaks']))}"
    )


def test_known_negative_leaks_are_not_forgotten(report, baseline):
    """The current retriever LEAKS on 5 of 8 negatives (measured). Recording that
    in the baseline is deliberate: it is a documented defect with a number
    attached, not a passing test. This asserts the list does not grow."""
    new = sorted(set(report.negative_leaks) - set(baseline["known_negative_leaks"]))
    assert not new, f"new fabricated-relevance leaks: {new}"


def test_baseline_file_matches_recorded_k(baseline):
    """Guards against comparing a recall@10 number to a recall@5 gate."""
    assert baseline["k"] == harness.K


@pytest.mark.parametrize("qid", ["q003", "q005", "q007", "q019", "q025"])
def test_anchor_questions_rank_first(report, qid):
    """A handful of questions whose target is unambiguous and lexically direct.
    If any of these stops ranking first, the pipeline is broken in a way the
    averaged metrics would mask."""
    qr = next(q for q in report.per_query if q.qid == qid)
    assert qr.first_hit_rank == 1, (
        f"{qid} ({qr.query!r}) expected {qr.expected} at rank 1, got {qr.returned}"
    )


def test_lexical_leakage_of_question_set_is_recorded(report, baseline, tmp_path_factory):
    """Honesty check on the BENCHMARK, not the retriever.

    If the questions reuse rare words from their own target facts, recall@5 is
    measuring string matching. This pins the share of questions winnable by
    rare-token exact match so it cannot creep upward unnoticed as questions are
    added."""
    store = harness.build_corpus(tmp_path_factory.mktemp("leak") / "c.db")
    try:
        retr = harness.make_retriever(store)
        leak = harness.lexical_leakage(store, retr, harness.load_jsonl(harness.QUESTIONS))
    finally:
        store.close()
    recorded = baseline.get("lexical_free_share")
    assert recorded is not None, "baseline is missing lexical_free_share"
    assert leak["free_share"] <= recorded + 1e-9, (
        f"question set got lexically easier: {leak['free_share']:.3f} > {recorded:.3f}. "
        "New questions are reusing rare words from their target facts; the "
        "benchmark is drifting toward measuring string equality."
    )
