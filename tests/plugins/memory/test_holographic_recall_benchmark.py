"""A recall benchmark for the memory, so retrieval changes stop being vibes.

Every retrieval change before this one was argued from a handful of hand-run
queries. This fixes a corpus, a question set and a floor, so a regression fails
CI instead of being noticed months later by an agent that quietly forgot things.

DESIGN, and the reasoning behind each choice:

* THE CORPUS IS SYNTHETIC BUT SHAPED LIKE THE REAL ONE. It cannot be the live
  store: a test that reads /home/rodrigo/... passes or fails depending on whose
  machine it runs on. The 24 facts below are paraphrases of the real corpus's
  SHAPE — model names, hostnames, account ids, hyphenated slugs, mixed
  English/Portuguese — because that shape is what breaks retrievers.

* THE QUESTIONS ARE WRITTEN BY HAND, NOT GENERATED. A question set generated from
  the facts by the same tokenisation the retriever uses measures itself. These are
  phrased the way a person asks, deliberately avoiding the stored wording.

* THERE IS A NEGATIVE SET. A retriever that returns its top-5 for everything scores
  100% on any positive-only benchmark. Questions with no answer in the corpus must
  return nothing.

* THE FLOOR IS A FLOOR, NOT A TARGET. It records what the pipeline achieves today.
  Raising it is a deliberate act; a drop below it is a regression.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

# ── the corpus ─────────────────────────────────────────────────────────
# (content, category, tags). Shaped like the live store: identifiers with
# hyphens and dots, account numbers, hostnames, two languages.
FACTS: list[tuple[str, str, str]] = [
    ("Tier T4 routes to gpt-5.6-terra on openai-codex for the hardest debugging work",
     "model-routing", "router, tier, gpt-5.6-terra"),
    ("Tier T1 is deepseek-v3.2 and handles trivial mechanical edits under 40 lines",
     "model-routing", "router, tier, deepseek"),
    ("The classifier that picks a tier is deepseek-v3.2 on the deepseek rail",
     "model-routing", "classifier"),
    ("Reach the cloud box araponga with ssh -i ~/.ssh/id_oci_araponga ubuntu@163.176.180.233",
     "project", "araponga, ssh, oracle"),
    ("Bedrock account 300094254121 in us-west-2 serves 11 Claude models",
     "provider-config", "bedrock, us-west-2"),
    ("Claude Code runs as a provider through the slug copilot-acp, not GitHub Copilot",
     "provider-config", "copilot-acp, claude"),
    ("openrouter free models return HTTP 429 once the daily quota is spent",
     "provider-config", "openrouter, quota"),
    ("The compaction threshold is 208352 tokens against a 272000 token window",
     "project", "compaction, threshold"),
    ("Parrot Mambo drone control works over USB with mamboctl and usbipd-win",
     "project", "parrot mambo, drone, mamboctl"),
    ("The Avell G1555 laptop needed an ACPI override in HKLM to boot cleanly",
     "tool", "avell, acpi"),
    ("Every Kanban profile is capped at max_in_progress_per_profile=1",
     "tool", "kanban"),
    ("The capability-router sidecar listens on 127.0.0.1:8791 behind a token gate",
     "tool", "capability-router, sidecar"),
    ("Updating hermes needs three components restarted: backend, webui and office",
     "tool", "update, restart"),
    ("Sessions are stored under the profile directory in webui/sessions",
     "tool", "sessions"),
    ("glm-4.7-flash on z.ai is the cheap high-volume option at 0.06 per million",
     "provider-config", "glm-4.7-flash, zai, pricing"),
    ("O bridge copilot-acp foi reparado e roda via hermes-mcp-gate.sh no Mac",
     "provider-config", "copilot-acp, mac"),
    ("A sessao carrega muito mais rapido depois que o cache foi corrigido",
     "tool", "sessao, cache"),
    ("Patches commitados mas nao pushados se perdem quando hermes update roda",
     "tool", "hermes update, git"),
    ("The a1-hunter cron polls Oracle every 15 minutes for Ampere A1 capacity",
     "project", "a1-hunter, oracle"),
    ("Vercel refuses a deploy when the commit is not on the tracked branch",
     "project", "vercel, deploy"),
    ("The holographic memory store keeps facts in SQLite with an FTS5 index",
     "tool", "memory, sqlite, fts5"),
    ("Rodrigo prefers TypeScript and Node.js for new service work",
     "user_pref", "typescript, node"),
    ("The office 3D view runs on port 3100 and serves under /office/",
     "tool", "office, port"),
    ("Trust below 0.3 makes a fact invisible to retrieval by default",
     "tool", "trust, retrieval"),
]

# ── the questions ──────────────────────────────────────────────────────
# (question, substring that identifies an acceptable answer). Hand-written,
# phrased away from the stored wording.
POSITIVE: list[tuple[str, str]] = [
    ("which model should take the hardest debugging job", "gpt-5.6-terra"),
    ("what handles small mechanical edits", "deepseek-v3.2"),
    ("how do I log into the cloud machine", "araponga"),
    ("which AWS account has the Claude models", "300094254121"),
    ("what happens when the free models run out", "429"),
    ("at how many tokens does it compact", "208352"),
    ("how is the drone driven", "mamboctl"),
    ("what was wrong with the laptop firmware", "ACPI"),
    ("how many tasks can one profile have in flight", "max_in_progress"),
    ("what port is the router sidecar on", "8791"),
    ("what has to be restarted after an upgrade", "restarted"),
    ("where do conversations get saved", "webui/sessions"),
    ("what is the cheapest model for bulk work", "glm-4.7-flash"),
    ("qual e o bridge para o Claude no Mac", "copilot-acp"),
    ("por que a sessao ficou mais rapida", "cache"),
    ("o que acontece com commits nao enviados", "pushados"),
    ("what polls for spare Ampere capacity", "a1-hunter"),
    ("why would a deployment be rejected", "Vercel"),
    ("where are the memories actually kept", "SQLite"),
    ("which language does he like for services", "TypeScript"),
    ("what makes a memory stop being retrieved", "0.3"),
    # Paraphrases with no shared vocabulary — the dense path's reason to exist.
    ("is there anything about flying machines", "Mambo"),
    ("which thing bridges to anthropic", "copilot-acp"),
    ("quanto tempo leva pra compactar", "208352"),
]

# Questions the corpus genuinely cannot answer. A retriever that always returns
# five things scores 100% on the positive set alone.
NEGATIVE: list[str] = [
    "what is my mother's maiden name",
    "how do I bake sourdough bread",
    "what is the capital of Mongolia",
    "which cryptocurrency should I buy",
]

# Recorded from the pipeline as it stands. A drop is a regression; raising this is
# a deliberate edit accompanied by the change that earned it.
#
# Measured on this corpus and question set:
#     lexical only        recall@5 0.67, MRR 0.625, 0/4 false positives
#     + dense fusion      recall@5 0.92, MRR 0.847, 0/4 false positives
#
# The floor is set to the LEXICAL number, deliberately. The dense path depends on
# a local ollama endpoint that may be absent on another machine or in CI, and a
# benchmark that fails when an optional dependency is missing teaches people to
# ignore it. test_dense_fusion_is_an_improvement_when_available asserts the higher
# bar, and skips honestly when it cannot.
BASELINE_RECALL_AT_5 = 0.66
BASELINE_MRR = 0.60
# What fusion must reach when the embedder IS available.
DENSE_RECALL_AT_5 = 0.87
DENSE_MRR = 0.80


@pytest.fixture(scope="module")
def bench_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "memory_store.db"
    store = MemoryStore(path)
    for content, category, tags in FACTS:
        store.add_fact(content, category=category, tags=tags)
    yield store
    store.close()


@pytest.fixture(scope="module")
def lexical(bench_store):
    return FactRetriever(bench_store, temporal_decay_half_life=0, fts_weight=0.55,
                         jaccard_weight=0.15, hrr_weight=0.3, hrr_dim=1024)


def _measure(retriever, store) -> dict:
    facts = {r["fact_id"]: r["content"] for r in store.list_facts(min_trust=0.0, limit=500)}

    def targets(needle: str) -> set[int]:
        return {fid for fid, text in facts.items() if needle.lower() in text.lower()}

    hits, reciprocal, missed = 0, [], []
    for question, needle in POSITIVE:
        want = targets(needle)
        assert want, f"benchmark bug: no corpus fact contains {needle!r}"
        got = [h["fact_id"] for h in retriever.search(question, min_trust=0.3, limit=5)]
        rank = next((i + 1 for i, fid in enumerate(got) if fid in want), None)
        if rank:
            hits += 1
            reciprocal.append(1.0 / rank)
        else:
            reciprocal.append(0.0)
            missed.append(question)

    noise = sum(1 for q in NEGATIVE if retriever.search(q, min_trust=0.3, limit=5))
    return {
        "recall_at_5": hits / len(POSITIVE),
        "mrr": sum(reciprocal) / len(POSITIVE),
        "missed": missed,
        "false_positive_questions": noise,
    }


def test_recall_at_5_does_not_regress(lexical, bench_store):
    """The floor. If this fails, retrieval got worse — find out why before
    lowering the number."""
    report = _measure(lexical, bench_store)
    assert report["recall_at_5"] >= BASELINE_RECALL_AT_5, (
        f"recall@5 fell to {report['recall_at_5']:.2f} (floor {BASELINE_RECALL_AT_5});"
        f" missed: {report['missed']}"
    )


def test_mean_reciprocal_rank_does_not_regress(lexical, bench_store):
    """Recall alone hides ranking: a fact at position 5 is nearly useless when the
    prefetch hook injects the top 5 and the model reads the first."""
    report = _measure(lexical, bench_store)
    assert report["mrr"] >= BASELINE_MRR, \
        f"MRR fell to {report['mrr']:.3f} (floor {BASELINE_MRR})"


def test_the_retriever_abstains_on_questions_it_cannot_answer(lexical, bench_store):
    """The guard against a retriever that scores well by never abstaining."""
    report = _measure(lexical, bench_store)
    assert report["false_positive_questions"] == 0, (
        f"{report['false_positive_questions']} of {len(NEGATIVE)} unanswerable"
        " questions returned memories"
    )


def test_the_benchmark_itself_is_answerable(bench_store):
    """A fence against silent rot: if a corpus edit removes the fact a question
    depends on, that question would 'fail' forever for the wrong reason."""
    facts = [r["content"].lower() for r in bench_store.list_facts(min_trust=0.0, limit=500)]
    for question, needle in POSITIVE:
        assert any(needle.lower() in text for text in facts), \
            f"{question!r} has no answer in the corpus (looked for {needle!r})"
    for question in NEGATIVE:
        assert question, "negative questions must not be empty"


def _dense_retriever(store):
    """A retriever with dense fusion attached, or None if no embedder is here."""
    from plugins.memory.holographic.embeddings import EmbeddingIndex

    index = EmbeddingIndex(store)
    if not index.embedder.available:
        return None
    index.backfill()
    retriever = FactRetriever(store, temporal_decay_half_life=0, fts_weight=0.55,
                              jaccard_weight=0.15, hrr_weight=0.3, hrr_dim=1024,
                              dense_weight=0.4)
    retriever.attach_dense(index)
    return retriever


def test_dense_fusion_is_an_improvement_when_available(bench_store):
    """Dense retrieval must EARN its place, not merely not hurt.

    Measured: it lifts recall@5 from 0.67 to 0.92 and MRR from 0.625 to 0.847 on
    this set, by answering questions whose vocabulary does not overlap the stored
    fact at all ("where are the memories actually kept", "which thing bridges to
    anthropic", two Portuguese questions).
    """
    retriever = _dense_retriever(bench_store)
    if retriever is None:
        pytest.skip("no embedding endpoint on this machine")

    report = _measure(retriever, bench_store)

    assert report["recall_at_5"] >= DENSE_RECALL_AT_5, (
        f"fusion recall@5 {report['recall_at_5']:.2f} below {DENSE_RECALL_AT_5};"
        f" missed: {report['missed']}"
    )
    assert report["mrr"] >= DENSE_MRR, f"fusion MRR {report['mrr']:.3f} below {DENSE_MRR}"


def test_dense_fusion_does_not_fabricate_relevance(bench_store):
    """The reason there is a similarity floor.

    An embedding always returns its nearest neighbours, so with no floor recall
    hit 1.00 while ALL FOUR unanswerable questions returned memories. That is
    fabrication dressed as recall, and it is worse than a miss: the agent would
    state something irrelevant with the confidence of a remembered fact.
    """
    retriever = _dense_retriever(bench_store)
    if retriever is None:
        pytest.skip("no embedding endpoint on this machine")

    report = _measure(retriever, bench_store)

    assert report["false_positive_questions"] == 0, (
        f"{report['false_positive_questions']} of {len(NEGATIVE)} unanswerable"
        " questions were answered — lower _DENSE_MIN_SIM at your peril"
    )


def test_disabling_dense_leaves_the_lexical_pipeline_untouched(bench_store):
    """The fail-open contract: on a machine with no embedder, or with the feature
    off, behaviour must be exactly what it was before any of this existed."""
    plain = FactRetriever(bench_store, temporal_decay_half_life=0, fts_weight=0.55,
                          jaccard_weight=0.15, hrr_weight=0.3, hrr_dim=1024,
                          dense_weight=0.4)
    assert not plain.dense_available, "no index attached means no dense path"

    report = _measure(plain, bench_store)
    assert report["recall_at_5"] >= BASELINE_RECALL_AT_5
    assert report["false_positive_questions"] == 0
