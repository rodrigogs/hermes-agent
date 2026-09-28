# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

## Requirements

None — uses SQLite (always available). NumPy optional for HRR algebra.

## Setup

```bash
hermes memory setup    # select "holographic"
```

Or manually:
```bash
hermes config set memory.provider holographic
```

## Config

Config in `config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `llm_extract` | `false` | Also extract facts with the auxiliary model at session end; requires `auto_extract: true` (see Model routing) |
| `llm_extract_max_messages` | `20` | Most recent eligible user messages sent to the model per session end (minimum 1; use `llm_extract: false` to disable model calls) |
| `llm_extract_max_seconds` | `60` | Wall-clock budget for the model calls per session end; no new call is issued past it and the rest falls to regex (minimum 1) |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |

## Model routing (`llm_extract`)

`llm_extract` requires `auto_extract: true` — with `auto_extract` off,
`on_session_end` returns before any extraction runs, so `llm_extract` alone has
no effect. With both true, every user message of at least 100 characters —
capped to the most recent `llm_extract_max_messages` per session end — is sent
to the shared auxiliary client under the task key **`memory_extract`**. The
plugin names no provider, model or API-key variable; routing is config:

```yaml
auxiliary:
  memory_extract:
    provider: bedrock
    model: us.anthropic.claude-haiku-4-5-20251001-v1:0
    timeout: 20
```

Leave the block out and the task runs with `provider: auto`, meaning the main
provider and main chat model — which works everywhere but is usually the
expensive rail, so pinning a cheap model is recommended. `memory_extract` is
registered when the provider loads, so this block is read from `config.yaml` in
the agent process; memory providers sit outside general plugin discovery, so the
task is configured by editing `config.yaml` rather than through the
`hermes model` picker.

Any OpenAI-compatible endpoint still works, including the z.ai route this used
to be hardcoded to:

```yaml
auxiliary:
  memory_extract:
    provider: custom
    base_url: https://api.z.ai/api/coding/paas/v4
    model: glm-4.5-flash
    key_env: ZAI_API_KEY
```

If the route fails (timeout, connection, auth), the plugin logs **one** WARNING
naming the task and the exception and finishes the session with the regex
extractor only. An HTTP 400 is treated as the route rejecting *that* message
(too long for the model, content filter): the message is skipped, the pass
continues, and the first such rejection per session end is a WARNING (later
ones DEBUG). The check is SDK-independent — `openai.BadRequestError` on an
OpenAI-compatible route, `anthropic.BadRequestError` or a botocore
`ValidationException` on the Bedrock route above; Bedrock's
`ThrottlingException` (also HTTP 400) stays a route failure. Messages over
4 000 characters are sent as head + tail. Model
output goes through the same write guard as an explicit `fact_store add`
(threat scan, secret/PII redaction), and compaction handoff summaries are never
sent.

**Where it runs, and for how long.** `on_session_end` reaches this pass on the
memory manager's background worker for `/new`, but **inline on the agent's
thread** at context compaction (`commit_memory_session`) and on CLI exit /
every oneshot run (`shutdown_memory_provider`). The pass therefore blocks
compaction or exit for at most `llm_extract_max_seconds` plus the wall time of
**one** `call_llm`, and never for `llm_extract_max_messages × timeout`. One
`call_llm` is not one `timeout`, though: on a timeout or connection error the
auxiliary client retries the same provider `auxiliary.transient_retries`
times (default 2, with 1 s / 2 s backoff) and then walks its provider/model
fallback chain, so on defaults a dead route costs 60 s + 3 × 20 s + 3 s ≈ 123 s
before any fallback attempts — not 80 s. Lower either knob, `timeout`, or
`auxiliary.transient_retries` if measured compaction latency matters. When the
model finds no facts and the route is healthy, nothing is logged above DEBUG
(`holographic llm_extract: N auxiliary calls, M facts stored`).

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
