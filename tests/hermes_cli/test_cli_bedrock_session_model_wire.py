"""CLI startup must resolve Bedrock's dual wire for the SESSION model, not the config default.

Regression for NousResearch/hermes-agent#50292 (fix PR #96457).  Backported to stack/main
after it was reproduced live on 2026-09-08: a kanban worker spawned with
``-m zai.glm-4.7-flash --provider bedrock`` on a profile whose ``model.default`` is a
Claude id went out on the AnthropicBedrock wire, Bedrock answered
``HTTP 400 ... Invalid 'tools': missing field 'type'``, and the fallback chain silently
replaced the requested model with Sonnet.

The resolver half already honours ``target_model`` (``_current_model = target_model or
default``); the defect is the startup caller, ``_ensure_runtime_credentials``, which never
passed it.  These tests drive that caller with the real ``resolve_runtime_provider`` and a
Claude default in config, and assert on the ``api_mode`` the CLI ends up with — the one
thing that decides which wire the first request uses.
"""

from __future__ import annotations

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


CLAUDE_DEFAULT = "us.anthropic.claude-opus-5"


class _RuntimeCLI(CLIAgentSetupMixin):
    """The minimal HermesCLI shell ``_ensure_runtime_credentials`` reads and writes."""

    def __init__(self, *, model: str, provider: str):
        self.model = model
        self.requested_provider = provider
        self.provider = provider
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self.service_tier = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


@pytest.fixture
def bedrock_claude_default(monkeypatch):
    """A profile whose default is Claude on Bedrock, with credentials the resolver accepts.

    ``requested_provider="bedrock"`` is an *explicit* selection, so the resolver trusts
    boto3's credential chain instead of probing for keys; the env vars below only keep
    boto3 itself from wandering into IMDS/SSO lookups inside the test.
    """
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        f"""
model:
  default: {CLAUDE_DEFAULT}
  provider: bedrock
bedrock:
  region: us-west-2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTNOTREAL0000000")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-real")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    from hermes_cli import config as _config

    # The resolver reads config through the mtime-cached loader; make sure the file
    # written above is what it sees rather than a cache entry from an earlier test.
    for name in ("_clear_config_cache", "clear_config_cache", "invalidate_config_cache"):
        fn = getattr(_config, name, None)
        if callable(fn):
            fn()
            break


def _startup(model: str) -> _RuntimeCLI:
    cli = _RuntimeCLI(model=model, provider="bedrock")
    assert cli._ensure_runtime_credentials() is True
    return cli


def test_non_claude_session_model_uses_converse_despite_claude_default(bedrock_claude_default):
    # The live failure: -m names a Converse-only model, the profile default is Claude.
    cli = _startup("zai.glm-4.7-flash")
    assert cli.api_mode == "bedrock_converse"
    assert cli.provider == "bedrock"
    # The model the session runs is the one that was asked for, not the default.
    assert cli.model == "zai.glm-4.7-flash"


def test_claude_session_model_keeps_anthropic_wire(bedrock_claude_default):
    # Control: a Claude -m on a Claude default must not regress to Converse.
    cli = _startup("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert cli.api_mode == "anthropic_messages"
    assert cli.provider == "bedrock"


def test_regional_and_global_prefixes_still_count_as_claude(bedrock_claude_default):
    # ``is_anthropic_bedrock_model`` strips the inference-profile prefix; passing
    # target_model must not bypass that normalisation.
    for model in ("global.anthropic.claude-sonnet-5", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"):
        cli = _startup(model)
        assert cli.api_mode == "anthropic_messages", model
