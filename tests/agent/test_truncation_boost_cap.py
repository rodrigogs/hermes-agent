"""Tests for the truncation-boost ceiling (models.dev source).

Regression: the continuation-retry path used to cap the boosted ``max_tokens``
at a hard-coded 32 768 — well below the output limits of modern models
(GLM-4.5-Flash: 98 304, GPT-5.x: 131 072, …).  When a long code block or tool
argument truncated, every retry hit the same artificial wall and forced the
agent into the fallback chain, where downstream models started from scratch.

Upstream's ``boosted_output_cap`` already raises the ceiling for
``anthropic_messages`` turns via its adapter.  This module covers the other
half: ``models_dev_output_limit`` resolves the same kind of ceiling for every
provider from the models.dev cache, failing soft (``None`` = "caller keeps its
current ceiling") so the continuation path can never be blocked by a lookup.
"""

from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from agent._truncation_boost_cap import models_dev_output_limit


class TestModelsDevOutputLimit:
    def test_model_output_limit_resolved_from_cache(self):
        """The declared limit.output is returned as an int ceiling."""
        fake_caps = SimpleNamespace(max_output_tokens=98304)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert models_dev_output_limit("zai", "glm-4.5-flash") == 98304

    def test_fallback_floor_when_model_unknown(self):
        """Unknown model / cache miss → None (caller keeps its ceiling)."""
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert models_dev_output_limit("acme", "unknown-model") is None

    def test_fallback_floor_when_lookup_raises(self):
        """A cache lookup failure must never block the continuation path."""
        with patch(
            "agent.models_dev.get_model_capabilities", side_effect=RuntimeError("boom")
        ):
            assert models_dev_output_limit("zai", "glm-5.2") is None

    def test_none_provider_and_model_returns_none(self):
        """Without a provider+model pair there is nothing to look up."""
        assert models_dev_output_limit(None, None) is None
        assert models_dev_output_limit("zai", None) is None
        assert models_dev_output_limit(None, "glm-5.2") is None

    def test_model_output_zero_is_ignored(self):
        """A declared limit of 0 (or falsy) is not a usable ceiling."""
        fake_caps = SimpleNamespace(max_output_tokens=0)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert models_dev_output_limit("zai", "glm-5.2") is None

    def test_model_output_float_is_coerced_to_int(self):
        """Float output limits (e.g. from a future cache schema) are coerced."""
        fake_caps = SimpleNamespace(max_output_tokens=98304.0)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            result = models_dev_output_limit("zai", "glm-4.5-flash")
        assert result == 98304
        assert isinstance(result, int)
