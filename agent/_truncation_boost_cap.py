"""Output-token boost ceiling for truncation retries.

The conversation loop retries a truncated response (``finish_reason='length'``)
by progressively raising ``max_tokens``.  A hard-coded ceiling of 32 768 tokens
used to cap that boost — well below the output limits of modern models
(GLM-4.5-Flash: 98 304, GPT-5.x: 131 072, Claude 4.x: 64 000).

Upstream's ``boosted_output_cap`` owns the ladder (``base·2ⁿ`` growth, the
requested cap as floor, the anthropic_messages adapter as the ceiling source).
This module supplies the one ceiling source that adapter cannot see: the
model's declared ``limit.output`` from the models.dev cache, which covers every
non-Anthropic provider (zai/glm, openai/gpt, …).  Callers take the larger of
the two ceilings so a known limit is never lost and the ladder keeps its room
to grow.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def models_dev_output_limit(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    """The model's declared ``limit.output`` from the models.dev cache, else ``None``.

    Requires both a provider and a model: without either there is nothing to
    look up and the caller keeps whatever ceiling it already had.
    """
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model)
        if caps is not None:
            model_output = getattr(caps, "max_output_tokens", None)
            if isinstance(model_output, (int, float)) and model_output > 0:
                return int(model_output)
    except Exception:
        logger.warning(
            "Could not resolve model output limit for %s/%s",
            provider,
            model,
            exc_info=True,
        )
    return None
