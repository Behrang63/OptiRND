"""
agents/llm_factory.py - provider factory facade (Phase 3).

The canonical factory lives in ``core.llm_provider.LLMFactory`` (agents must
not create import cycles toward core); this facade provides the requested
``agents/llm_factory.py`` entry point with the same contract.

Routing (highest priority first):
    1. explicit argument  - ``get_provider("ollama")`` / ``get_provider("mock")``
    2. environment        - OPTIRND_LLM_PROVIDER=ollama|mock
    3. default            - mock (deterministic offline fallback)

No-crash guarantee (SKILL.md): requesting "ollama" without a running server
still returns a provider object whose every method degrades to mock behavior.
"""
from __future__ import annotations

import os
from typing import Optional

from core.llm_provider import (
    BaseLLMProvider,
    LLMFactory as _CoreLLMFactory,
    MockLLMProvider,
    SchemaValidationError,
)

_ENV_PROVIDER = "OPTIRND_LLM_PROVIDER"


def get_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Resolve an LLM provider instance.

    Order: explicit arg > OPTIRND_LLM_PROVIDER env var > "mock".
    """
    resolved = provider_type or os.environ.get(_ENV_PROVIDER, "mock")
    return _CoreLLMFactory.get_provider(resolved, api_key=api_key)


__all__ = ["get_provider", "BaseLLMProvider", "MockLLMProvider", "SchemaValidationError"]
