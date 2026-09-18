"""
Ollama LLM provider (Phase 3 - Real AI Stack Integration).

Breaks the exclusive dependency on MockLLMProvider while preserving the
project's no-crash guarantee (SKILL.md): every failure mode - server absent,
connection refused, HTTP error, malformed JSON, schema violation - degrades
gracefully to MockLLMProvider behavior instead of raising.

Design:
    * Talks to the local Ollama REST API (default http://localhost:11434)
      using ONLY the standard library (urllib) - zero new dependencies.
    * generate_json(): prompts the model for a JSON object, extracts the
      first JSON block from the response, and validates it against a
      Pydantic v2 schema with bounded retries. On exhaustion it falls back
      to MockLLMProvider.
    * is_available(): cheap reachability probe for health checks / factory
      routing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from core.llm_provider import BaseLLMProvider, MockLLMProvider

logger = logging.getLogger("optirnd.ollama")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2          # initial attempt + N retries (bounded)

# Environment configuration (documented contract):
#   OPTIRND_OLLAMA_HOST   - base URL of the Ollama server
#   OPTIRND_OLLAMA_MODEL  - model tag (e.g. llama3, qwen2)
_ENV_HOST = "OPTIRND_OLLAMA_HOST"
_ENV_MODEL = "OPTIRND_OLLAMA_MODEL"

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _host() -> str:
    return os.environ.get(_ENV_HOST, DEFAULT_OLLAMA_HOST).rstrip("/")


def _model() -> str:
    return os.environ.get(_ENV_MODEL, DEFAULT_MODEL)


class OllamaLLMProvider(BaseLLMProvider):
    """
    Local Ollama-backed provider with schema-bound JSON generation and an
    unconditional mock fallback. Never raises on LLM failure paths.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model or _model()
        self.host = (host or _host()).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def is_available(self, timeout_seconds: Optional[float] = None) -> bool:
        """Cheap reachability probe against the Ollama server root."""
        try:
            req = urllib.request.Request(f"{self.host}/", method="GET")
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds if timeout_seconds is None
                else float(timeout_seconds)
            ):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def generate_json(self, prompt: str, schema_description: str = "") -> Dict[str, Any]:
        """
        Legacy single-shot contract (no schema bound).

        Contract-first rule: this method performs NO schema validation, so its
        output must never enter the pipeline. It therefore always resolves to
        the deterministic MockLLMProvider contract - raw model output is never
        returned from an unbound call. Schema-bound callers must use
        generate_validated() to get real Ollama output.
        """
        logger.warning(
            "Ollama generate_json called without a schema model; "
            "returning mock contract (use generate_validated for real output)"
        )
        return self._fallback().generate_json(prompt, schema_description)

    def generate_validated(
        self,
        prompt: str,
        schema_description: str,
        schema_model: Type[BaseModel],
    ) -> Dict[str, Any]:
        """
        Schema-bound generation: prompt Ollama for JSON, extract it, and
        validate against ``schema_model`` (Pydantic v2) with bounded retries.
        On exhaustion, falls back to the deterministic MockLLMProvider.
        """
        instruction = (
            f"{prompt}\n\nRespond with ONLY a valid JSON object that conforms "
            f"to this schema: {schema_description or schema_model.__name__}. "
            "No prose, no markdown fences."
        )

        last_error: Optional[str] = None
        for attempt in range(1 + self.max_retries):
            raw_text = self._chat(instruction)
            if raw_text is None:
                last_error = "ollama_unreachable_or_error"
                continue

            payload = self._extract_json(raw_text)
            if payload is None:
                last_error = "malformed_json"
                logger.warning(
                    "Ollama returned unparseable JSON (attempt %d)",
                    attempt + 1,
                    extra={"model": self.model},
                )
                continue

            try:
                validated = schema_model.model_validate(payload)
            except ValidationError as exc:
                last_error = "schema_validation_failed"
                logger.warning(
                    "Ollama payload failed schema validation (attempt %d): %s",
                    attempt + 1,
                    exc.error_count(),
                    extra={"model": self.model},
                )
                continue

            return validated.model_dump()

        logger.warning(
            "Ollama generation failed after retries; falling back to mock",
            extra={"model": self.model, "reason": last_error},
        )
        return self._fallback().generate_json(prompt, schema_description)

    def generate_text(self, prompt: str) -> str:
        """Plain-text generation; falls back to the mock on any failure."""
        raw_text = self._chat(prompt)
        if raw_text is None:
            logger.warning("Ollama text generation failed; falling back to mock",
                           extra={"model": self.model})
            return self._fallback().generate_text(prompt)
        return raw_text.strip()

    def interpret_monte_carlo(self, metrics: Dict[str, Any], p_success: float) -> str:
        """Managerial interpretation; falls back to the mock's deterministic text."""
        prompt = (
            f"Interpret these Monte Carlo results for an executive audience: "
            f"mean_roi={metrics.get('mean_roi')}%, var_95={metrics.get('var_95')}%, "
            f"p_success={p_success}. Answer in Persian, 4 concise bullet points."
        )
        return self.generate_text(prompt)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fallback(self) -> MockLLMProvider:
        return MockLLMProvider()

    def _chat(self, prompt: str) -> Optional[str]:
        """Call Ollama /api/generate; return the response text or None."""
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                envelope = json.loads(resp.read().decode("utf-8"))
            return str(envelope.get("response", ""))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            logger.warning(
                "Ollama request failed",
                extra={"model": self.model, "error_class": type(exc).__name__},
            )
            return None

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the first JSON object from a model response. Tolerates prose
        wrappers and markdown fences; returns None when nothing parses.
        """
        if not raw_text:
            return None
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
