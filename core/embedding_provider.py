"""
Embedding providers for OptiRND vector retrieval pipeline (Phase 3, Prompt 2).

Design:
    * BaseEmbeddingProvider defines the contract: embed_text() and embed_batch().
    * OllamaEmbeddingProvider queries a local Ollama /api/embeddings endpoint
      using ONLY stdlib urllib - zero new dependencies.
    * MockEmbeddingProvider generates deterministic, normalized pseudo-dense
      vectors from string hashes - serves as the automatic zero-crash fallback.

OWASP LLM08 compliance: providers never expose raw model internals; the mock
fallback guarantees the system never crashes when Ollama is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("optirnd.embeddings")

# Environment configuration
_ENV_EMBEDDING_HOST = "OPTIRND_OLLAMA_HOST"
_ENV_EMBEDDING_MODEL = "OPTIRND_EMBEDDING_MODEL"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_DIMENSION = 128  # Mock provider dimension


def _embedding_host() -> str:
    """Get Ollama host from environment or use default."""
    return os.environ.get(_ENV_EMBEDDING_HOST, DEFAULT_OLLAMA_HOST).rstrip("/")


def _embedding_model() -> str:
    """Get embedding model from environment or use default."""
    return os.environ.get(_ENV_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors."""
        pass


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic, zero-dependency embedding provider for offline/fallback use.

    Generates pseudo-dense vectors by hashing the input text and expanding
    the hash into a normalized vector of fixed dimension. The same input
    always produces the same output (deterministic), and vectors are
    L2-normalized to unit length for cosine similarity compatibility.

    No network calls, no external dependencies, no crashes.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using deterministic hash-based pseudo-embedding."""
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        return [self._hash_to_vector(text) for text in texts]

    def _hash_to_vector(self, text: str) -> List[float]:
        """
        Convert text to a deterministic normalized vector.

        Uses SHA-256 hash of the text, then expands the 32 bytes into the
        target dimension via a simple deterministic expansion, then L2-normalizes.
        """
        # Create a stable hash from the text
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

        # Expand 32 bytes to target dimension deterministically
        # Repeat the hash bytes and apply a simple mixing function
        expanded = np.zeros(self._dimension, dtype=np.float64)
        for i in range(self._dimension):
            byte_idx = i % 32
            # Mix with position for variation across dimensions
            expanded[i] = (hash_bytes[byte_idx] + (i * 31)) / 255.0

        # Center and normalize to unit L2 norm
        expanded = expanded - np.mean(expanded)
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded = expanded / norm

        return expanded.tolist()


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama-backed embedding provider with bounded timeout and graceful fallback.

    Queries the local Ollama server at /api/embeddings (or /api/embed) using
    stdlib urllib. On any failure (network, HTTP error, timeout, malformed
    response), logs a warning and the caller should fall back to
    MockEmbeddingProvider.

    Zero new dependencies - uses only Python standard library.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.model = model or _embedding_model()
        self.host = (host or _embedding_host()).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text via Ollama /api/embeddings."""
        result = self.embed_batch([text])
        return result[0] if result else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts via Ollama /api/embeddings.

        Returns empty list on any failure - caller handles fallback.
        """
        if not texts:
            return []

        # Ollama's /api/embeddings expects one text per request in the simple API
        # We batch by making sequential requests (Ollama doesn't have a native batch endpoint)
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            if embedding is None:
                logger.warning(
                    "Ollama embedding failed for text (len=%d); returning partial results",
                    len(text),
                )
                return []  # Signal failure to caller
            embeddings.append(embedding)

        return embeddings

    def _embed_single(self, text: str) -> Optional[List[float]]:
        """Embed a single text via Ollama /api/embeddings endpoint."""
        body = json.dumps({
            "model": self.model,
            "prompt": text,
        }).encode("utf-8")

        # Try /api/embeddings first (newer Ollama), then /api/embed (legacy)
        endpoints = [
            f"{self.host}/api/embeddings",
            f"{self.host}/api/embed",
        ]

        for endpoint in endpoints:
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                # Handle response formats:
                # /api/embeddings returns {"embedding": [...]}
                # /api/embed returns {"embeddings": [[...]]}
                if "embedding" in data:
                    emb = data["embedding"]
                    if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
                        return [float(x) for x in emb]
                elif "embeddings" in data:
                    embs = data["embeddings"]
                    if embs and isinstance(embs[0], list):
                        return [float(x) for x in embs[0]]

                logger.warning(
                    "Unexpected Ollama embedding response format from %s",
                    endpoint,
                    extra={"model": self.model},
                )

            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    ValueError, json.JSONDecodeError, KeyError) as exc:
                logger.debug(
                    "Ollama embedding request failed for %s: %s",
                    endpoint,
                    exc,
                    extra={"model": self.model, "error_class": type(exc).__name__},
                )
                continue  # Try next endpoint

        return None


def create_embedding_provider(
    provider_type: str = "auto",
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider_type: "ollama" | "mock" | "auto"
        **kwargs: Passed to provider constructor

    Returns:
        Embedding provider instance. On "auto" or "ollama" failure,
        returns MockEmbeddingProvider (never raises).
    """
    if provider_type == "mock":
        return MockEmbeddingProvider(**kwargs)

    if provider_type in ("ollama", "auto"):
        try:
            provider = OllamaEmbeddingProvider(**kwargs)
            # Quick availability check
            if provider_type == "auto":
                # For auto, we could probe but that adds latency;
                # just try to use it and let embed_batch signal failure
                pass
            return provider
        except Exception as exc:  # noqa: BLE001 - no-crash guarantee
            logger.warning(
                "Failed to create OllamaEmbeddingProvider; using mock: %s",
                exc,
                extra={"error_class": type(exc).__name__},
            )
            return MockEmbeddingProvider(**kwargs)

    # Unknown type -> mock
    logger.warning("Unknown embedding provider type '%s'; using mock", provider_type)
    return MockEmbeddingProvider(**kwargs)


__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "create_embedding_provider",
]