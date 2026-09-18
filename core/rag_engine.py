"""
RAG Engine for OptiRND (Phase 3, Prompt 2).

Implements a resilient, zero-dependency vector retrieval pipeline:
- Recursive text chunking (~500 chars, ~50 overlap)
- Sanitization to strip hidden control characters (OWASP LLM01/LLM08)
- Dependency-injected embedding provider and vector store
- Failure containment: returns empty list on zero matches or internal error

No external dependencies beyond stdlib, NumPy, and optional SQLite/JSON persistence.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.embedding_provider import BaseEmbeddingProvider, MockEmbeddingProvider, create_embedding_provider
from core.vector_store import LocalVectorStore, RetrievedChunk, ChunkRecord, create_vector_store

logger = logging.getLogger("optirnd.rag")

# Control character regex for sanitization (OWASP LLM01/LLM08)
# Matches: null, vertical tab, form feed, and other C0/C1 control chars except \n, \r, \t
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MIN_SIMILARITY = 0.60


class RAGEngine:
    """
    Vector-based Retrieval-Augmented Generation engine.

    Accepts injected embedding_provider and vector_store for testability and
    flexibility. Provides document indexing and context retrieval with
    comprehensive failure containment.
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[LocalVectorStore] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        dimension: Optional[int] = None,
    ):
        """
        Initialize RAG engine with injected dependencies.

        Args:
            embedding_provider: Embedding provider instance. Defaults to mock.
            vector_store: Vector store instance. Defaults to local SQLite+NPZ store.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            min_similarity: Minimum cosine similarity for retrieval.
            dimension: Explicit embedding dimension (used if vector_store not provided).
        """
        self.chunk_size = max(100, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.min_similarity = float(min_similarity)

        # Dependency injection with safe defaults
        if embedding_provider is None:
            if dimension is not None:
                self.embedding_provider = MockEmbeddingProvider(dimension=dimension)
            else:
                self.embedding_provider = MockEmbeddingProvider()
        else:
            self.embedding_provider = embedding_provider
        effective_dimension = dimension or getattr(self.embedding_provider, "dimension", 128)
        self.vector_store = vector_store or create_vector_store(
            dimension=effective_dimension
        )

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text by stripping hidden control characters.

        OWASP LLM01/LLM08: Removes potential indirect prompt injection vectors
        like null bytes, vertical tabs, and other non-printing control characters
        while preserving normal whitespace (newlines, tabs, spaces).
        """
        if not text:
            return ""
        # Remove control characters except \n, \r, \t
        sanitized = _CONTROL_CHAR_RE.sub("", text)
        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized.strip()

    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using recursive character splitting.

        Strategy:
        1. Try to split at paragraph boundaries (\n\n)
        2. Then at sentence boundaries (.!?)
        3. Then at word boundaries
        4. Finally hard-split at chunk_size

        Each chunk overlaps with the previous by chunk_overlap characters.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end, try to find a good break point
            if end < text_len:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind(". ", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("? ", start, end),
                    )
                    if sentence_break > start:
                        end = sentence_break + 2
                    else:
                        # Look for word break
                        word_break = text.rfind(" ", start, end)
                        if word_break > start:
                            end = word_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward with overlap
            if end >= text_len:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def index_document(
        self,
        doc_id: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,  # Backward compatibility alias
    ) -> int:
        """
        Index a document by chunking, embedding, and storing.

        Args:
            doc_id: Unique document identifier.
            content: Document text content (primary parameter).
            metadata: Optional metadata dict to attach to all chunks.
            text: Backward compatibility alias for content.

        Returns:
            Number of chunks successfully indexed (0 on failure).
        """
        # Backward compatibility: allow `text` as alias for `content`
        if text is not None:
            content = text

        if not doc_id or not content:
            logger.warning("index_document called with empty doc_id or content")
            return 0

        try:
            # Sanitize content
            clean_content = self._sanitize_text(content)
            if not clean_content:
                logger.warning("Document %s empty after sanitization", doc_id)
                return 0

            # Chunk the content
            text_chunks = self._recursive_chunk(clean_content)
            if not text_chunks:
                logger.warning("No chunks generated for document %s", doc_id)
                return 0

            # Generate embeddings
            embeddings = self.embedding_provider.embed_batch(text_chunks)
            if not embeddings or len(embeddings) != len(text_chunks):
                logger.error("Embedding generation failed for document %s", doc_id)
                return 0

            # Build chunk records
            chunk_records = []
            base_metadata = metadata or {}
            for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
                chunk_records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                    embedding=np.asarray(embedding, dtype=np.float32),
                ))

            # Store in vector store
            added = self.vector_store.add_chunks(chunk_records)
            logger.info("Indexed document %s: %d/%d chunks", doc_id, added, len(chunk_records))
            return added

        except Exception as exc:  # noqa: BLE001 - failure containment
            logger.error("Failed to index document %s: %s", doc_id, exc)
            return 0

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant context chunks for a query.

        Args:
            query: Search query text.
            top_k: Maximum number of chunks to return.
            min_similarity: Override default minimum similarity threshold.

        Returns:
            List of RetrievedChunk sorted by similarity (descending).
            Returns empty list on any failure (no exceptions propagated).
        """
        if not query:
            return []

        try:
            # Sanitize query
            clean_query = self._sanitize_text(query)
            if not clean_query:
                return []

            # Embed query
            query_embedding = self.embedding_provider.embed_text(clean_query)
            if not query_embedding:
                logger.warning("Query embedding failed for: %s", query[:50])
                return []

            # Search vector store
            threshold = min_similarity if min_similarity is not None else self.min_similarity
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=threshold,
            )

            logger.debug("Retrieved %d chunks for query: %s", len(results), query[:50])
            return results

        except Exception as exc:  # noqa: BLE001 - failure containment
            logger.error("Retrieve context failed for query '%s': %s", query[:50], exc)
            return []

    # Backward compatibility method for old SimpleRAGEngine API
    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility with old SimpleRAGEngine.

        Returns list of dicts with chunk_id, doc_id, text fields.
        """
        chunks = self.retrieve_context(query, top_k=top_k)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "similarity": chunk.similarity,
            }
            for chunk in chunks
        ]
        """
        Retrieve relevant context chunks for a query.

        Args:
            query: Search query text.
            top_k: Maximum number of chunks to return.
            min_similarity: Override default minimum similarity threshold.

        Returns:
            List of RetrievedChunk sorted by similarity (descending).
            Returns empty list on any failure (no exceptions propagated).
        """
        if not query:
            return []

        try:
            # Sanitize query
            clean_query = self._sanitize_text(query)
            if not clean_query:
                return []

            # Embed query
            query_embedding = self.embedding_provider.embed_text(clean_query)
            if not query_embedding:
                logger.warning("Query embedding failed for: %s", query[:50])
                return []

            # Search vector store
            threshold = min_similarity if min_similarity is not None else self.min_similarity
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=threshold,
            )

            logger.debug("Retrieved %d chunks for query: %s", len(results), query[:50])
            return results

        except Exception as exc:  # noqa: BLE001 - failure containment
            logger.error("Retrieve context failed for query '%s': %s", query[:50], exc)
            return []

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns number deleted."""
        try:
            return self.vector_store.delete_document(doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to delete document %s: %s", doc_id, exc)
            return 0

    def clear(self) -> None:
        """Clear all indexed documents."""
        try:
            self.vector_store.clear()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to clear vector store: %s", exc)

    def stats(self) -> Dict[str, Any]:
        """Return engine and store statistics."""
        store_stats = self.vector_store.stats()
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_similarity": self.min_similarity,
            "embedding_provider": type(self.embedding_provider).__name__,
            **store_stats,
        }


# Backward compatibility alias
SimpleRAGEngine = RAGEngine


def create_rag_engine(
    provider_type: str = "auto",
    store_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    dimension: Optional[int] = None,
    **provider_kwargs: Any,
) -> RAGEngine:
    """
    Factory function to create a RAGEngine with default wiring.

    Args:
        provider_type: "ollama" | "mock" | "auto" for embedding provider.
        store_dir: Optional custom vector store directory.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.
        min_similarity: Minimum cosine similarity for retrieval.
        dimension: Explicit embedding dimension (overrides provider default).
        **provider_kwargs: Passed to embedding provider constructor.

    Returns:
        Configured RAGEngine instance.
    """
    if dimension is not None:
        provider_kwargs.setdefault("dimension", dimension)
    embedding_provider = create_embedding_provider(provider_type, **provider_kwargs)
    effective_dimension = getattr(embedding_provider, "dimension", 128)
    vector_store = create_vector_store(
        store_dir=Path(store_dir) if store_dir else None,
        dimension=effective_dimension,
    )
    return RAGEngine(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_similarity=min_similarity,
        dimension=effective_dimension,
    )


__all__ = [
    "RAGEngine",
    "SimpleRAGEngine",  # Backward compatibility
    "create_rag_engine",
    "RetrievedChunk",
]