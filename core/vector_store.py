"""
Persistent local vector store for OptiRND (Phase 3, Prompt 2).

Implements an embedded vector store backed by SQLite for metadata and
NumPy .npz for embeddings. Provides fast cosine similarity search over
normalized vectors using vectorized NumPy operations.

OWASP LLM08 compliance: persistent storage with no external dependencies.
Auto-creates storage directory. All operations are failure-contained.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("optirnd.vector_store")

DEFAULT_STORE_DIR = Path("data/vector_store")
DEFAULT_MIN_SIMILARITY = 0.60


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float


@dataclass
class ChunkRecord:
    """Internal representation of a stored chunk."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray


class LocalVectorStore:
    """
    Embedded persistent vector store using SQLite + NumPy.

    Storage layout:
        data/vector_store/
        ├── metadata.db       # SQLite: chunk metadata, doc_id, text, etc.
        └── embeddings.npz    # NumPy: matrix of embeddings (chunks x dim)

    All vectors are stored L2-normalized for fast cosine similarity via dot product.
    """

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        dimension: int = 128,
    ):
        self.store_dir = Path(store_dir or DEFAULT_STORE_DIR)
        self.dimension = dimension

        # Ensure directory exists
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.store_dir / "metadata.db"
        self._npz_path = self.store_dir / "embeddings.npz"

        self._init_db()
        self._load_embeddings()

    def _init_db(self) -> None:
        """Initialize SQLite schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)
            """)
            conn.commit()

    def _load_embeddings(self) -> None:
        """Load embeddings matrix from .npz file."""
        if self._npz_path.exists():
            try:
                data = np.load(self._npz_path, allow_pickle=True)
                self._embeddings = data["embeddings"]
                self._chunk_ids = data["chunk_ids"].tolist()
                logger.debug("Loaded %d embeddings from %s", len(self._chunk_ids), self._npz_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load embeddings, starting fresh: %s", exc)
                self._embeddings = np.zeros((0, self.dimension), dtype=np.float32)
                self._chunk_ids = []
        else:
            self._embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            self._chunk_ids = []

    def _save_embeddings(self) -> None:
        """Persist embeddings matrix to .npz file."""
        try:
            np.savez_compressed(
                self._npz_path,
                embeddings=self._embeddings.astype(np.float32),
                chunk_ids=np.array(self._chunk_ids, dtype=object),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save embeddings: %s", exc)
            raise

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector (in-place for 1D, row-wise for 2D)."""
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vec / norms

    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        """
        Add multiple chunks to the store.

        Args:
            chunks: List of ChunkRecord objects with pre-computed embeddings.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        added = 0
        new_embeddings = []
        new_chunk_ids = []

        with sqlite3.connect(self._db_path) as conn:
            for chunk in chunks:
                # Validate embedding dimension
                if chunk.embedding.shape != (self.dimension,):
                    logger.warning(
                        "Skipping chunk %s: embedding dimension %s != %d",
                        chunk.chunk_id,
                        chunk.embedding.shape,
                        self.dimension,
                    )
                    continue

                # Normalize embedding
                normalized = self._normalize(chunk.embedding.astype(np.float32))

                # Store in SQLite
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.text,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                        ),
                    )
                except sqlite3.Error as exc:
                    logger.warning("Failed to insert chunk %s: %s", chunk.chunk_id, exc)
                    continue

                new_embeddings.append(normalized)
                new_chunk_ids.append(chunk.chunk_id)
                added += 1

            conn.commit()

        if new_embeddings:
            # Append to embeddings matrix
            new_matrix = np.vstack(new_embeddings)
            if self._embeddings.size == 0:
                self._embeddings = new_matrix
            else:
                self._embeddings = np.vstack([self._embeddings, new_matrix])
            self._chunk_ids.extend(new_chunk_ids)
            self._save_embeddings()

        return added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[RetrievedChunk]:
        """
        Cosine similarity search over stored embeddings.

        Args:
            query_embedding: Query vector (will be normalized).
            top_k: Maximum number of results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of RetrievedChunk sorted by similarity (descending).
        """
        if self._embeddings.size == 0 or not query_embedding:
            return []

        # Normalize query
        query_vec = self._normalize(np.asarray(query_embedding, dtype=np.float32))

        # Vectorized cosine similarity via dot product (both normalized)
        similarities = self._embeddings @ query_vec

        # Filter by threshold
        mask = similarities >= min_similarity
        if not np.any(mask):
            return []

        # Get top-k indices
        candidate_indices = np.where(mask)[0]
        candidate_sims = similarities[mask]

        # Sort by similarity descending
        sorted_idx = np.argsort(candidate_sims)[::-1]
        top_indices = candidate_indices[sorted_idx[:top_k]]
        top_sims = candidate_sims[sorted_idx[:top_k]]

        # Retrieve metadata from SQLite
        results = []
        if len(top_indices) > 0:
            placeholders = ",".join("?" * len(top_indices))
            chunk_ids = [self._chunk_ids[i] for i in top_indices]

            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    f"SELECT chunk_id, doc_id, text, metadata FROM chunks WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
                rows = {row["chunk_id"]: row for row in cursor.fetchall()}

            for idx, sim in zip(top_indices, top_sims):
                chunk_id = self._chunk_ids[idx]
                row = rows.get(chunk_id)
                if row:
                    results.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        doc_id=row["doc_id"],
                        text=row["text"],
                        metadata=json.loads(row["metadata"]),
                        similarity=float(sim),
                    ))

        return results

    def get_chunks_by_doc_id(self, doc_id: str) -> List[RetrievedChunk]:
        """Retrieve all chunks for a given document ID."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT chunk_id, doc_id, text, metadata FROM chunks WHERE doc_id = ?",
                (doc_id,),
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            # Find embedding index
            try:
                idx = self._chunk_ids.index(row["chunk_id"])
                emb = self._embeddings[idx]
                # Compute self-similarity (always 1.0 for normalized vectors)
                results.append(RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]),
                    similarity=1.0,
                ))
            except ValueError:
                # Chunk in DB but not in embeddings matrix (shouldn't happen)
                pass

        return results

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns number deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            if not chunk_ids:
                return 0

            # Delete from SQLite
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            conn.commit()

        # Rebuild embeddings matrix (filter out deleted)
        if chunk_ids:
            keep_mask = np.array([cid not in chunk_ids for cid in self._chunk_ids])
            self._embeddings = self._embeddings[keep_mask]
            self._chunk_ids = [cid for cid, keep in zip(self._chunk_ids, keep_mask) if keep]
            self._save_embeddings()

        return len(chunk_ids)

    def clear(self) -> None:
        """Clear all data from the store."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM chunks")
            conn.commit()

        self._embeddings = np.zeros((0, self.dimension), dtype=np.float32)
        self._chunk_ids = []
        self._save_embeddings()

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            count = cursor.fetchone()[0]

        return {
            "total_chunks": count,
            "dimension": self.dimension,
            "store_dir": str(self.store_dir),
            "embeddings_shape": self._embeddings.shape,
        }

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def __bool__(self) -> bool:
        # Always truthy, even when empty (avoids `vector_store or default` fallback)
        return True


def create_vector_store(
    store_dir: Optional[Path] = None,
    dimension: int = 128,
) -> LocalVectorStore:
    """Factory function to create a LocalVectorStore."""
    return LocalVectorStore(store_dir=store_dir, dimension=dimension)


__all__ = [
    "RetrievedChunk",
    "ChunkRecord",
    "LocalVectorStore",
    "create_vector_store",
]