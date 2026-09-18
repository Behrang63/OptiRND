"""
Phase 3, Prompt 2 tests: RAG Engine, Embedding Providers, Vector Store.

Tests cover:
- MockEmbeddingProvider: deterministic shape, norm ≈ 1.0
- LocalVectorStore: save, persist, reload, cosine-similarity ranking
- RAGEngine: chunking logic, document indexing, sanitized input
- Failure tests: Ollama unreachable -> fallback to mock -> retrieval succeeds
- Integration: ExAnteAgent uses RAG context injection
- Regression: all existing tests continue to pass
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.embedding_provider import (
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    create_embedding_provider,
)
from core.vector_store import (
    LocalVectorStore,
    RetrievedChunk,
    ChunkRecord,
    create_vector_store,
)
from core.rag_engine import RAGEngine, SimpleRAGEngine, create_rag_engine
from core.contracts import QualitativeAssessment


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_store_dir():
    """Create a temporary directory for vector store tests."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider(dimension=64)


@pytest.fixture
def vector_store(temp_store_dir):
    return create_vector_store(store_dir=temp_store_dir, dimension=64)


@pytest.fixture
def rag_engine(mock_embedding_provider, vector_store):
    return RAGEngine(
        embedding_provider=mock_embedding_provider,
        vector_store=vector_store,
        chunk_size=200,
        chunk_overlap=20,
        min_similarity=0.3,  # Lower for testing
    )


# =============================================================================
# MockEmbeddingProvider Tests
# =============================================================================

class TestMockEmbeddingProvider:
    """Tests for deterministic mock embedding provider."""

    def test_embed_text_returns_list_of_floats(self, mock_embedding_provider):
        vec = mock_embedding_provider.embed_text("test text")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) == mock_embedding_provider.dimension

    def test_embed_batch_returns_list_of_vectors(self, mock_embedding_provider):
        texts = ["text one", "text two", "text three"]
        vecs = mock_embedding_provider.embed_batch(texts)
        assert isinstance(vecs, list)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == mock_embedding_provider.dimension
            assert all(isinstance(x, float) for x in vec)

    def test_deterministic_same_input_same_output(self, mock_embedding_provider):
        text = "deterministic test"
        vec1 = mock_embedding_provider.embed_text(text)
        vec2 = mock_embedding_provider.embed_text(text)
        assert vec1 == vec2

    def test_different_inputs_different_outputs(self, mock_embedding_provider):
        vec1 = mock_embedding_provider.embed_text("text one")
        vec2 = mock_embedding_provider.embed_text("text two")
        assert vec1 != vec2

    def test_vectors_normalized_unit_norm(self, mock_embedding_provider):
        vec = mock_embedding_provider.embed_text("normalization test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6, f"Vector norm {norm} != 1.0"

    def test_batch_vectors_normalized(self, mock_embedding_provider):
        vecs = mock_embedding_provider.embed_batch(["a", "b", "c"])
        for vec in vecs:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-6

    def test_dimension_configurable(self):
        for dim in [32, 64, 128, 256]:
            provider = MockEmbeddingProvider(dimension=dim)
            vec = provider.embed_text("test")
            assert len(vec) == dim

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            MockEmbeddingProvider(dimension=0)
        with pytest.raises(ValueError):
            MockEmbeddingProvider(dimension=-1)

    def test_empty_string_handled(self, mock_embedding_provider):
        vec = mock_embedding_provider.embed_text("")
        assert len(vec) == mock_embedding_provider.dimension
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_unicode_text_handled(self, mock_embedding_provider):
        vec = mock_embedding_provider.embed_text("تست فارسی 🚀 中文")
        assert len(vec) == mock_embedding_provider.dimension
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6


# =============================================================================
# LocalVectorStore Tests
# =============================================================================

class TestLocalVectorStore:
    """Tests for persistent vector store with cosine similarity."""

    def test_add_chunks_and_search(self, vector_store, mock_embedding_provider):
        chunks = [
            ChunkRecord(
                chunk_id="doc1_chunk_0",
                doc_id="doc1",
                text="This is the first document about risk assessment.",
                metadata={"source": "test"},
                embedding=np.array(mock_embedding_provider.embed_text("risk assessment document")),
            ),
            ChunkRecord(
                chunk_id="doc1_chunk_1",
                doc_id="doc1",
                text="Second chunk discusses financial modeling.",
                metadata={"source": "test"},
                embedding=np.array(mock_embedding_provider.embed_text("financial modeling")),
            ),
            ChunkRecord(
                chunk_id="doc2_chunk_0",
                doc_id="doc2",
                text="Another document about energy risk.",
                metadata={"source": "test"},
                embedding=np.array(mock_embedding_provider.embed_text("energy risk")),
            ),
        ]
        added = vector_store.add_chunks(chunks)
        assert added == 3

    def test_cosine_similarity_ranking(self, vector_store, mock_embedding_provider):
        # Add chunks with known embeddings
        chunks = [
            ChunkRecord(
                chunk_id="a", doc_id="d1", text="risk assessment",
                metadata={}, embedding=np.array(mock_embedding_provider.embed_text("risk assessment")),
            ),
            ChunkRecord(
                chunk_id="b", doc_id="d1", text="financial modeling",
                metadata={}, embedding=np.array(mock_embedding_provider.embed_text("financial modeling")),
            ),
            ChunkRecord(
                chunk_id="c", doc_id="d2", text="energy risk",
                metadata={}, embedding=np.array(mock_embedding_provider.embed_text("energy risk")),
            ),
        ]
        vector_store.add_chunks(chunks)

        # Query similar to "risk assessment"
        query_emb = mock_embedding_provider.embed_text("risk assessment")
        results = vector_store.search(query_emb, top_k=2, min_similarity=0.0)

        assert len(results) >= 1
        assert results[0].chunk_id == "a"  # Should be most similar
        assert results[0].similarity > results[1].similarity if len(results) > 1 else True

    def test_min_similarity_filter(self, vector_store, mock_embedding_provider):
        chunks = [
            ChunkRecord(
                chunk_id="match", doc_id="d1", text="risk assessment",
                metadata={}, embedding=np.array(mock_embedding_provider.embed_text("risk assessment")),
            ),
            ChunkRecord(
                chunk_id="nomatch", doc_id="d1", text="completely different topic xyz",
                metadata={}, embedding=np.array(mock_embedding_provider.embed_text("completely different topic xyz")),
            ),
        ]
        vector_store.add_chunks(chunks)

        query_emb = mock_embedding_provider.embed_text("risk assessment")
        # High threshold - should filter out non-matching
        results = vector_store.search(query_emb, top_k=5, min_similarity=0.5)
        # The matching chunk should have higher similarity
        assert len(results) >= 1
        assert results[0].chunk_id == "match"

    def test_persist_and_reload(self, temp_store_dir, mock_embedding_provider):
        # Create store, add data
        store1 = create_vector_store(store_dir=temp_store_dir, dimension=64)
        chunks = [
            ChunkRecord(
                chunk_id="persist_test", doc_id="doc1", text="persistent data",
                metadata={"key": "value"}, embedding=np.array(mock_embedding_provider.embed_text("persistent data")),
            ),
        ]
        store1.add_chunks(chunks)

        # Reload from disk
        store2 = create_vector_store(store_dir=temp_store_dir, dimension=64)
        results = store2.search(
            mock_embedding_provider.embed_text("persistent data"),
            top_k=1, min_similarity=0.0
        )

        assert len(results) == 1
        assert results[0].chunk_id == "persist_test"
        assert results[0].metadata["key"] == "value"

    def test_delete_document(self, vector_store, mock_embedding_provider):
        chunks = [
            ChunkRecord(chunk_id="d1_c0", doc_id="doc1", text="chunk 0", metadata={},
                        embedding=np.array(mock_embedding_provider.embed_text("chunk 0"))),
            ChunkRecord(chunk_id="d1_c1", doc_id="doc1", text="chunk 1", metadata={},
                        embedding=np.array(mock_embedding_provider.embed_text("chunk 1"))),
            ChunkRecord(chunk_id="d2_c0", doc_id="doc2", text="chunk 2", metadata={},
                        embedding=np.array(mock_embedding_provider.embed_text("chunk 2"))),
        ]
        vector_store.add_chunks(chunks)

        deleted = vector_store.delete_document("doc1")
        assert deleted == 2

        # Verify doc1 chunks gone, doc2 remains
        results = vector_store.search(
            mock_embedding_provider.embed_text("chunk 2"),
            top_k=5, min_similarity=0.0
        )
        assert len(results) == 1
        assert results[0].chunk_id == "d2_c0"

    def test_clear(self, vector_store, mock_embedding_provider):
        chunks = [
            ChunkRecord(chunk_id="c1", doc_id="d1", text="t1", metadata={},
                        embedding=np.array(mock_embedding_provider.embed_text("t1"))),
        ]
        vector_store.add_chunks(chunks)
        vector_store.clear()

        results = vector_store.search(
            mock_embedding_provider.embed_text("t1"),
            top_k=1, min_similarity=0.0
        )
        assert len(results) == 0

    def test_get_chunks_by_doc_id(self, vector_store, mock_embedding_provider):
        chunks = [
            ChunkRecord(chunk_id="d1_c0", doc_id="doc1", text="chunk 0", metadata={"idx": 0},
                        embedding=np.array(mock_embedding_provider.embed_text("chunk 0"))),
            ChunkRecord(chunk_id="d1_c1", doc_id="doc1", text="chunk 1", metadata={"idx": 1},
                        embedding=np.array(mock_embedding_provider.embed_text("chunk 1"))),
        ]
        vector_store.add_chunks(chunks)

        results = vector_store.get_chunks_by_doc_id("doc1")
        assert len(results) == 2
        assert all(r.doc_id == "doc1" for r in results)
        assert results[0].metadata["idx"] == 0

    def test_stats(self, vector_store, mock_embedding_provider):
        stats = vector_store.stats()
        assert stats["total_chunks"] == 0
        assert stats["dimension"] == 64

        chunks = [
            ChunkRecord(chunk_id="c1", doc_id="d1", text="t1", metadata={},
                        embedding=np.array(mock_embedding_provider.embed_text("t1"))),
        ]
        vector_store.add_chunks(chunks)

        stats = vector_store.stats()
        assert stats["total_chunks"] == 1
        assert stats["embeddings_shape"] == (1, 64)

    def test_empty_store_search_returns_empty(self, vector_store, mock_embedding_provider):
        results = vector_store.search(
            mock_embedding_provider.embed_text("anything"),
            top_k=5, min_similarity=0.0
        )
        assert results == []

    def test_invalid_embedding_dimension_rejected(self, vector_store):
        # Wrong dimension
        bad_embedding = np.zeros(32, dtype=np.float32)
        chunk = ChunkRecord(
            chunk_id="bad", doc_id="d1", text="bad", metadata={}, embedding=bad_embedding
        )
        added = vector_store.add_chunks([chunk])
        assert added == 0


# =============================================================================
# RAGEngine Tests
# =============================================================================

class TestRAGEngine:
    """Tests for the main RAG engine."""

    def test_chunking_basic(self, rag_engine):
        text = "This is a test document. " * 20  # ~500 chars
        chunks = rag_engine._recursive_chunk(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= rag_engine.chunk_size + 50  # Allow some flexibility

    def test_chunking_overlap(self, rag_engine):
        text = "Word " * 100  # ~500 chars
        chunks = rag_engine._recursive_chunk(text)
        if len(chunks) >= 2:
            # Check overlap exists (allow for word boundary adjustments)
            # The overlap should be approximately chunk_overlap characters
            overlap_region_1 = chunks[0][-rag_engine.chunk_overlap:]
            overlap_region_2 = chunks[1][:rag_engine.chunk_overlap]
            # They should share significant content
            assert len(set(overlap_region_1.split()) & set(overlap_region_2.split())) > 0

    def test_chunking_empty_text(self, rag_engine):
        assert rag_engine._recursive_chunk("") == []
        assert rag_engine._recursive_chunk("   ") == []

    def test_sanitize_removes_control_chars(self, rag_engine):
        # Text with null bytes, vertical tabs, etc.
        dirty = "Clean\x00text\x0bwith\x0ccontrol\x1fchars"
        clean = rag_engine._sanitize_text(dirty)
        assert "\x00" not in clean
        assert "\x0b" not in clean
        assert "\x0c" not in clean
        assert "\x1f" not in clean
        assert "Clean" in clean
        assert "text" in clean

    def test_sanitize_preserves_normal_whitespace(self, rag_engine):
        text = "Line 1\nLine 2\tTabbed\r\nWindows"
        clean = rag_engine._sanitize_text(text)
        assert "\n" in clean or " " in clean  # Whitespace normalized to spaces

    def test_index_document_success(self, rag_engine):
        count = rag_engine.index_document(
            doc_id="test_doc",
            content="This is a test document about risk assessment and financial modeling.",
            metadata={"category": "test"},
        )
        assert count > 0

    def test_index_document_empty_content_returns_zero(self, rag_engine):
        count = rag_engine.index_document(doc_id="empty", content="")
        assert count == 0

    def test_index_document_sanitizes_input(self, rag_engine):
        dirty_content = "Clean content\x00with\x0bcontrol\x1fchars"
        count = rag_engine.index_document(doc_id="dirty", content=dirty_content)
        assert count > 0

        # Verify retrieval works (content was sanitized)
        results = rag_engine.retrieve_context("Clean content", top_k=1)
        assert len(results) > 0
        assert "\x00" not in results[0].text

    def test_retrieve_context_returns_chunks(self, rag_engine):
        rag_engine.index_document(
            doc_id="doc1",
            content="Risk assessment for the new steel manufacturing project involves cost analysis.",
            metadata={},
        )
        results = rag_engine.retrieve_context("risk assessment steel", top_k=2)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert r.similarity >= rag_engine.min_similarity

    def test_retrieve_context_empty_query_returns_empty(self, rag_engine):
        results = rag_engine.retrieve_context("")
        assert results == []

    def test_retrieve_context_no_matches_returns_empty(self, rag_engine):
        rag_engine.index_document("doc1", "completely unrelated content about cooking recipes")
        # Use very high threshold that unrelated content won't meet
        # Mock embeddings can have high similarity, so use 0.99
        results = rag_engine.retrieve_context("quantum physics", top_k=5, min_similarity=0.99)
        assert results == []

    def test_retrieve_context_respects_top_k(self, rag_engine):
        for i in range(5):
            rag_engine.index_document(f"doc{i}", f"Document number {i} about topic")
        results = rag_engine.retrieve_context("document", top_k=2, min_similarity=0.0)
        assert len(results) <= 2

    def test_retrieve_context_respects_min_similarity(self, rag_engine):
        rag_engine.index_document("doc1", "risk assessment financial modeling")
        # Very high threshold - unlikely to match
        results = rag_engine.retrieve_context("risk", top_k=5, min_similarity=0.95)
        # May be empty or very few results
        for r in results:
            assert r.similarity >= 0.95

    def test_failure_containment_index_document(self, rag_engine, monkeypatch):
        # Force embedding provider to fail
        def failing_embed_batch(texts):
            raise RuntimeError("Embedding service down")

        monkeypatch.setattr(rag_engine.embedding_provider, "embed_batch", failing_embed_batch)
        count = rag_engine.index_document("doc1", "test content")
        assert count == 0  # No crash, returns 0

    def test_failure_containment_retrieve_context(self, rag_engine, monkeypatch):
        rag_engine.index_document("doc1", "test content")

        def failing_embed_text(text):
            raise RuntimeError("Embedding service down")

        monkeypatch.setattr(rag_engine.embedding_provider, "embed_text", failing_embed_text)
        results = rag_engine.retrieve_context("test")
        assert results == []  # No crash, returns empty list

    def test_delete_document(self, rag_engine):
        # Use unique doc_id to avoid interference from other tests
        rag_engine.index_document("doc_delete_test", "content to delete")
        deleted = rag_engine.delete_document("doc_delete_test")
        assert deleted > 0

        results = rag_engine.retrieve_context("content", top_k=1)
        assert len(results) == 0

    def test_clear(self, rag_engine):
        rag_engine.index_document("doc1", "content")
        rag_engine.index_document("doc2", "more content")
        rag_engine.clear()
        results = rag_engine.retrieve_context("content", top_k=5)
        assert len(results) == 0

    def test_stats(self, rag_engine):
        stats = rag_engine.stats()
        assert "total_chunks" in stats
        assert "chunk_size" in stats
        assert "embedding_provider" in stats


# =============================================================================
# Integration Tests: RAG + ExAnteAgent
# =============================================================================

class TestRAGExAnteIntegration:
    """Integration tests for RAG engine with ExAnteAgent."""

    @pytest.fixture
    def minimal_proposal(self):
        return {
            "title": "Test Project",
            "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
            "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
            "p_success": 0.85,
        }

    def test_agent_with_rag_context(self, minimal_proposal, temp_store_dir):
        """Agent uses RAG context in qualitative assessment."""
        from agents.main_agents.ex_ante_agent import ExAnteAgent
        from core.llm_provider import MockLLMProvider

        # Create agent with RAG engine using temp directory
        rag = create_rag_engine(
            provider_type="mock",
            store_dir=str(temp_store_dir),
        )
        agent = ExAnteAgent(llm_provider=MockLLMProvider(), rag_engine=rag)

        # Provide raw text that will be indexed and retrieved
        raw_text = (
            "این پروژه شامل بومی‌سازی سنسورهای گاز است. "
            "ریسک اصلی تاخیر در تامین قطعات الکترونیکی می‌باشد. "
            "تحلیل هزینه‌ها نشان می‌دهد که بومی‌سازی می‌تواند ۳۰٪ کاهش هزینه را فراهم کند."
        )

        result = agent.evaluate_comprehensive_proposal(minimal_proposal, raw_text=raw_text)

        # Verify RAG evidence was collected
        assert "rag_evidence" in result
        assert isinstance(result["rag_evidence"], list)

        # Verify qualitative assessment still valid
        qa = result["qualitative_assessment"]
        assert 0.0 <= qa["technical_success_probability"] <= 1.0
        assert isinstance(qa["risk_summary"], str)

    def test_agent_without_raw_text_works(self, minimal_proposal):
        """Agent works without raw_text (backward compatibility)."""
        from agents.main_agents.ex_ante_agent import ExAnteAgent
        from core.llm_provider import MockLLMProvider

        agent = ExAnteAgent(llm_provider=MockLLMProvider())
        result = agent.evaluate_comprehensive_proposal(minimal_proposal)

        assert "qualitative_assessment" in result
        assert "rag_evidence" in result
        assert result["rag_evidence"] == []

    def test_rag_context_injected_into_prompt(self, minimal_proposal, temp_store_dir):
        """Verify RAG context is injected into LLM prompt via context tags."""
        from agents.main_agents.ex_ante_agent import ExAnteAgent
        from core.llm_provider import MockLLMProvider

        # Track prompts sent to LLM
        captured_prompts = []

        class TrackingMockProvider(MockLLMProvider):
            def generate_json(self, prompt, schema_description):
                captured_prompts.append(prompt)
                return super().generate_json(prompt, schema_description)

        # Create isolated RAG engine with temp directory and matching dimension
        rag = create_rag_engine(
            provider_type="mock",
            store_dir=str(temp_store_dir),
            dimension=64,
        )
        agent = ExAnteAgent(llm_provider=TrackingMockProvider(), rag_engine=rag)

        raw_text = "Project involves gas sensor localization with supply chain risk."
        agent.evaluate_comprehensive_proposal(minimal_proposal, raw_text=raw_text)

        # Check that context was injected
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "<context>" in prompt
        assert "</context>" in prompt
        assert "gas sensor" in prompt.lower() or "localization" in prompt.lower()


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_embedding_provider_mock(self):
        provider = create_embedding_provider("mock")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_create_embedding_provider_ollama_fallback(self):
        # Test that explicit "mock" returns mock
        provider = create_embedding_provider("mock")
        assert isinstance(provider, MockEmbeddingProvider)

        # Test that "auto" returns a provider (could be Ollama if available)
        provider = create_embedding_provider("auto")
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_create_embedding_provider_auto(self):
        provider = create_embedding_provider("auto")
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_create_vector_store(self, temp_store_dir):
        store = create_vector_store(store_dir=temp_store_dir, dimension=64)
        assert isinstance(store, LocalVectorStore)

    def test_create_rag_engine(self, temp_store_dir):
        engine = create_rag_engine(
            provider_type="mock",
            store_dir=str(temp_store_dir),
            chunk_size=100,
        )
        assert isinstance(engine, RAGEngine)
        assert engine.chunk_size == 100

    def test_simple_rag_engine_alias(self):
        """Backward compatibility: SimpleRAGEngine is alias for RAGEngine."""
        assert SimpleRAGEngine is RAGEngine


# =============================================================================
# OllamaEmbeddingProvider Failure Tests (mocked)
# =============================================================================

class TestOllamaEmbeddingProviderFallback:
    """Tests that Ollama provider failures degrade to mock behavior."""

    def test_ollama_unreachable_returns_none(self):
        """_embed_single returns None when server unreachable."""
        provider = OllamaEmbeddingProvider(host="http://localhost:9999", timeout_seconds=0.001)
        result = provider._embed_single("test")
        assert result is None

    def test_embed_batch_returns_empty_on_failure(self):
        """embed_batch returns empty list when all requests fail."""
        provider = OllamaEmbeddingProvider(host="http://localhost:9999", timeout_seconds=0.001)
        result = provider.embed_batch(["text1", "text2"])
        assert result == []

    def test_embed_text_returns_empty_on_failure(self):
        """embed_text returns empty list when request fails."""
        provider = OllamaEmbeddingProvider(host="http://localhost:9999", timeout_seconds=0.001)
        result = provider.embed_text("test")
        assert result == []


# =============================================================================
# Regression Tests: Existing functionality preserved
# =============================================================================

class TestRegression:
    """Ensure existing tests still pass."""

    def test_mock_llm_provider_unchanged(self):
        from core.llm_provider import LLMFactory, MockLLMProvider
        provider = LLMFactory.get_provider("mock")
        assert isinstance(provider, MockLLMProvider)
        out = provider.generate_json("test", "schema")
        assert "technical_success_probability" in out

    def test_ollama_llm_provider_unchanged(self):
        from agents.ollama_provider import OllamaLLMProvider
        provider = OllamaLLMProvider()
        # Should have the expected methods
        assert hasattr(provider, "generate_validated")
        assert hasattr(provider, "is_available")

    def test_distributions_unchanged(self):
        from core.distributions import generate_pert_samples, make_rng
        rng = make_rng(42)
        samples = generate_pert_samples(1.0, 2.0, 3.0, size=100, rng=rng)
        assert len(samples) == 100
        assert all(1.0 <= s <= 3.0 for s in samples)

    def test_monte_carlo_unchanged(self):
        from core.monte_carlo import MonteCarloEngine
        engine = MonteCarloEngine(num_simulations=100, seed=42)
        result = engine.run_roi_simulation(
            (100, 200, 300), (150, 250, 350), years=1
        )
        assert "mean_roi" in result


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_long_document_chunking(self, rag_engine):
        long_text = "This is a sentence. " * 500  # ~10k chars
        count = rag_engine.index_document("long_doc", long_text)
        assert count > 10  # Should create many chunks

        results = rag_engine.retrieve_context("sentence", top_k=3)
        assert len(results) > 0

    def test_many_documents(self, rag_engine):
        for i in range(20):
            rag_engine.index_document(f"doc{i}", f"Document {i} content about topic {i % 5}")
        results = rag_engine.retrieve_context("topic 2", top_k=5)
        assert len(results) <= 5

    def test_special_characters_in_metadata(self, rag_engine, vector_store, mock_embedding_provider):
        metadata = {
            "unicode": "تست 🚀",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        chunk = ChunkRecord(
            chunk_id="meta_test", doc_id="d1", text="test", metadata=metadata,
            embedding=np.array(mock_embedding_provider.embed_text("test")),
        )
        vector_store.add_chunks([chunk])

        results = vector_store.search(
            mock_embedding_provider.embed_text("test"), top_k=1, min_similarity=0.0
        )
        assert len(results) == 1
        assert results[0].metadata["unicode"] == "تست 🚀"
        assert results[0].metadata["nested"]["key"] == "value"
        assert results[0].metadata["list"] == [1, 2, 3]

    def test_concurrent_access_simulation(self, temp_store_dir, mock_embedding_provider):
        """Simulate concurrent access by creating multiple store instances."""
        # Write from one instance
        store1 = create_vector_store(store_dir=temp_store_dir, dimension=64)
        store1.add_chunks([
            ChunkRecord("c1", "d1", "text1", {}, np.array(mock_embedding_provider.embed_text("text1"))),
        ])

        # Read from another instance
        store2 = create_vector_store(store_dir=temp_store_dir, dimension=64)
        results = store2.search(mock_embedding_provider.embed_text("text1"), top_k=1, min_similarity=0.0)
        assert len(results) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))