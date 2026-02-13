"""
Comprehensive tests for hybrid search functionality.

This module tests the hybrid search feature that combines vector search
with BM25 keyword search using the gemma parameter.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
class TestHybridSearch:
    """Test suite for hybrid search functionality."""

    @pytest.fixture
    def sample_index(self):
        """Create a sample index for testing."""
        from leann.api import LeannBuilder, LeannSearcher

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = str(Path(temp_dir) / "test_hybrid.hnsw")

            # Create documents with diverse content for testing
            # Some documents are keyword-rich, others are semantically similar
            texts = [
                "The quick brown fox jumps over the lazy dog",
                "A fast auburn canine leaps above a sleepy hound",  # Semantically similar to first
                "Python programming language is great for data science",
                "Machine learning and artificial intelligence are transforming technology",
                "The weather today is sunny and warm",
                "Climate conditions are pleasant with clear skies",  # Semantically similar to weather
                "Database management systems store and retrieve data efficiently",
                "SQL queries help extract information from databases",  # Related to databases
                "Cooking recipes require precise measurements and timing",
                "Baking bread needs flour water yeast and patience",  # Related to cooking
            ]

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="facebook/contriever",
                embedding_mode="sentence-transformers",
                M=16,
                efConstruction=200,
            )

            for i, text in enumerate(texts):
                builder.add_text(text, metadata={"id": str(i), "doc_num": i})

            builder.build_index(index_path)

            searcher = LeannSearcher(index_path)
            yield searcher, texts

            # Cleanup
            searcher.cleanup()

    def test_pure_vector_search(self, sample_index):
        """Test pure vector search (gemma=1.0, default)."""
        searcher, texts = sample_index

        # Search with gemma=1.0 (pure vector search)
        results = searcher.search("canine animal", top_k=3, gemma=1.0)

        assert len(results) > 0
        assert len(results) <= 3
        # Should find semantically similar documents about animals/dogs
        assert any(
            "fox" in r.text.lower() or "dog" in r.text.lower() or "canine" in r.text.lower()
            for r in results
        )

    def test_pure_keyword_search(self, sample_index):
        """Test pure keyword search (gemma=0.0)."""
        searcher, texts = sample_index

        # Search with gemma=0.0 (pure BM25 keyword search)
        results = searcher.search("database SQL", top_k=3, gemma=0.0)

        assert len(results) > 0
        assert len(results) <= 3
        # Should find documents with exact keyword matches
        # BM25 should prioritize documents containing "database" or "SQL"
        top_result_text = results[0].text.lower()
        assert "database" in top_result_text or "sql" in top_result_text

    def test_hybrid_search_balanced(self, sample_index):
        """Test balanced hybrid search (gemma=0.5)."""
        searcher, texts = sample_index

        # Search with gemma=0.5 (balanced hybrid)
        results = searcher.search("programming Python code", top_k=5, gemma=0.5)

        assert len(results) > 0
        assert len(results) <= 5
        # Should combine both semantic and keyword matching
        # At least one result should contain "Python" or "programming"
        assert any("python" in r.text.lower() or "programming" in r.text.lower() for r in results)

    def test_hybrid_search_vector_heavy(self, sample_index):
        """Test vector-heavy hybrid search (gemma=0.8)."""
        searcher, texts = sample_index

        # Search with gemma=0.8 (mostly vector, some keyword)
        results = searcher.search("sunny weather conditions", top_k=3, gemma=0.8)

        assert len(results) > 0
        # Should prioritize semantic similarity but consider keywords
        # Should find weather-related documents
        assert any(
            "weather" in r.text.lower() or "sunny" in r.text.lower() or "climate" in r.text.lower()
            for r in results
        )

    def test_hybrid_search_keyword_heavy(self, sample_index):
        """Test keyword-heavy hybrid search (gemma=0.2)."""
        searcher, texts = sample_index

        # Search with gemma=0.2 (mostly keyword, some vector)
        results = searcher.search("bread flour baking", top_k=3, gemma=0.2)

        assert len(results) > 0
        # Should prioritize keyword matches
        # Should find documents with exact keyword matches
        top_results_text = " ".join([r.text.lower() for r in results[:2]])
        assert (
            "bread" in top_results_text
            or "flour" in top_results_text
            or "baking" in top_results_text
        )

    def test_hybrid_search_score_combination(self, sample_index):
        """Test that hybrid search properly combines scores."""
        searcher, texts = sample_index

        # Get results with different gemma values
        pure_vector = searcher.search("machine learning AI", top_k=5, gemma=1.0)
        pure_keyword = searcher.search("machine learning AI", top_k=5, gemma=0.0)
        hybrid = searcher.search("machine learning AI", top_k=5, gemma=0.5)

        # All should return results
        assert len(pure_vector) > 0
        assert len(pure_keyword) > 0
        assert len(hybrid) > 0

        # Hybrid results should potentially differ from pure approaches
        # (though with small dataset, there might be overlap)
        assert all(r.score > 0 for r in hybrid)

    def test_hybrid_search_with_metadata_filters(self, sample_index):
        """Test hybrid search combined with metadata filtering."""
        searcher, texts = sample_index

        # Search with hybrid and metadata filter
        results = searcher.search(
            "data information", top_k=5, gemma=0.6, metadata_filters={"doc_num": {"<": 8}}
        )

        assert len(results) > 0
        # All results should satisfy the metadata filter
        for r in results:
            assert r.metadata.get("doc_num", 999) < 8
