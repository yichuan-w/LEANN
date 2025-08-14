"""
Basic functionality tests for CI pipeline using pytest.
"""

import os
import tempfile
from pathlib import Path

import pytest


def test_imports():
    """Test that all packages can be imported."""

    # Test C++ extensions


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_backend_basic(backend_name):
    """Test basic functionality for each backend."""
    from leann.api import LeannBuilder, LeannSearcher, SearchResult

    # Create temporary directory for index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / f"test.{backend_name}")

        # Test with small data
        texts = [f"This is document {i} about topic {i % 5}" for i in range(100)]

        # Configure builder based on backend
        if backend_name == "hnsw":
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="facebook/contriever",
                embedding_mode="sentence-transformers",
                M=16,
                efConstruction=200,
            )
        else:  # diskann
            builder = LeannBuilder(
                backend_name="diskann",
                embedding_model="facebook/contriever",
                embedding_mode="sentence-transformers",
                num_neighbors=32,
                search_list_size=50,
            )

        # Add texts
        for text in texts:
            builder.add_text(text)

        # Build index
        builder.build_index(index_path)

        # Test search
        searcher = LeannSearcher(index_path)
        results = searcher.search("document about topic 2", top_k=5)

        # Verify results
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert "topic 2" in results[0].text or "document" in results[0].text

        # Ensure cleanup to avoid hanging background servers
        searcher.cleanup()


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_large_index():
    """Test with larger dataset."""
    from leann.api import LeannBuilder, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / "test_large.hnsw")
        texts = [f"Document {i}: {' '.join([f'word{j}' for j in range(50)])}" for i in range(1000)]

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)

        searcher = LeannSearcher(index_path)
        results = searcher.search(["word10 word20"], top_k=10)
        assert len(results[0]) == 10
        # Cleanup
        searcher.cleanup()
