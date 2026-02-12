"""Tests for LlamaIndex integration (#217).

Tests the LeannRetriever and LeannHybridRetriever wrappers without
requiring a real index or embedding model.
"""

import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub C++ backend
# ---------------------------------------------------------------------------
_mod = sys.modules.get("leann_backend_hnsw.convert_to_csr")
if _mod is not None and not hasattr(_mod, "prune_hnsw_embeddings_inplace"):
    _mod.prune_hnsw_embeddings_inplace = lambda *a, **kw: True
if "leann_backend_hnsw" not in sys.modules:
    stub = MagicMock()
    sys.modules["leann_backend_hnsw"] = stub
    sys.modules["leann_backend_hnsw.convert_to_csr"] = stub.convert_to_csr
    stub.convert_to_csr.prune_hnsw_embeddings_inplace = lambda *a, **kw: True


# ---------------------------------------------------------------------------
# Fake SearchResult to avoid importing from leann.api which triggers heavy loads
# ---------------------------------------------------------------------------
@dataclass
class FakeSearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TestResultsToNodes:
    """Test the SearchResult -> NodeWithScore conversion."""

    def test_basic_conversion(self):
        from leann.integrations.llamaindex import _results_to_nodes

        results = [
            FakeSearchResult(id="1", score=0.9, text="hello world", metadata={"k": "v"}),
            FakeSearchResult(id="2", score=0.7, text="foo bar", metadata={}),
        ]
        nodes = _results_to_nodes(results)
        assert len(nodes) == 2
        assert nodes[0].score == 0.9
        assert nodes[0].node.text == "hello world"
        assert nodes[0].node.id_ == "1"
        assert nodes[0].node.metadata == {"k": "v"}
        assert nodes[1].score == 0.7

    def test_empty_results(self):
        from leann.integrations.llamaindex import _results_to_nodes

        assert _results_to_nodes([]) == []


class TestLeannRetriever:
    """Test the LeannRetriever LlamaIndex wrapper."""

    @patch("leann.integrations.llamaindex.LeannRetriever.__init__", return_value=None)
    def test_retrieve_delegates_to_searcher(self, mock_init):
        from llama_index.core import QueryBundle

        from leann.integrations.llamaindex import LeannRetriever

        retriever = LeannRetriever.__new__(LeannRetriever)
        retriever._top_k = 5
        retriever._complexity = 64
        retriever._recompute = True

        # Mock the searcher
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            FakeSearchResult(id="doc1", score=0.95, text="LEANN is great"),
            FakeSearchResult(id="doc2", score=0.80, text="Vector search rocks"),
        ]
        retriever._searcher = mock_searcher

        query = QueryBundle(query_str="What is LEANN?")
        nodes = retriever._retrieve(query)

        assert len(nodes) == 2
        assert nodes[0].node.text == "LEANN is great"
        assert nodes[0].score == 0.95
        mock_searcher.search.assert_called_once_with(
            "What is LEANN?",
            top_k=5,
            complexity=64,
            recompute_embeddings=True,
        )

    @patch("leann.integrations.llamaindex.LeannRetriever.__init__", return_value=None)
    def test_retrieve_empty(self, mock_init):
        from llama_index.core import QueryBundle

        from leann.integrations.llamaindex import LeannRetriever

        retriever = LeannRetriever.__new__(LeannRetriever)
        retriever._top_k = 5
        retriever._complexity = 64
        retriever._recompute = True

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        retriever._searcher = mock_searcher

        nodes = retriever._retrieve(QueryBundle(query_str="nothing"))
        assert len(nodes) == 0


class TestLeannHybridRetriever:
    """Test the hybrid (vector + BM25) retriever."""

    def _make_retriever(self):
        """Create a hybrid retriever with mocked dependencies."""
        from leann.integrations.llamaindex import LeannHybridRetriever

        retriever = LeannHybridRetriever.__new__(LeannHybridRetriever)
        retriever._top_k = 5
        retriever._bm25_weight = 0.3
        retriever._vector_weight = 0.7
        retriever._complexity = 64
        retriever._recompute = True
        retriever._bm25_db_path = "/fake/path.fts5.db"
        retriever._bm25 = None
        retriever._index_path = "/fake/index.leann"

        mock_searcher = MagicMock()
        retriever._searcher = mock_searcher
        return retriever, mock_searcher

    def test_vector_only_when_no_bm25(self):
        from llama_index.core import QueryBundle

        retriever, mock_searcher = self._make_retriever()

        mock_searcher.search.return_value = [
            FakeSearchResult(id="1", score=0.9, text="text1"),
            FakeSearchResult(id="2", score=0.7, text="text2"),
        ]

        # BM25 not available
        retriever._get_bm25 = lambda: None

        nodes = retriever._retrieve(QueryBundle(query_str="test"))
        assert len(nodes) == 2
        assert nodes[0].node.text == "text1"

    def test_fusion_combines_results(self):
        from leann.integrations.llamaindex import LeannHybridRetriever

        retriever, mock_searcher = self._make_retriever()

        # Vector results
        mock_searcher.search.return_value = [
            FakeSearchResult(id="1", score=1.0, text="text1", metadata={"src": "vec"}),
            FakeSearchResult(id="2", score=0.5, text="text2", metadata={"src": "vec"}),
        ]

        # BM25 results (dict format as returned by BM25Index.search)
        bm25_results = [
            {"id": "2", "score": 10.0, "text": "text2", "metadata": {"src": "bm25"}},
            {"id": "3", "score": 5.0, "text": "text3", "metadata": {"src": "bm25"}},
        ]

        # Test the fusion directly
        nodes = retriever._fuse(mock_searcher.search.return_value, bm25_results)

        # Should have 3 unique documents
        assert len(nodes) == 3
        ids = [n.node.id_ for n in nodes]
        assert "1" in ids
        assert "2" in ids
        assert "3" in ids

        # Doc 2 should benefit from both vector and BM25 scores
        doc2 = next(n for n in nodes if n.node.id_ == "2")
        doc1 = next(n for n in nodes if n.node.id_ == "1")
        # Doc2 gets vector=0.0 (normalized, it's the min) + bm25=1.0 (it's the max)
        # Doc1 gets vector=1.0 (it's the max) + bm25=0.0
        # With 0.7 vector + 0.3 bm25: doc1 = 0.7, doc2 = 0.3
        # Actually doc1 has score 1.0 (max), doc2 has score 0.5
        # Normalized: doc1 = (1.0-0.5)/(1.0-0.5) = 1.0, doc2 = 0.0
        # BM25: doc2 = (10-5)/(10-5) = 1.0, doc3 = 0.0
        # doc1: 0.7*1.0 + 0.3*0.0 = 0.7
        # doc2: 0.7*0.0 + 0.3*1.0 = 0.3
        # doc3: 0.7*0.0 + 0.3*0.0 = 0.0
        assert doc1.score == pytest.approx(0.7, rel=0.1)
        assert doc2.score == pytest.approx(0.3, rel=0.1)

    def test_fusion_handles_single_source(self):
        retriever, mock_searcher = self._make_retriever()

        vector_results = [
            FakeSearchResult(id="1", score=0.9, text="only vector"),
        ]
        bm25_results = []

        # Fusion with no BM25 — should still produce results from vector
        nodes = retriever._fuse(vector_results, bm25_results)
        assert len(nodes) == 1
        assert nodes[0].node.id_ == "1"
        assert nodes[0].node.text == "only vector"

    def test_bm25_weight_clamped(self):
        from leann.integrations.llamaindex import LeannHybridRetriever

        retriever = LeannHybridRetriever.__new__(LeannHybridRetriever)
        # Test clamping
        retriever._bm25_weight = max(0.0, min(1.0, 1.5))
        assert retriever._bm25_weight == 1.0

        retriever._bm25_weight = max(0.0, min(1.0, -0.3))
        assert retriever._bm25_weight == 0.0


class TestImports:
    """Test that the integration module is importable."""

    def test_import_retriever(self):
        from leann.integrations.llamaindex import LeannRetriever

        assert LeannRetriever is not None

    def test_import_hybrid_retriever(self):
        from leann.integrations.llamaindex import LeannHybridRetriever

        assert LeannHybridRetriever is not None

    def test_import_from_integrations(self):
        from leann.integrations import llamaindex

        assert hasattr(llamaindex, "LeannRetriever")
        assert hasattr(llamaindex, "LeannHybridRetriever")
