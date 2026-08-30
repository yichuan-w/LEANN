from unittest.mock import patch

from leann.integrations.llamaindex import LeannHybridRetriever, _results_to_nodes
from llama_index.core.schema import NodeWithScore


class MockSearchResult:
    def __init__(self, id, score, text, metadata):
        self.id = id
        self.score = score
        self.text = text
        self.metadata = metadata


def test_results_to_nodes():
    """Test converting LEANN SearchResults to LlamaIndex Nodes."""
    results = [
        MockSearchResult("1", 0.9, "text1", {"source": "doc1"}),
        MockSearchResult("2", 0.8, "text2", None),  # Should normalize None metadata to {}
        MockSearchResult("3", 0.7, "text3", "not a dict"),  # Should normalize string metadata to {}
    ]

    nodes = _results_to_nodes(results)
    assert len(nodes) == 3
    assert isinstance(nodes[0], NodeWithScore)
    assert nodes[0].node.id_ == "1"
    assert nodes[0].node.text == "text1"
    assert nodes[0].node.metadata == {"source": "doc1"}

    assert nodes[1].node.metadata == {}
    assert nodes[2].node.metadata == {}


@patch("leann.integrations.llamaindex.LeannSearcher")
def test_leann_hybrid_retriever_bm25_weight(mock_searcher):
    """Test that bm25_weight is correctly converted to gemma."""
    retriever1 = LeannHybridRetriever("dummy_path", bm25_weight=0.3)
    assert retriever1._gemma == 0.7

    retriever2 = LeannHybridRetriever("dummy_path", bm25_weight=0.0)
    assert retriever2._gemma == 1.0  # Pure vector search

    retriever3 = LeannHybridRetriever("dummy_path", bm25_weight=1.0)
    assert retriever3._gemma == 0.0  # Pure keyword search

    # Test clamping
    retriever4 = LeannHybridRetriever("dummy_path", bm25_weight=1.5)
    assert retriever4._gemma == 0.0  # clamped to 1.0 -> 0.0 gemma
