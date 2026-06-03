import asyncio
from typing import Any, cast

import pytest
from leann.api import SearchResult
from leann.integrations.llamaindex import (
    LeannHybridRetriever,
    LeannRetriever,
    _results_to_nodes,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def _text_node(node: NodeWithScore) -> TextNode:
    return cast(TextNode, node.node)


def test_results_to_nodes_preserves_text_id_score_and_dict_metadata():
    results = [
        SearchResult(id="p1", score=0.9, text="alpha text", metadata={"source": "doc.md"}),
        SearchResult(id="p2", score=0.4, text="beta text", metadata={}),
    ]

    nodes = _results_to_nodes(results)

    assert len(nodes) == 2
    assert isinstance(nodes[0], NodeWithScore)
    assert nodes[0].node.id_ == "p1"
    assert _text_node(nodes[0]).text == "alpha text"
    assert nodes[0].node.metadata == {"source": "doc.md"}
    assert nodes[0].score == 0.9
    assert nodes[1].node.id_ == "p2"
    assert nodes[1].node.metadata == {}


def test_results_to_nodes_normalizes_non_dict_metadata():
    result = SearchResult(id="p1", score=1.0, text="text", metadata=cast(Any, "not metadata"))

    nodes = _results_to_nodes([result])

    assert nodes[0].node.metadata == {}


def test_results_to_nodes_converts_l2_distance_scores_and_preserves_raw_score():
    result = SearchResult(id="p1", score=3.0, text="text", metadata={"source": "doc"})

    nodes = _results_to_nodes([result], lower_is_better_scores=True)

    assert nodes[0].score == 0.25
    assert nodes[0].node.metadata == {"source": "doc", "leann_raw_score": 3.0}


def test_leann_retriever_calls_search_with_vector_weight(monkeypatch):
    calls = []

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self.meta_data = {"backend_name": "hnsw", "backend_kwargs": {"distance_metric": "mips"}}

        def search(self, **kwargs):
            calls.append(("search", kwargs))
            return [SearchResult(id="p1", score=0.5, text="result", metadata={})]

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    retriever = LeannRetriever(
        "idx.leann",
        top_k=3,
        complexity=24,
        recompute_embeddings=False,
        search_kwargs={"metadata_filters": {"source": {"==": "doc"}}},
        use_daemon=False,
        verbose=True,
    )
    nodes = retriever._retrieve(QueryBundle("where is search?"))

    assert calls[0] == (
        "init",
        ("idx.leann",),
        {"use_daemon": False, "recompute_embeddings": False},
    )
    assert calls[1] == (
        "search",
        {
            "query": "where is search?",
            "top_k": 3,
            "complexity": 24,
            "vector_weight": 1.0,
            "metadata_filters": {"source": {"==": "doc"}},
        },
    )
    assert _text_node(nodes[0]).text == "result"


def test_leann_hybrid_retriever_maps_bm25_weight_to_vector_weight(monkeypatch):
    calls = []

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.meta_data = {"backend_name": "hnsw", "backend_kwargs": {"distance_metric": "mips"}}

        def search(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    retriever = LeannHybridRetriever("idx.leann", bm25_weight=0.3, top_k=4, complexity=16)
    nodes = retriever._retrieve(QueryBundle("exact identifier"))

    assert nodes == []
    assert calls == [
        {
            "query": "exact identifier",
            "top_k": 4,
            "complexity": 16,
            "vector_weight": 0.7,
        }
    ]


@pytest.mark.parametrize("retriever_cls", [LeannRetriever, LeannHybridRetriever])
def test_retrievers_reject_vector_weight_search_kwarg(monkeypatch, retriever_cls):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    with pytest.raises(ValueError, match="vector_weight is controlled"):
        retriever_cls("idx.leann", search_kwargs={"vector_weight": 0.2})


def test_leann_retriever_rejects_query_bundle_embeddings(monkeypatch):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.meta_data = {}

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    retriever = LeannRetriever("idx.leann")

    with pytest.raises(ValueError, match="custom embeddings"):
        retriever._retrieve(QueryBundle("query", embedding=[1.0]))


def test_leann_hybrid_retriever_converts_l2_scores_only_for_pure_vector(monkeypatch):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.meta_data = {"backend_name": "ivf", "backend_kwargs": {"distance_metric": "l2"}}

        def search(self, **kwargs):
            return [SearchResult(id="p1", score=3.0, text="result", metadata={})]

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    pure_vector = LeannHybridRetriever("idx.leann", bm25_weight=0.0)
    hybrid = LeannHybridRetriever("idx.leann", bm25_weight=0.3)

    pure_vector_node = pure_vector._retrieve(QueryBundle("q"))[0]
    hybrid_node = hybrid._retrieve(QueryBundle("q"))[0]

    assert pure_vector_node.score == 0.25
    assert pure_vector_node.node.metadata["leann_raw_score"] == 3.0
    assert hybrid_node.score == 3.0


def test_leann_retriever_async_path_runs_sync_retrieve(monkeypatch):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.meta_data = {"backend_name": "hnsw", "backend_kwargs": {"distance_metric": "mips"}}

        def search(self, **kwargs):
            return [SearchResult(id="p1", score=0.5, text="result", metadata={})]

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    retriever = LeannRetriever("idx.leann")

    nodes = asyncio.run(retriever._aretrieve(QueryBundle("q")))

    assert _text_node(nodes[0]).text == "result"


@pytest.mark.parametrize("bm25_weight", [-0.1, 1.1, "heavy"])
def test_leann_hybrid_retriever_rejects_invalid_bm25_weight(monkeypatch, bm25_weight):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("leann.integrations.llamaindex.LeannSearcher", DummySearcher)

    with pytest.raises(ValueError, match="bm25_weight"):
        LeannHybridRetriever("idx.leann", bm25_weight=bm25_weight)
