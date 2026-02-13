"""LlamaIndex integration for LEANN (#217).

Exposes LEANN indexes as LlamaIndex ``BaseRetriever`` implementations, so you
can plug LEANN into any LlamaIndex ``RetrieverQueryEngine`` or agent pipeline.

Usage::

    from leann.integrations.llamaindex import LeannRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    retriever = LeannRetriever(index_path="/path/to/index.leann", top_k=10)
    engine = RetrieverQueryEngine.from_args(retriever)
    response = engine.query("What is LEANN?")

Hybrid search (BM25 + vector) is also supported::

    from leann.integrations.llamaindex import LeannHybridRetriever

    retriever = LeannHybridRetriever(
        index_path="/path/to/index.leann",
        top_k=10,
        bm25_weight=0.3,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

logger = logging.getLogger(__name__)


class LeannRetriever(BaseRetriever):
    """LlamaIndex retriever backed by a LEANN vector index.

    Parameters
    ----------
    index_path:
        Path to the LEANN index file (``*.leann``).
    top_k:
        Number of results to return (default 10).
    complexity:
        Search complexity / candidate list size (default 64).
    recompute_embeddings:
        Whether to recompute embeddings via the ZMQ server (default True).
    searcher_kwargs:
        Extra keyword arguments forwarded to ``LeannSearcher``.
    """

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        complexity: int = 64,
        recompute_embeddings: bool = True,
        **searcher_kwargs: Any,
    ):
        super().__init__()
        from leann.api import LeannSearcher

        self._index_path = index_path
        self._top_k = top_k
        self._complexity = complexity
        self._recompute = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes from the LEANN index."""
        query_str = query_bundle.query_str

        results = self._searcher.search(
            query_str,
            top_k=self._top_k,
            complexity=self._complexity,
            recompute_embeddings=self._recompute,
        )

        return _results_to_nodes(results)

    # Alias for older LlamaIndex versions that call retrieve()
    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Async retrieve (delegates to sync implementation)."""
        return self._retrieve(query_bundle)


class LeannHybridRetriever(BaseRetriever):
    """LlamaIndex retriever that fuses LEANN vector search with BM25.

    Delegates to ``LeannSearcher.search(sparse_score_ratio=...)`` which
    already implements weighted score fusion via ``leann.hybrid``.

    Parameters
    ----------
    index_path:
        Path to the LEANN index file (``*.leann``).
    top_k:
        Number of results to return after fusion (default 10).
    bm25_weight:
        Weight for BM25 scores in [0, 1]. Vector weight is ``1 - bm25_weight``.
        Default 0.3 (70% vector, 30% BM25).
    complexity:
        Search complexity for the vector backend (default 64).
    recompute_embeddings:
        Whether to recompute embeddings (default True).
    searcher_kwargs:
        Extra keyword arguments forwarded to ``LeannSearcher``.
    """

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        complexity: int = 64,
        recompute_embeddings: bool = True,
        **searcher_kwargs: Any,
    ):
        super().__init__()
        from leann.api import LeannSearcher

        self._index_path = index_path
        self._top_k = top_k
        self._bm25_weight = max(0.0, min(1.0, bm25_weight))
        self._complexity = complexity
        self._recompute = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes using hybrid search (vector + BM25)."""
        results = self._searcher.search(
            query_bundle.query_str,
            top_k=self._top_k,
            complexity=self._complexity,
            recompute_embeddings=self._recompute,
            sparse_score_ratio=self._bm25_weight,
        )
        return _results_to_nodes(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._retrieve(query_bundle)


def _results_to_nodes(results: list) -> list[NodeWithScore]:
    """Convert LEANN ``SearchResult`` objects to LlamaIndex ``NodeWithScore``."""
    nodes = []
    for r in results:
        node = TextNode(
            text=r.text,
            id_=r.id,
            metadata=r.metadata if isinstance(r.metadata, dict) else {},
        )
        nodes.append(NodeWithScore(node=node, score=r.score))
    return nodes
