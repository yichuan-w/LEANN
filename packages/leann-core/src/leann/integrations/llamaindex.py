from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from leann.api import LeannSearcher


def _results_to_nodes(results: list) -> list[NodeWithScore]:
    nodes = []

    for r in results:
        metadata = getattr(r, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        node = TextNode(text=r.text, id_=r.id, metadata=metadata)

        nodes.append(NodeWithScore(node=node, score=r.score))

    return nodes


class LeannRetriever(BaseRetriever):
    """LlamaIndex Retriever for LEANN"""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        complexity: int = 64,
        recompute_embeddings: bool = True,
        **searcher_kwargs: Any,
    ):
        super().__init__()
        self._top_k = top_k
        self._complexity = complexity
        self._recompute_embeddings = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes from LEANN index using pure vector search"""
        results = self._searcher.search(
            query=query_bundle.query_str,
            top_k=self._top_k,
            complexity=self._complexity,
            recompute_embeddings=self._recompute_embeddings,
        )

        return _results_to_nodes(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Async retrieve"""

        return self._retrieve(query_bundle)


class LeannHybridRetriever(BaseRetriever):
    """LlamaIndex retriever with hybrid search (vector + BM25).
    Parameters
    ----------
    index_path : str
        Path to LEANN index file (*.leann)
    top_k : int
        Number of results to return (default 10)
    bm25_weight : float
        Weight for BM25 (keyword) search, range [0, 1] (default 0.3)
        - 0.0 = pure vector search (no keywords)
        - 0.3 = 70% vector, 30% keywords (recommended)
        - 0.5 = balanced hybrid search
        - 1.0 = pure keyword search (no vectors)
    Notes
    -----
    Internally converts `bm25_weight` to LEANN's `gemma` parameter:
        gemma = 1.0 - bm25_weight
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

        self._bm25_weight = max(0.0, min(1.0, bm25_weight))
        self._gemma = 1.0 - self._bm25_weight
        self._top_k = top_k
        self._complexity = complexity
        self._recompute = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes from LEANN index using hybrid search"""

        results = self._searcher.search(
            query=query_bundle.query_str,
            top_k=self._top_k,
            complexity=self._complexity,
            recompute_embeddings=self._recompute,
            gemma=self._gemma,
        )

        return _results_to_nodes(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Async retrieve"""

        return self._retrieve(query_bundle)
