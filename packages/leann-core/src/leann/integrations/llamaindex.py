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
from pathlib import Path
from typing import Any, Optional

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

    Performs both dense (vector) and sparse (BM25) retrieval, then merges
    results using weighted score fusion.

    Parameters
    ----------
    index_path:
        Path to the LEANN index file (``*.leann``).
    top_k:
        Number of results to return after fusion (default 10).
    bm25_weight:
        Weight for BM25 scores in [0, 1]. Vector weight is ``1 - bm25_weight``.
        Default 0.3 (70% vector, 30% BM25).
    bm25_db_path:
        Path to the BM25 SQLite FTS5 database. If ``None`` (default),
        derived from ``index_path`` as ``<index>.fts5.db``.
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
        bm25_db_path: Optional[str] = None,
        complexity: int = 64,
        recompute_embeddings: bool = True,
        **searcher_kwargs: Any,
    ):
        super().__init__()
        from leann.api import LeannSearcher

        self._index_path = index_path
        self._top_k = top_k
        self._bm25_weight = max(0.0, min(1.0, bm25_weight))
        self._vector_weight = 1.0 - self._bm25_weight
        self._complexity = complexity
        self._recompute = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

        # Resolve BM25 database path
        if bm25_db_path is None:
            idx = Path(index_path)
            bm25_db_path = str(idx.parent / f"{idx.name}.fts5.db")
        self._bm25_db_path = bm25_db_path
        self._bm25 = None

    def _get_bm25(self):
        """Lazily load the BM25 index."""
        if self._bm25 is not None:
            return self._bm25

        if not Path(self._bm25_db_path).exists():
            logger.warning(
                "BM25 index not found at %s. Falling back to vector-only search. "
                "Build with hybrid search enabled to create the BM25 index.",
                self._bm25_db_path,
            )
            return None

        try:
            from leann.bm25 import BM25Index
            self._bm25 = BM25Index(self._bm25_db_path)
            return self._bm25
        except ImportError:
            logger.warning("BM25 module not available. Using vector-only search.")
            return None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes using hybrid search (vector + BM25)."""
        query_str = query_bundle.query_str

        # Vector search — fetch more than top_k to allow fusion to pick best
        fetch_k = min(self._top_k * 3, self._top_k + 50)
        vector_results = self._searcher.search(
            query_str,
            top_k=fetch_k,
            complexity=self._complexity,
            recompute_embeddings=self._recompute,
        )

        # BM25 search — BM25Index.search returns list[tuple[str, float]];
        # normalize to list[dict] for the fusion step.
        bm25 = self._get_bm25()
        bm25_results: list[dict] = []
        if bm25 is not None:
            try:
                raw = bm25.search(query_str, top_k=fetch_k)
                bm25_results = [{"id": pid, "score": score} for pid, score in raw]
            except Exception as e:
                logger.warning("BM25 search failed: %s", e)

        # Fuse results
        if not bm25_results:
            # Fall back to vector-only
            return _results_to_nodes(vector_results[:self._top_k])

        return self._fuse(vector_results, bm25_results)

    def _fuse(
        self, vector_results: list, bm25_results: list
    ) -> list[NodeWithScore]:
        """Weighted score fusion of vector and BM25 results."""
        # Normalize vector scores to [0, 1]
        v_scores = {}
        if vector_results:
            max_v = max(r.score for r in vector_results)
            min_v = min(r.score for r in vector_results)
            span_v = max_v - min_v
            for r in vector_results:
                # When all scores are equal, use 0.5 ("equal relevance")
                # rather than 0.0 ("irrelevant").
                v_scores[r.id] = (r.score - min_v) / span_v if span_v else 0.5

        # Normalize BM25 scores to [0, 1]
        b_scores = {}
        if bm25_results:
            max_b = max(r["score"] for r in bm25_results)
            min_b = min(r["score"] for r in bm25_results)
            span_b = max_b - min_b
            for r in bm25_results:
                b_scores[r["id"]] = (r["score"] - min_b) / span_b if span_b else 0.5

        # Combine all IDs
        all_ids = set(v_scores.keys()) | set(b_scores.keys())

        # Build combined scores and text lookup
        text_lookup = {}
        meta_lookup = {}
        for r in vector_results:
            text_lookup[r.id] = r.text
            meta_lookup[r.id] = r.metadata
        for r in bm25_results:
            if r["id"] not in text_lookup:
                text_lookup[r["id"]] = r.get("text", "")
                meta_lookup[r["id"]] = r.get("metadata", {})

        scored = []
        for doc_id in all_ids:
            vs = v_scores.get(doc_id, 0.0)
            bs = b_scores.get(doc_id, 0.0)
            combined = self._vector_weight * vs + self._bm25_weight * bs
            scored.append((doc_id, combined))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        nodes = []
        for doc_id, score in scored[:self._top_k]:
            text = text_lookup.get(doc_id, "")
            metadata = meta_lookup.get(doc_id, {})
            node = TextNode(text=text, id_=doc_id, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

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
