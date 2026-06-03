"""LlamaIndex retrievers backed by LEANN indexes."""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from leann.api import LeannSearcher, SearchResult


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    return dict(metadata) if isinstance(metadata, dict) else {}


def _distance_metric_from_searcher(searcher: Any) -> str:
    meta = getattr(searcher, "meta_data", {})
    backend_name = meta.get("backend_name", "")
    default_metric = "l2" if backend_name == "ivf" else "mips"
    return meta.get("backend_kwargs", {}).get("distance_metric", default_metric).lower()


def _results_to_nodes(
    results: list[SearchResult],
    lower_is_better_scores: bool = False,
) -> list[NodeWithScore]:
    nodes: list[NodeWithScore] = []
    for result in results:
        raw_score = float(result.score)
        score = 1.0 / (1.0 + max(raw_score, 0.0)) if lower_is_better_scores else raw_score
        metadata = _normalize_metadata(result.metadata)
        if lower_is_better_scores:
            metadata["leann_raw_score"] = raw_score
        node = TextNode(
            id_=str(result.id),
            text=result.text,
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=score))
    return nodes


def _validate_weight(name: str, value: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0.0 and 1.0") from exc
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return weight


def _normalize_search_kwargs(search_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(search_kwargs or {})
    if "vector_weight" in normalized:
        raise ValueError(
            "vector_weight is controlled by the retriever; use LeannHybridRetriever "
            "bm25_weight to configure dense-vector versus keyword weighting"
        )
    return normalized


class LeannRetriever(BaseRetriever):
    """LlamaIndex retriever for pure dense-vector LEANN search."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        complexity: int = 64,
        recompute_embeddings: bool | None = None,
        search_kwargs: dict[str, Any] | None = None,
        callback_manager: Any = None,
        object_map: dict[Any, Any] | None = None,
        objects: list[Any] | None = None,
        verbose: bool = False,
        **searcher_kwargs: Any,
    ) -> None:
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
        self._top_k = top_k
        self._complexity = complexity
        self._search_kwargs = _normalize_search_kwargs(search_kwargs)
        if recompute_embeddings is not None:
            searcher_kwargs["recompute_embeddings"] = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    @staticmethod
    def _query_text(query_bundle: QueryBundle) -> str:
        if query_bundle.custom_embedding_strs is not None or query_bundle.embedding is not None:
            raise ValueError(
                "LEANN LlamaIndex retrievers accept plain text QueryBundle values only; "
                "custom embeddings and custom embedding strings are not supported."
            )
        return query_bundle.query_str

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        search_kwargs: dict[str, Any] = {
            "query": self._query_text(query_bundle),
            "top_k": self._top_k,
            "complexity": self._complexity,
            "vector_weight": 1.0,
        }
        search_kwargs.update(self._search_kwargs)
        lower_is_better = search_kwargs["vector_weight"] == 1.0 and (
            _distance_metric_from_searcher(self._searcher) == "l2"
        )
        return _results_to_nodes(
            self._searcher.search(**search_kwargs),
            lower_is_better_scores=lower_is_better,
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve, query_bundle)


class LeannHybridRetriever(BaseRetriever):
    """LlamaIndex retriever for LEANN hybrid dense-vector plus BM25 search.

    `bm25_weight` is the keyword-side weight. LEANN's public `vector_weight`
    is therefore `1.0 - bm25_weight`.
    """

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        complexity: int = 64,
        recompute_embeddings: bool | None = None,
        search_kwargs: dict[str, Any] | None = None,
        callback_manager: Any = None,
        object_map: dict[Any, Any] | None = None,
        objects: list[Any] | None = None,
        verbose: bool = False,
        **searcher_kwargs: Any,
    ) -> None:
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
        self._top_k = top_k
        self._complexity = complexity
        self._bm25_weight = _validate_weight("bm25_weight", bm25_weight)
        self._vector_weight = 1.0 - self._bm25_weight
        self._search_kwargs = _normalize_search_kwargs(search_kwargs)
        if recompute_embeddings is not None:
            searcher_kwargs["recompute_embeddings"] = recompute_embeddings
        self._searcher = LeannSearcher(index_path, **searcher_kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        search_kwargs: dict[str, Any] = {
            "query": LeannRetriever._query_text(query_bundle),
            "top_k": self._top_k,
            "complexity": self._complexity,
            "vector_weight": self._vector_weight,
        }
        search_kwargs.update(self._search_kwargs)
        lower_is_better = search_kwargs["vector_weight"] == 1.0 and (
            _distance_metric_from_searcher(self._searcher) == "l2"
        )
        return _results_to_nodes(
            self._searcher.search(**search_kwargs),
            lower_is_better_scores=lower_is_better,
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve, query_bundle)
