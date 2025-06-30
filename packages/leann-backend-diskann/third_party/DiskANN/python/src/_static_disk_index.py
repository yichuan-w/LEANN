# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from typing import Optional

import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_2d,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
)

__ALL__ = ["StaticDiskIndex"]


class StaticDiskIndex:
    """
    A StaticDiskIndex is a disk-backed index that is not mutable.
    """

    def __init__(
        self,
        index_directory: str,
        num_threads: int,
        num_nodes_to_cache: int,
        cache_mechanism: int = 1,
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
        index_prefix: str = "ann",
        pq_prefix: str = "",
        partition_prefix: str = "",
    ):
        """
        ### Parameters
        - **index_directory**: The directory containing the index files. This directory must contain the following
            files:
            - `{index_prefix}_sample_data.bin`
            - `{index_prefix}_mem.index.data`
            - `{index_prefix}_pq_compressed.bin`
            - `{index_prefix}_pq_pivots.bin`
            - `{index_prefix}_sample_ids.bin`
            - `{index_prefix}_disk.index`

          It may also include the following optional files:
            - `{index_prefix}_vectors.bin`: Optional. `diskannpy` builder functions may create this file in the
              `index_directory` if the index was created from a numpy array
            - `{index_prefix}_metadata.bin`: Optional. `diskannpy` builder functions create this file to store metadata
            about the index, such as vector dtype, distance metric, number of vectors and vector dimensionality.
            If an index is built from the `diskann` cli tools, this file will not exist.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        - **num_nodes_to_cache**: Number of nodes to cache in memory (> -1)
        - **cache_mechanism**: 1 -> use the generated sample_data.bin file for
            the index to initialize a set of cached nodes, up to `num_nodes_to_cache`, 2 -> ready the cache for up to
            `num_nodes_to_cache`, but do not initialize it with any nodes. Any other value disables node caching.
        - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
          vector dtypes, but `mips` is only available for single precision floats. Default is `None`. **This
          value is only used if a `{index_prefix}_metadata.bin` file does not exist.** If it does not exist,
          you are required to provide it.
        - **vector_dtype**: The vector dtype this index has been built with. **This value is only used if a
          `{index_prefix}_metadata.bin` file does not exist.** If it does not exist, you are required to provide it.
        - **dimensions**: The vector dimensionality of this index. All new vectors inserted must be the same
          dimensionality. **This value is only used if a `{index_prefix}_metadata.bin` file does not exist.** If it
          does not exist, you are required to provide it.
        - **index_prefix**: The prefix of the index files. Defaults to "ann".
        """
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)
        vector_dtype, metric, _, _ = _ensure_index_metadata(
            index_prefix_path,
            vector_dtype,
            distance_metric,
            1,  # it doesn't matter because we don't need it in this context anyway
            dimensions,
        )
        dap_metric = _valid_metric(metric)

        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(num_nodes_to_cache, "num_nodes_to_cache")

        self._vector_dtype = vector_dtype
        if vector_dtype == np.uint8:
            _index = _native_dap.StaticDiskUInt8Index
        elif vector_dtype == np.int8:
            _index = _native_dap.StaticDiskInt8Index
        else:
            _index = _native_dap.StaticDiskFloatIndex
        self._index = _index(
            distance_metric=dap_metric,
            index_path_prefix=index_prefix_path,
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=cache_mechanism,
            pq_prefix=pq_prefix,
            partition_prefix=partition_prefix,
        )
        print("After index init")

    def search(
        self,
        query: VectorLike,
        k_neighbors: int,
        complexity: int,
        beam_width: int = 2,
        USE_DEFERRED_FETCH: bool = False,
        skip_search_reorder: bool = False,
        recompute_beighbor_embeddings: bool = False,
        dedup_node_dis: bool = False,
        prune_ratio: float = 0,
        batch_recompute: bool = False,
        global_pruning: bool = False,
    ) -> QueryResponse:
        """
        Searches the index by a single query vector.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **beam_width**: The beamwidth to be used for search. This is the maximum number of IO requests each query
          will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query,
          but might result in slightly higher total number of IO requests to SSD per query. For the highest query
          throughput with a fixed SSD IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
          Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will
          involve some tuning overhead.
        - **skip_search_reorder**: Whether to skip search reorder for diskann search.
        - **recompute_beighbor_embeddings**: Whether to recompute the neighbor embeddings.
        - **dedup_node_dis**: Whether to dedup node distances.
        - **batch_recompute**: Whether to batch recompute.
        """
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(beam_width, "beam_width")

        if k_neighbors > complexity:
            warnings.warn(
                f"{k_neighbors=} asked for, but {complexity=} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        neighbors, distances = self._index.search(
            query=_query,
            knn=k_neighbors,
            complexity=complexity,
            beam_width=beam_width,
            USE_DEFERRED_FETCH=USE_DEFERRED_FETCH,
            skip_search_reorder=skip_search_reorder,
            recompute_beighbor_embeddings=recompute_beighbor_embeddings,
            dedup_node_dis=dedup_node_dis,
            prune_ratio=prune_ratio,
            batch_recompute=batch_recompute,
            global_pruning=global_pruning,
        )
        return QueryResponse(identifiers=neighbors, distances=distances)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
        beam_width: int = 2,
        USE_DEFERRED_FETCH: bool = False,
        skip_search_reorder: bool = False,
        recompute_beighbor_embeddings: bool = False,
        dedup_node_dis: bool = False,
        prune_ratio: float = 0,
        batch_recompute: bool = False,
        global_pruning: bool = False,
    ) -> QueryResponseBatch:
        """
        Searches the index by a batch of query vectors.

        This search is parallelized and far more efficient than searching for each vector individually.

        ### Parameters
        - **queries**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of queries intended to search for in parallel. Dtype must match dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        - **beam_width**: The beamwidth to be used for search. This is the maximum number of IO requests each query
          will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query,
          but might result in slightly higher total number of IO requests to SSD per query. For the highest query
          throughput with a fixed SSD IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
          Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will
          involve some tuning overhead.
        - **skip_search_reorder**: Whether to skip search reorder for diskann search.
        """
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert_2d(_queries, "queries")
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(beam_width, "beam_width")

        if k_neighbors > complexity:
            warnings.warn(
                f"{k_neighbors=} asked for, but {complexity=} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = _queries.shape
        print(
            f"USE_DEFERRED_FETCH={USE_DEFERRED_FETCH} skip_search_reorder={skip_search_reorder} recompute_beighbor_embeddings={recompute_beighbor_embeddings}, dedup_node_dis={dedup_node_dis}"
        )
        neighbors, distances = self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            beam_width=beam_width,
            num_threads=num_threads,
            USE_DEFERRED_FETCH=USE_DEFERRED_FETCH,
            skip_search_reorder=skip_search_reorder,
            recompute_beighbor_embeddings=recompute_beighbor_embeddings,
            dedup_node_dis=dedup_node_dis,
            prune_ratio=prune_ratio,
            batch_recompute=batch_recompute,
            global_pruning=global_pruning,
        )
        return QueryResponseBatch(identifiers=neighbors, distances=distances)
