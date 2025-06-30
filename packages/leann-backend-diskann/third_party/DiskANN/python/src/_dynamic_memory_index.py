# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorIdentifier,
    VectorIdentifierBatch,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_existing_directory,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
    _write_index_metadata,
)
from ._diskannpy import defaults

__ALL__ = ["DynamicMemoryIndex"]


class DynamicMemoryIndex:
    """
    A DynamicMemoryIndex instance is used to both search and mutate a `diskannpy` memory index. This index is unlike
    either `diskannpy.StaticMemoryIndex` or `diskannpy.StaticDiskIndex` in the following ways:

    - It requires an explicit vector identifier for each vector added to it.
    - Insert and (lazy) deletion operations are provided for a flexible, living index

    The mutable aspect of this index will absolutely impact search time performance as new vectors are added and
    old deleted. `DynamicMemoryIndex.consolidate_deletes()` should be called periodically to restructure the index
    to remove deleted vectors and improve per-search performance, at the cost of an expensive index consolidation to
    occur.
    """

    @classmethod
    def from_file(
        cls,
        index_directory: str,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
        index_prefix: str = "ann",
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
    ) -> "DynamicMemoryIndex":
        """
        The `from_file` classmethod is used to load a previously saved index from disk. This index *must* have been
        created with a valid `tags` file or `tags` np.ndarray of `diskannpy.VectorIdentifier`s. It is *strongly*
        recommended that you use the same parameters as the `diskannpy.build_memory_index()` function that created
        the index.

        ### Parameters
        - **index_directory**: The directory containing the index files. This directory must contain the following
            files:
            - `{index_prefix}.data`
            - `{index_prefix}.tags`
            - `{index_prefix}`

          It may also include the following optional files:
            - `{index_prefix}_vectors.bin`: Optional. `diskannpy` builder functions may create this file in the
              `index_directory` if the index was created from a numpy array
            - `{index_prefix}_metadata.bin`: Optional. `diskannpy` builder functions create this file to store metadata
            about the index, such as vector dtype, distance metric, number of vectors and vector dimensionality.
            If an index is built from the `diskann` cli tools, this file will not exist.
        - **max_vectors**: Capacity of the memory index including space for future insertions.
        - **complexity**: Complexity (a.k.a `L`) references the size of the list we store candidate approximate
          neighbors in. It's used during save (which is an index rebuild), and it's used as an initial search size to
          warm up our index and lower the latency for initial real searches.
        - **graph_degree**: Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph
          structure. This degree will be pruned throughout the course of the index build, but it will never grow beyond
          this value. Higher R values require longer index build times, but may result in an index showing excellent
          recall and latency characteristics.
        - **saturate_graph**: If True, the adjacency list of each node will be saturated with neighbors to have exactly
          `graph_degree` neighbors. If False, each node will have between 1 and `graph_degree` neighbors.
        - **max_occlusion_size**: The maximum number of points that can be considered by occlude_list function.
        - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
          graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably
          more distance comparisons compared to a lower alpha value.
        - **num_threads**: Number of threads to use when creating this index. `0` indicates we should use all available
          logical processors.
        - **filter_complexity**: Complexity to use when using filters. Default is 0.
        - **num_frozen_points**: Number of points to freeze. Default is 1.
        - **initial_search_complexity**: Should be set to the most common `complexity` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `search` or `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **search_threads**: Should be set to the most common `num_threads` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **concurrent_consolidation**: This flag dictates whether consolidation can be run alongside inserts and
          deletes, or whether the index is locked down to changes while consolidation is ongoing.
        - **index_prefix**: The prefix of the index files. Defaults to "ann".
        - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
          vector dtypes, but `mips` is only available for single precision floats. Default is `None`. **This
          value is only used if a `{index_prefix}_metadata.bin` file does not exist.** If it does not exist,
          you are required to provide it.
        - **vector_dtype**: The vector dtype this index has been built with. **This value is only used if a
          `{index_prefix}_metadata.bin` file does not exist.** If it does not exist, you are required to provide it.
        - **dimensions**: The vector dimensionality of this index. All new vectors inserted must be the same
          dimensionality. **This value is only used if a `{index_prefix}_metadata.bin` file does not exist.** If it
          does not exist, you are required to provide it.

        ### Returns
        A `diskannpy.DynamicMemoryIndex` object, with the index loaded from disk and ready to use for insertions,
        deletions, and searches.

        """
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)

        # do tags exist?
        tags_file = index_prefix_path + ".tags"
        _assert(
            Path(tags_file).exists(),
            f"The file {tags_file} does not exist in {index_directory}",
        )
        vector_dtype, dap_metric, num_vectors, dimensions = _ensure_index_metadata(
            index_prefix_path, vector_dtype, distance_metric, max_vectors, dimensions, warn_size_exceeded=True
        )

        index = cls(
            distance_metric=dap_metric,  # type: ignore
            vector_dtype=vector_dtype,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation,
        )
        index._index.load(index_prefix_path)
        index._num_vectors = num_vectors  # current number of vectors loaded
        return index

    def __init__(
        self,
        distance_metric: DistanceMetric,
        vector_dtype: VectorDType,
        dimensions: int,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
    ):
        """
        The `diskannpy.DynamicMemoryIndex` represents our python API into a mutable DiskANN memory index.

        This constructor is used to create a new, empty index. If you wish to load a previously saved index from disk,
        please use the `diskannpy.DynamicMemoryIndex.from_file` classmethod instead.

        ### Parameters
        - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
          vector dtypes, but `mips` is only available for single precision floats.
        - **vector_dtype**: One of {`np.float32`, `np.int8`, `np.uint8`}. The dtype of the vectors this index will
          be storing.
        - **dimensions**: The vector dimensionality of this index. All new vectors inserted must be the same
          dimensionality.
        - **max_vectors**: Capacity of the data store including space for future insertions
        - **graph_degree**: Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph
          structure. This degree will be pruned throughout the course of the index build, but it will never grow beyond
          this value. Higher `graph_degree` values require longer index build times, but may result in an index showing
          excellent recall and latency characteristics.
        - **saturate_graph**: If True, the adjacency list of each node will be saturated with neighbors to have exactly
          `graph_degree` neighbors. If False, each node will have between 1 and `graph_degree` neighbors.
        - **max_occlusion_size**: The maximum number of points that can be considered by occlude_list function.
        - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
          graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably
          more distance comparisons compared to a lower alpha value.
        - **num_threads**: Number of threads to use when creating this index. `0` indicates we should use all available
          logical processors.
        - **filter_complexity**: Complexity to use when using filters. Default is 0.
        - **num_frozen_points**: Number of points to freeze. Default is 1.
        - **initial_search_complexity**: Should be set to the most common `complexity` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `search` or `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **search_threads**: Should be set to the most common `num_threads` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **concurrent_consolidation**: This flag dictates whether consolidation can be run alongside inserts and
          deletes, or whether the index is locked down to changes while consolidation is ongoing.

        """
        self._num_vectors = 0
        self._removed_num_vectors = 0
        dap_metric = _valid_metric(distance_metric)
        self._dap_metric = dap_metric
        _assert_dtype(vector_dtype)
        _assert_is_positive_uint32(dimensions, "dimensions")

        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        _assert_is_positive_uint32(max_vectors, "max_vectors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(graph_degree, "graph_degree")
        _assert(
            alpha >= 1,
            "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)",
        )
        _assert_is_nonnegative_uint32(max_occlusion_size, "max_occlusion_size")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
        _assert_is_nonnegative_uint32(num_frozen_points, "num_frozen_points")
        _assert_is_nonnegative_uint32(
            initial_search_complexity, "initial_search_complexity"
        )
        _assert_is_nonnegative_uint32(search_threads, "search_threads")

        self._max_vectors = max_vectors
        self._complexity = complexity
        self._graph_degree = graph_degree

        if vector_dtype == np.uint8:
            _index = _native_dap.DynamicMemoryUInt8Index
        elif vector_dtype == np.int8:
            _index = _native_dap.DynamicMemoryInt8Index
        else:
            _index = _native_dap.DynamicMemoryFloatIndex

        self._index = _index(
            distance_metric=dap_metric,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation,
        )
        self._points_deleted = False

    def search(
        self, query: VectorLike, k_neighbors: int, complexity: int
    ) -> QueryResponse:
        """
        Searches the index by a single query vector.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        """
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert(
            _query.shape[0] == self._dimensions,
            f"query vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_query.shape[0]}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors
        neighbors, distances = self._index.search(query=_query, knn=k_neighbors, complexity=complexity)
        return QueryResponse(identifiers=neighbors, distances=distances)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
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
        """
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert_2d(_queries, "queries")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}",
        )

        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = queries.shape
        neighbors, distances = self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )
        return QueryResponseBatch(identifiers=neighbors, distances=distances)

    def save(self, save_path: str, index_prefix: str = "ann"):
        """
        Saves this index to file.

        ### Parameters
        - **save_path**: The path to save these index files to.
        - **index_prefix**: The prefix of the index files. Defaults to "ann".
        """
        if save_path == "":
            raise ValueError("save_path cannot be empty")
        if index_prefix == "":
            raise ValueError("index_prefix cannot be empty")

        index_prefix = index_prefix.format(complexity=self._complexity, graph_degree=self._graph_degree)
        _assert_existing_directory(save_path, "save_path")
        save_path = os.path.join(save_path, index_prefix)
        if self._points_deleted is True:
            warnings.warn(
                "DynamicMemoryIndex.save() currently requires DynamicMemoryIndex.consolidate_delete() to be called "
                "prior to save when items have been marked for deletion. This is being done automatically now, though"
                "it will increase the time it takes to save; on large sets of data it can take a substantial amount of "
                "time. In the future, we will implement a faster save with unconsolidated deletes, but for now this is "
                "required."
            )
            self._index.consolidate_delete()
        self._index.save(
            save_path=save_path, compact_before_save=True
        )  # we do not yet support uncompacted saves
        _write_index_metadata(
            save_path,
            self._vector_dtype,
            self._dap_metric,
            self._index.num_points(),
            self._dimensions,
        )

    def insert(self, vector: VectorLike, vector_id: VectorIdentifier):
        """
        Inserts a single vector into the index with the provided vector_id.

        If this insertion will overrun the `max_vectors` count boundaries of this index, `consolidate_delete()` will
        be executed automatically.

        ### Parameters
        - **vector**: The vector to insert. Note that dtype must match.
        - **vector_id**: The vector_id to use for this vector.
        """
        _vector = _castable_dtype_or_raise(vector, expected=self._vector_dtype)
        _assert(len(vector.shape) == 1, "insert vector must be 1-d")
        _assert_is_positive_uint32(vector_id, "vector_id")
        if self._num_vectors + 1 > self._max_vectors:
            if self._removed_num_vectors > 0:
                warnings.warn(f"Inserting this vector would overrun the max_vectors={self._max_vectors} specified at index "
                              f"construction. We are attempting to consolidate_delete() to make space.")
                self.consolidate_delete()
            else:
                raise RuntimeError(f"Inserting this vector would overrun the max_vectors={self._max_vectors} specified "
                                   f"at index construction. Unable to make space by consolidating deletions. The insert"
                                   f"operation has failed.")
        status = self._index.insert(_vector, np.uint32(vector_id))
        if status == 0:
            self._num_vectors += 1
        else:
            raise RuntimeError(
                f"Insert was unable to complete successfully; error code returned from diskann C++ lib: {status}"
            )


    def batch_insert(
        self,
        vectors: VectorLikeBatch,
        vector_ids: VectorIdentifierBatch,
        num_threads: int = 0,
    ):
        """
        Inserts a batch of vectors into the index with the provided vector_ids.

        If this batch insertion will overrun the `max_vectors` count boundaries of this index, `consolidate_delete()`
        will be executed automatically.

        ### Parameters
        - **vectors**: The 2d numpy array of vectors to insert.
        - **vector_ids**: The 1d array of vector ids to use. This array must have the same number of elements as
            the vectors array has rows. The dtype of vector_ids must be `np.uint32`
        - **num_threads**: Number of threads to use when inserting into this index. (>= 0), 0 = num_threads in system
        """
        _query = _castable_dtype_or_raise(vectors, expected=self._vector_dtype)
        _assert(len(vectors.shape) == 2, "vectors must be a 2-d array")
        _assert(
            vectors.shape[0] == vector_ids.shape[0],
            "Number of vectors must be equal to number of ids",
        )
        _vectors = vectors.astype(dtype=self._vector_dtype, casting="safe", copy=False)
        _vector_ids = vector_ids.astype(dtype=np.uint32, casting="safe", copy=False)

        if self._num_vectors + _vector_ids.shape[0] > self._max_vectors:
            if self._max_vectors + self._removed_num_vectors >= _vector_ids.shape[0]:
                warnings.warn(f"Inserting these vectors, count={_vector_ids.shape[0]} would overrun the "
                              f"max_vectors={self._max_vectors} specified at index construction. We are attempting to "
                              f"consolidate_delete() to make space.")
                self.consolidate_delete()
            else:
                raise RuntimeError(f"Inserting these vectors count={_vector_ids.shape[0]} would overrun the "
                                   f"max_vectors={self._max_vectors} specified at index construction. Unable to make "
                                   f"space by consolidating deletions. The batch insert operation has failed.")

        statuses = self._index.batch_insert(
            _vectors, _vector_ids, _vector_ids.shape[0], num_threads
        )
        successes = []
        failures = []
        for i in range(0, len(statuses)):
            if statuses[i] == 0:
                successes.append(i)
            else:
                failures.append(i)
        self._num_vectors += len(successes)
        if len(failures) == 0:
            return
        failed_ids = vector_ids[failures]
        raise RuntimeError(
            f"During batch insert, the following vector_ids were unable to be inserted into the index: {failed_ids}. "
            f"{len(successes)} were successfully inserted"
        )


    def mark_deleted(self, vector_id: VectorIdentifier):
        """
        Mark vector for deletion. This is a soft delete that won't return the vector id in any results, but does not
        remove it from the underlying index files or memory structure. To execute a hard delete, call this method and
        then call the much more expensive `consolidate_delete` method on this index.
        ### Parameters
        - **vector_id**: The vector id to delete. Must be a uint32.
        """
        _assert_is_positive_uint32(vector_id, "vector_id")
        self._points_deleted = True
        self._removed_num_vectors += 1
        # we do not decrement self._num_vectors until consolidate_delete
        self._index.mark_deleted(np.uint32(vector_id))

    def consolidate_delete(self):
        """
        This method actually restructures the DiskANN index to remove the items that have been marked for deletion.
        """
        self._index.consolidate_delete()
        self._points_deleted = False
        self._num_vectors -= self._removed_num_vectors
        self._removed_num_vectors = 0
