import diskannpy as dap
import numpy as np
import numpy.typing as npt

import fire

from contextlib import contextmanager
from time import perf_counter

from typing import Tuple


def _basic_setup(
    dtype: str,
    query_vectors_file: str
) -> Tuple[dap.VectorDType, npt.NDArray[dap.VectorDType]]:
    _dtype = dap.valid_dtype(dtype)
    vectors_to_query = dap.vectors_from_binary(query_vectors_file, dtype=_dtype)
    return _dtype, vectors_to_query


def dynamic(
    dtype: str,
    index_vectors_file: str,
    query_vectors_file: str,
    build_complexity: int,
    graph_degree: int,
    K: int,
    search_complexity: int,
    num_insert_threads: int,
    num_search_threads: int,
    gt_file: str = "",
):
    _dtype, vectors_to_query = _basic_setup(dtype, query_vectors_file)
    vectors_to_index = dap.vectors_from_binary(index_vectors_file, dtype=_dtype)

    npts, ndims = vectors_to_index.shape
    index = dap.DynamicMemoryIndex(
        "l2", _dtype, ndims, npts, build_complexity, graph_degree
    )

    tags = np.arange(1, npts+1, dtype=np.uintc)
    timer = Timer()

    with timer.time("batch insert"):
        index.batch_insert(vectors_to_index, tags, num_insert_threads)

    delete_tags = np.random.choice(
        np.array(range(1, npts + 1, 1), dtype=np.uintc),
        size=int(0.5 * npts),
        replace=False
    )
    with timer.time("mark deletion"):
        for tag in delete_tags:
            index.mark_deleted(tag)

    with timer.time("consolidation"):
        index.consolidate_delete()

    deleted_data = vectors_to_index[delete_tags - 1, :]

    with timer.time("re-insertion"):
        index.batch_insert(deleted_data, delete_tags, num_insert_threads)

    with timer.time("batch searched"):
        tags, dists = index.batch_search(vectors_to_query, K, search_complexity, num_search_threads)

    # res_ids = tags - 1
    # if gt_file != "":
    #     recall = utils.calculate_recall_from_gt_file(K, res_ids, gt_file)
    #     print(f"recall@{K} is {recall}")

def static(
    dtype: str,
    index_directory: str,
    index_vectors_file: str,
    query_vectors_file: str,
    build_complexity: int,
    graph_degree: int,
    K: int,
    search_complexity: int,
    num_threads: int,
    gt_file: str = "",
    index_prefix: str = "ann"
):
    _dtype, vectors_to_query = _basic_setup(dtype, query_vectors_file)
    timer = Timer()
    with timer.time("build static index"):
        # build index
        dap.build_memory_index(
            data=index_vectors_file,
            metric="l2",
            vector_dtype=_dtype,
            index_directory=index_directory,
            complexity=build_complexity,
            graph_degree=graph_degree,
            num_threads=num_threads,
            index_prefix=index_prefix,
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
        )

    with timer.time("load static index"):
        # ready search object
        index = dap.StaticMemoryIndex(
            metric="l2",
            vector_dtype=_dtype,
            data_path=index_vectors_file,
            index_directory=index_directory,
            num_threads=num_threads,  # this can be different at search time if you would like
            initial_search_complexity=search_complexity,
            index_prefix=index_prefix
    )

    ids, dists = index.batch_search(vectors_to_query, K, search_complexity, num_threads)

    # if gt_file != "":
    #     recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
    #     print(f"recall@{K} is {recall}")

def dynamic_clustered():
    pass

def generate_clusters():
    pass


class Timer:
    def __init__(self):
        self._start = -1

    @contextmanager
    def time(self, message: str):
        start = perf_counter()
        if self._start == -1:
            self._start = start
        yield
        now = perf_counter()
        print(f"Operation {message} completed in {(now - start):.3f}s, total: {(now - self._start):.3f}s")




if __name__ == "__main__":
    fire.Fire({
        "in-mem-dynamic": dynamic,
        "in-mem-static": static,
        "in-mem-dynamic-clustered": dynamic_clustered,
        "generate-clusters": generate_clusters
    }, name="cli")
