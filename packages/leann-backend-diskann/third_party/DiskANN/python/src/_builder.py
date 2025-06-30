# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from . import DistanceMetric, VectorDType, VectorIdentifierBatch, VectorLikeBatch
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _valid_metric,
    _write_index_metadata,
    valid_dtype,
)
from ._diskannpy import defaults
from ._files import tags_to_file, vectors_metadata_from_file, vectors_to_file


def _valid_path_and_dtype(
    data: Union[str, VectorLikeBatch],
    vector_dtype: VectorDType,
    index_path: str,
    index_prefix: str,
) -> Tuple[str, VectorDType]:
    if isinstance(data, str):
        vector_bin_path = data
        _assert(
            Path(data).exists() and Path(data).is_file(),
            "if data is of type `str`, it must both exist and be a file",
        )
        vector_dtype_actual = valid_dtype(vector_dtype)
    else:
        vector_bin_path = os.path.join(index_path, f"{index_prefix}_vectors.bin")
        # if Path(vector_bin_path).exists():
        #     raise ValueError(
        #         f"The path {vector_bin_path} already exists. Remove it and try again."
        #     )
        vector_dtype_actual = valid_dtype(data.dtype)
        # vectors_to_file(vector_file=vector_bin_path, vectors=data)

    return vector_bin_path, vector_dtype_actual


def build_disk_index(
    data: Union[str, VectorLikeBatch],
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int = defaults.PQ_DISK_BYTES,
    vector_dtype: Optional[VectorDType] = None,
    index_prefix: str = "ann",
    codebook_prefix: str = "",
) -> None:
    """
    This function will construct a DiskANN disk index. Disk indices are ideal for very large datasets that
    are too large to fit in memory. Memory is still used, but it is primarily used to provide precise disk
    locations for fast retrieval of smaller subsets of the index without compromising much on recall.

    If you provide a numpy array, it will save this array to disk in a temp location
    in the format DiskANN's PQ Flash Index builder requires. This temp folder is deleted upon index creation completion
    or error.

    ## Distance Metric and Vector Datatype Restrictions
    | Metric \ Datatype | np.float32 | np.uint8 | np.int8 |
    |-------------------|------------|----------|---------|
    | L2                |      ✅     |     ✅    |    ✅    |
    | MIPS              |      ✅     |     ❌    |    ❌    |
    | Cosine [^bug-in-disk-cosine]     |      ❌     |     ❌    |    ❌    |

    [^bug-in-disk-cosine]: For StaticDiskIndex, Cosine distances are not currently supported.

    ### Parameters
    - **data**: Either a `str` representing a path to a DiskANN vector bin file, or a numpy.ndarray,
      of a supported dtype, in 2 dimensions. Note that `vector_dtype` must be provided if data is a `str`
    - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
      vector dtypes, but `mips` is only available for single precision floats.
    - **index_directory**: The index files will be saved to this **existing** directory path
    - **complexity**: The size of the candidate nearest neighbor list to use when building the index. Values between 75
      and 200 are typical. Larger values will take more time to build but result in indices that provide higher recall
      for the same search complexity. Use a value that is at least as large as `graph_degree` unless you are prepared
      to compromise on quality
    - **graph_degree**: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
      result in larger indices and longer indexing times, but better search quality.
    - **search_memory_maximum**: Build index with the expectation that the search will use at most
      `search_memory_maximum`, in gb.
    - **build_memory_maximum**: Build index using at most `build_memory_maximum` in gb. Building processes typically
      require more memory, while search memory can be reduced.
    - **num_threads**: Number of threads to use when creating this index. `0` is used to indicate all available
      logical processors should be used.
    - **pq_disk_bytes**: Use `0` to store uncompressed data on SSD. This allows the index to asymptote to 100%
      recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the
      vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater
      than the number of bytes used for the PQ compressed data stored in-memory. Default is `0`.
    - **vector_dtype**: Required if the provided `data` is of type `str`, else we use the `data.dtype` if np array.
    - **index_prefix**: The prefix of the index files. Defaults to "ann".
    """

    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _valid_metric(distance_metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(search_memory_maximum > 0, "search_memory_maximum must be larger than 0")
    _assert(build_memory_maximum > 0, "build_memory_maximum must be larger than 0")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(pq_disk_bytes, "pq_disk_bytes")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory, index_prefix
    )
    _assert(dap_metric != _native_dap.COSINE, "Cosine is currently not supported in StaticDiskIndex")
    if dap_metric == _native_dap.INNER_PRODUCT:
        _assert(
            vector_dtype_actual == np.float32,
            "Integral vector dtypes (np.uint8, np.int8) are not supported with distance metric mips"
        )

    num_points, dimensions = vectors_metadata_from_file(vector_bin_path)

    if vector_dtype_actual == np.uint8:
        _builder = _native_dap.build_disk_uint8_index
    elif vector_dtype_actual == np.int8:
        _builder = _native_dap.build_disk_int8_index
    else:
        _builder = _native_dap.build_disk_float_index

    index_prefix_path = os.path.join(index_directory, index_prefix)

    _builder(
        distance_metric=dap_metric,
        data_file_path=vector_bin_path,
        index_prefix_path=index_prefix_path,
        complexity=complexity,
        graph_degree=graph_degree,
        final_index_ram_limit=search_memory_maximum,
        indexing_ram_budget=build_memory_maximum,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
        codebook_prefix=codebook_prefix,
    )
    _write_index_metadata(
        index_prefix_path, vector_dtype_actual, dap_metric, num_points, dimensions
    )


def build_memory_index(
    data: Union[str, VectorLikeBatch],
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    num_threads: int,
    alpha: float = defaults.ALPHA,
    use_pq_build: bool = defaults.USE_PQ_BUILD,
    num_pq_bytes: int = defaults.NUM_PQ_BYTES,
    use_opq: bool = defaults.USE_OPQ,
    vector_dtype: Optional[VectorDType] = None,
    tags: Union[str, VectorIdentifierBatch] = "",
    filter_labels: Optional[list[list[str]]] = None,
    universal_label: str = "",
    filter_complexity: int = defaults.FILTER_COMPLEXITY,
    index_prefix: str = "ann",
) -> None:
    """
    This function will construct a DiskANN memory index. Memory indices are ideal for smaller datasets whose
    indices can fit into memory. Memory indices are faster than disk indices, but usually cannot scale to massive
    sizes in an individual index on an individual machine.

    `diskannpy`'s memory indices take two forms: a `diskannpy.StaticMemoryIndex`, which will not be mutated, only
    searched upon, and a `diskannpy.DynamicMemoryIndex`, which can be mutated AND searched upon in the same process.

    ## Important Note:
    You **must** determine the type of index you are building for. If you are building for a
    `diskannpy.DynamicMemoryIndex`, you **must** supply a valid value for the `tags` parameter. **Do not supply
    tags if the index is intended to be `diskannpy.StaticMemoryIndex`**!

    ## Distance Metric and Vector Datatype Restrictions

    | Metric \ Datatype | np.float32 | np.uint8 | np.int8 |
    |-------------------|------------|----------|---------|
    | L2                |      ✅     |     ✅    |    ✅    |
    | MIPS              |      ✅     |     ❌    |    ❌    |
    | Cosine            |      ✅     |     ✅    |    ✅    |

    ### Parameters

    - **data**: Either a `str` representing a path to an existing DiskANN vector bin file, or a numpy.ndarray of a
      supported dtype in 2 dimensions. Note that `vector_dtype` must be provided if `data` is a `str`.
    - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
      vector dtypes, but `mips` is only available for single precision floats.
    - **index_directory**: The index files will be saved to this **existing** directory path
    - **complexity**: The size of the candidate nearest neighbor list to use when building the index. Values between 75
      and 200 are typical. Larger values will take more time to build but result in indices that provide higher recall
      for the same search complexity. Use a value that is at least as large as `graph_degree` unless you are prepared
      to compromise on quality
    - **graph_degree**: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
      result in larger indices and longer indexing times, but better search quality.
    - **num_threads**: Number of threads to use when creating this index. `0` is used to indicate all available
      logical processors should be used.
    - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
      graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably more
      distance comparisons compared to a lower alpha value.
    - **use_pq_build**: Use product quantization during build. Product quantization is a lossy compression technique
      that can reduce the size of the index on disk. This will trade off recall. Default is `True`.
    - **num_pq_bytes**: The number of bytes used to store the PQ compressed data in memory. This will trade off recall.
      Default is `0`.
    - **use_opq**: Use optimized product quantization during build.
    - **vector_dtype**: Required if the provided `data` is of type `str`, else we use the `data.dtype` if np array.
    - **tags**: Tags can be defined either as a path on disk to an existing .tags file, or provided as a np.array of
      the same length as the number of vectors. Tags are used to identify vectors in the index via your *own*
      numbering conventions, and is absolutely required for loading DynamicMemoryIndex indices `from_file`.
    - **filter_labels**: An optional, but exhaustive list of categories for each vector. This is used to filter
      search results by category. If provided, this must be a list of lists, where each inner list is a list of
      categories for the corresponding vector. For example, if you have 3 vectors, and the first vector belongs to
      categories "a" and "b", the second vector belongs to category "b", and the third vector belongs to no categories,
      you would provide `filter_labels=[["a", "b"], ["b"], []]`. If you do not want to provide categories for a
      particular vector, you can provide an empty list. If you do not want to provide categories for any vectors,
      you can provide `None` for this parameter (which is the default)
    - **universal_label**: An optional label that indicates that this vector should be included in *every* search
      in which it also meets the knn search criteria.
    - **filter_complexity**: Complexity to use when using filters. Default is 0. 0 is strictly invalid if you are
      using filters.
    - **index_prefix**: The prefix of the index files. Defaults to "ann".
    """
    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _valid_metric(distance_metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(
        alpha >= 1,
        "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)",
    )
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(num_pq_bytes, "num_pq_bytes")
    _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")
    _assert(
        filter_labels is None or filter_complexity > 0,
        "if filter_labels is provided, filter_complexity must not be 0"
    )

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory, index_prefix
    )
    if dap_metric == _native_dap.INNER_PRODUCT:
        _assert(
            vector_dtype_actual == np.float32,
            "Integral vector dtypes (np.uint8, np.int8) are not supported with distance metric mips"
        )

    num_points, dimensions = vectors_metadata_from_file(vector_bin_path)
    if filter_labels is not None:
        _assert(
            len(filter_labels) == num_points,
            "filter_labels must be the same length as the number of points"
        )

    if vector_dtype_actual == np.uint8:
        _builder = _native_dap.build_memory_uint8_index
    elif vector_dtype_actual == np.int8:
        _builder = _native_dap.build_memory_int8_index
    else:
        _builder = _native_dap.build_memory_float_index

    index_prefix_path = os.path.join(index_directory, index_prefix)

    filter_labels_file = ""
    if filter_labels is not None:
        label_counts = {}
        filter_labels_file = f"{index_prefix_path}_pylabels.txt"
        with open(filter_labels_file, "w") as labels_file:
            for labels in filter_labels:
                for label in labels:
                    label_counts[label] = 1 if label not in label_counts else label_counts[label] + 1
                if len(labels) == 0:
                    print("default", file=labels_file)
                else:
                    print(",".join(labels), file=labels_file)
        with open(f"{index_prefix_path}_label_metadata.json", "w") as label_metadata_file:
            json.dump(label_counts, label_metadata_file, indent=True)

    if isinstance(tags, str) and tags != "":
        use_tags = True
        shutil.copy(tags, index_prefix_path + ".tags")
    elif not isinstance(tags, str):
        use_tags = True
        tags_as_array = _castable_dtype_or_raise(tags, expected=np.uint32)
        _assert(len(tags_as_array.shape) == 1, "Provided tags must be 1 dimensional")
        _assert(
            tags_as_array.shape[0] == num_points,
            "Provided tags must contain an identical population to the number of points, "
            f"{tags_as_array.shape[0]=}, {num_points=}",
        )
        tags_to_file(index_prefix_path + ".tags", tags_as_array)
    else:
        use_tags = False

    _builder(
        distance_metric=dap_metric,
        data_file_path=vector_bin_path,
        index_output_path=index_prefix_path,
        complexity=complexity,
        graph_degree=graph_degree,
        alpha=alpha,
        num_threads=num_threads,
        use_pq_build=use_pq_build,
        num_pq_bytes=num_pq_bytes,
        use_opq=use_opq,
        use_tags=use_tags,
        filter_labels_file=filter_labels_file,
        universal_label=universal_label,
        filter_complexity=filter_complexity,
    )

    _write_index_metadata(
        index_prefix_path, vector_dtype_actual, dap_metric, num_points, dimensions
    )
