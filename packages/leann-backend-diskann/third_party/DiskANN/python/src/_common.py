# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Tuple, Type, Union

import numpy as np

from . import (
    DistanceMetric,
    VectorDType,
    VectorIdentifierBatch,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap

__ALL__ = ["valid_dtype"]

_VALID_DTYPES = [np.float32, np.int8, np.uint8]


def valid_dtype(dtype: Type) -> VectorDType:
    """
    Utility method to determine whether the provided dtype is supported by `diskannpy`, and if so, the canonical
    dtype we will use internally (e.g. np.single -> np.float32)
    """
    _assert_dtype(dtype)
    if dtype == np.uint8:
        return np.uint8
    if dtype == np.int8:
        return np.int8
    if dtype == np.float32:
        return np.float32


def _assert(statement_eval: bool, message: str):
    if not statement_eval:
        raise ValueError(message)


def _valid_metric(metric: str) -> _native_dap.Metric:
    if not isinstance(metric, str):
        raise ValueError("distance_metric must be a string")
    if metric.lower() == "l2":
        return _native_dap.L2
    elif metric.lower() == "mips":
        return _native_dap.INNER_PRODUCT
    elif metric.lower() == "cosine":
        return _native_dap.COSINE
    else:
        raise ValueError("distance_metric must be one of 'l2', 'mips', or 'cosine'")


def _assert_dtype(dtype: Type):
    _assert(
        any(np.can_cast(dtype, _dtype) for _dtype in _VALID_DTYPES),
        f"Vector dtype must be of one of type {{(np.single, np.float32), (np.byte, np.int8), (np.ubyte, np.uint8)}}",
    )


def _castable_dtype_or_raise(
    data: Union[VectorLike, VectorLikeBatch, VectorIdentifierBatch], expected: np.dtype
) -> np.ndarray:
    if isinstance(data, np.ndarray) and np.can_cast(data.dtype, expected):
        return data.astype(expected, casting="safe")
    else:
        raise TypeError(
            f"expecting a numpy ndarray of dtype {expected}, not a {type(data)}"
        )


def _assert_2d(vectors: np.ndarray, name: str):
    _assert(len(vectors.shape) == 2, f"{name} must be 2d numpy array")


__MAX_UINT32_VAL = 4_294_967_295


def _assert_is_positive_uint32(test_value: int, parameter: str):
    _assert(
        test_value is not None and 0 < test_value < __MAX_UINT32_VAL,
        f"{parameter} must be a positive integer in the uint32 range",
    )


def _assert_is_nonnegative_uint32(test_value: int, parameter: str):
    _assert(
        test_value is not None and -1 < test_value < __MAX_UINT32_VAL,
        f"{parameter} must be a non-negative integer in the uint32 range",
    )


def _assert_is_nonnegative_uint64(test_value: int, parameter: str):
    _assert(
        -1 < test_value,
        f"{parameter} must be a non-negative integer in the uint64 range",
    )


def _assert_existing_directory(path: str, parameter: str):
    _path = Path(path)
    _assert(
        _path.exists() and _path.is_dir(), f"{parameter} must be an existing directory"
    )


def _assert_existing_file(path: str, parameter: str):
    _path = Path(path)
    _assert(_path.exists() and _path.is_file(), f"{parameter} must be an existing file")


class _DataType(Enum):
    FLOAT32 = 0
    INT8 = 1
    UINT8 = 2

    @classmethod
    def from_type(cls, vector_dtype: VectorDType) -> "DataType":
        if vector_dtype == np.float32:
            return cls.FLOAT32
        if vector_dtype == np.int8:
            return cls.INT8
        if vector_dtype == np.uint8:
            return cls.UINT8

    def to_type(self) -> VectorDType:
        if self is _DataType.FLOAT32:
            return np.float32
        if self is _DataType.INT8:
            return np.int8
        if self is _DataType.UINT8:
            return np.uint8


class _Metric(Enum):
    L2 = 0
    MIPS = 1
    COSINE = 2

    @classmethod
    def from_native(cls, metric: _native_dap.Metric) -> "_Metric":
        if metric == _native_dap.L2:
            return cls.L2
        if metric == _native_dap.INNER_PRODUCT:
            return cls.MIPS
        if metric == _native_dap.COSINE:
            return cls.COSINE

    def to_native(self) -> _native_dap.Metric:
        if self is _Metric.L2:
            return _native_dap.L2
        if self is _Metric.MIPS:
            return _native_dap.INNER_PRODUCT
        if self is _Metric.COSINE:
            return _native_dap.COSINE

    def to_str(self) -> _native_dap.Metric:
        if self is _Metric.L2:
            return "l2"
        if self is _Metric.MIPS:
            return "mips"
        if self is _Metric.COSINE:
            return "cosine"


def _build_metadata_path(index_path_and_prefix: str) -> str:
    return index_path_and_prefix + "_metadata.bin"


def _write_index_metadata(
    index_path_and_prefix: str,
    dtype: VectorDType,
    metric: _native_dap.Metric,
    num_points: int,
    dimensions: int,
):
    np.array(
        [
            _DataType.from_type(dtype).value,
            _Metric.from_native(metric).value,
            num_points,
            dimensions,
        ],
        dtype=np.uint64,
    ).tofile(_build_metadata_path(index_path_and_prefix))


def _read_index_metadata(
    index_path_and_prefix: str,
) -> Optional[Tuple[VectorDType, str, np.uint64, np.uint64]]:
    path = _build_metadata_path(index_path_and_prefix)
    if not Path(path).exists():
        return None
    else:
        metadata = np.fromfile(path, dtype=np.uint64, count=-1)
        return (
            _DataType(int(metadata[0])).to_type(),
            _Metric(int(metadata[1])).to_str(),
            metadata[2],
            metadata[3],
        )


def _ensure_index_metadata(
    index_path_and_prefix: str,
    vector_dtype: Optional[VectorDType],
    distance_metric: Optional[DistanceMetric],
    max_vectors: int,
    dimensions: Optional[int],
    warn_size_exceeded: bool = False,
) -> Tuple[VectorDType, str, np.uint64, np.uint64]:
    possible_metadata = _read_index_metadata(index_path_and_prefix)
    if possible_metadata is None:
        _assert(
            all([vector_dtype, distance_metric, dimensions]),
            "distance_metric, vector_dtype, and dimensions must provided if a corresponding metadata file has not "
            "been built for this index, such as when an index was built via the CLI tools or prior to the addition "
            "of a metadata file",
        )
        _assert_dtype(vector_dtype)
        _assert_is_positive_uint32(max_vectors, "max_vectors")
        _assert_is_positive_uint32(dimensions, "dimensions")
        return vector_dtype, distance_metric, max_vectors, dimensions  # type: ignore
    else:
        vector_dtype, distance_metric, num_vectors, dimensions = possible_metadata
        if warn_size_exceeded:
            if max_vectors is not None and num_vectors > max_vectors:
                warnings.warn(
                    "The number of vectors in the saved index exceeds the max_vectors parameter. "
                    "max_vectors is being adjusted to accommodate the dataset, but any insertions will fail."
                )
                max_vectors = num_vectors
            if num_vectors == max_vectors:
                warnings.warn(
                    "The number of vectors in the saved index equals max_vectors parameter. Any insertions will fail."
                )
        return possible_metadata


def _valid_index_prefix(index_directory: str, index_prefix: str) -> str:
    _assert(
        index_directory is not None and index_directory != "",
        "index_directory cannot be None or empty",
    )
    _assert_existing_directory(index_directory, "index_directory")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")
    return os.path.join(index_directory, index_prefix)
