# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import warnings
from typing import BinaryIO, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from . import VectorDType, VectorIdentifierBatch, VectorLikeBatch
from ._common import _assert, _assert_2d, _assert_dtype, _assert_existing_file


class Metadata(NamedTuple):
    """DiskANN binary vector files contain a small stanza containing some metadata about them."""

    num_vectors: int
    """ The number of vectors in the file. """
    dimensions: int
    """ The dimensionality of the vectors in the file. """


def vectors_metadata_from_file(vector_file: str) -> Metadata:
    """
    Read the metadata from a DiskANN binary vector file.
    ### Parameters
    - **vector_file**: The path to the vector file to read the metadata from.

    ### Returns
    `diskannpy.Metadata`
    """
    _assert_existing_file(vector_file, "vector_file")
    points, dims = np.fromfile(file=vector_file, dtype=np.int32, count=2)
    return Metadata(points, dims)


def _write_bin(data: np.ndarray, file_handler: BinaryIO):
    if len(data.shape) == 1:
        _ = file_handler.write(np.array([data.shape[0], 1], dtype=np.int32).tobytes())
    else:
        _ = file_handler.write(np.array(data.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(data.tobytes())


def vectors_to_file(vector_file: str, vectors: VectorLikeBatch) -> None:
    """
    Utility function that writes a DiskANN binary vector formatted file to the location of your choosing.

    ### Parameters
    - **vector_file**: The path to the vector file to write the vectors to.
    - **vectors**: A 2d array of dtype `numpy.float32`, `numpy.uint8`, or `numpy.int8`
    """
    _assert_dtype(vectors.dtype)
    _assert_2d(vectors, "vectors")
    with open(vector_file, "wb") as fh:
        _write_bin(vectors, fh)


def vectors_from_file(
    vector_file: str,
    dtype: VectorDType,
    use_memmap: bool = False,
    mode: Literal["r", "r+"] = "r"
) -> npt.NDArray[VectorDType]:
    """
    Read vectors from a DiskANN binary vector file.

    ### Parameters
    - **vector_file**: The path to the vector file to read the vectors from.
    - **dtype**: The data type of the vectors in the file. Ensure you match the data types exactly
    - **use_memmap**: If True, return a np.memmap, else a standard np.ndarray will be returned
    - **mode**: Read-only (r) or read-write (r+) (memmap only). Unlike np.memmap, default is read-only (r)

    ### Returns
    `numpy.typing.NDArray[dtype] | numpy.memmap`
    """
    assert mode in ["r", "r+"]
    points, dims = vectors_metadata_from_file(vector_file)
    if not use_memmap:
        return np.fromfile(file=vector_file, dtype=dtype, offset=8).reshape(points, dims)
    else:
        return np.memmap(vector_file, dtype=dtype, mode=mode, offset=8, shape=(points, dims), order='C')


def tags_to_file(tags_file: str, tags: VectorIdentifierBatch) -> None:
    """
    Write tags to a DiskANN binary tag file.

    ### Parameters
    - **tags_file**: The path to the tag file to write the tags to.
    - **tags**: A 1d array of dtype `numpy.uint32` containing the tags to write. If you have a 2d array of tags with
      one column, you can pass it here and it will be reshaped and copied to a new array. It is more efficient for you
      to reshape on your own without copying it first, as it should be a constant time operation vs. linear time

    """
    _assert(np.can_cast(tags.dtype, np.uint32), "valid tags must be uint32")
    _assert(
        len(tags.shape) == 1 or tags.shape[1] == 1,
        "tags must be 1d or 2d with 1 column",
    )
    if len(tags.shape) == 2:
        warnings.warn(
            "Tags in 2d with one column will be reshaped and copied to a new array. "
            "It is more efficient for you to reshape without copying first."
        )
        tags = tags.reshape(tags.shape[0], copy=True)
    with open(tags_file, "wb") as fh:
        _write_bin(tags.astype(np.uint32), fh)


def tags_from_file(tags_file: str) -> VectorIdentifierBatch:
    """
    Read tags from a DiskANN binary tag file and return them as a 1d array of dtype `numpy.uint32`.

    ### Parameters
    - **tags_file**: The path to the tag file to read the tags from.
    """
    _assert_existing_file(tags_file, "tags_file")
    points, dims = vectors_metadata_from_file(
        tags_file
    )  # tag files contain the same metadata stanza
    return np.fromfile(file=tags_file, dtype=np.uint32, offset=8).reshape(points)
