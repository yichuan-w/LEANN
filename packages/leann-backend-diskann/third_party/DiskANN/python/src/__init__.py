# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
# Documentation Overview
`diskannpy` is mostly structured around 2 distinct processes: [Index Builder Functions](#index-builders) and [Search Classes](#search-classes)

It also includes a few nascent [utilities](#utilities).

And lastly, it makes substantial use of type hints, with various shorthand [type aliases](#parameter-and-response-type-aliases) documented. 
When reading the `diskannpy` code we refer to the type aliases, though `pdoc` helpfully expands them.

## Index Builders
- `build_disk_index` - To build an index that cannot fully fit into memory when searching
- `build_memory_index` - To build an index that can fully fit into memory when searching

## Search Classes
- `StaticMemoryIndex` - for indices that can fully fit in memory and won't be changed during the search operations
- `StaticDiskIndex` - for indices that cannot fully fit in memory, thus relying on disk IO to search, and also won't be changed during search operations
- `DynamicMemoryIndex` - for indices that can fully fit in memory and will be mutated via insert/deletion operations as well as search operations

## Parameter Defaults
- `diskannpy.defaults` - Default values exported from the C++ extension for Python users

## Parameter and Response Type Aliases
- `DistanceMetric` - What distance metrics does `diskannpy` support?
- `VectorDType` - What vector datatypes does `diskannpy` support?
- `QueryResponse` - What can I expect as a response to my search?
- `QueryResponseBatch` - What can I expect as a response to my batch search?
- `VectorIdentifier` - What types do `diskannpy` support as vector identifiers?
- `VectorIdentifierBatch` - A batch of identifiers of the exact same type. The type can change, but they must **all** change.
- `VectorLike` - How does a vector look to `diskannpy`, to be inserted or searched with.
- `VectorLikeBatch` - A batch of those vectors, to be inserted or searched with.
- `Metadata` - DiskANN vector binary file metadata (num_points, vector_dim)

## Utilities
- `vectors_to_file` - Turns a 2 dimensional `numpy.typing.NDArray[VectorDType]` with shape `(number_of_points, vector_dim)` into a DiskANN vector bin file.
- `vectors_from_file` - Reads a DiskANN vector bin file representing stored vectors into a numpy ndarray.
- `vectors_metadata_from_file` - Reads metadata stored in a DiskANN vector bin file without reading the entire file
- `tags_to_file` - Turns a 1 dimensional `numpy.typing.NDArray[VectorIdentifier]` into a DiskANN tags bin file.
- `tags_from_file` - Reads a DiskANN tags bin file representing stored tags into a numpy ndarray.
- `valid_dtype` - Checks if a given vector dtype is supported by `diskannpy`
"""

from typing import Any, Literal, NamedTuple, Type, Union

import numpy as np
from numpy import typing as npt

DistanceMetric = Literal["l2", "mips", "cosine"]
""" Type alias for one of {"l2", "mips", "cosine"} """
VectorDType = Union[Type[np.float32], Type[np.int8], Type[np.uint8]]
""" Type alias for one of {`numpy.float32`, `numpy.int8`, `numpy.uint8`} """
VectorLike = npt.NDArray[VectorDType]
""" Type alias for something that can be treated as a vector """
VectorLikeBatch = npt.NDArray[VectorDType]
""" Type alias for a batch of VectorLikes """
VectorIdentifier = np.uint32
""" 
Type alias for a vector identifier, whether it be an implicit array index identifier from StaticMemoryIndex or 
StaticDiskIndex, or an explicit tag identifier from DynamicMemoryIndex 
"""
VectorIdentifierBatch = npt.NDArray[np.uint32]
""" Type alias for a batch of VectorIdentifiers """


class QueryResponse(NamedTuple):
    """
    Tuple with two values, identifiers and distances. Both are 1d arrays, positionally correspond, and will contain the
    nearest neighbors from [0..k_neighbors)
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ A `numpy.typing.NDArray[VectorIdentifier]` array of vector identifiers, 1 dimensional """
    distances: npt.NDArray[np.float32]
    """
    A `numpy.typing.NDAarray[numpy.float32]` of distances as calculated by the distance metric function,  1 dimensional
    """


class QueryResponseBatch(NamedTuple):
    """
    Tuple with two values, identifiers and distances. Both are 2d arrays, with dimensionality determined by the
    rows corresponding to the number of queries made, and the columns corresponding to the k neighbors
    requested. The two 2d arrays have an implicit, position-based relationship
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ 
    A `numpy.typing.NDArray[VectorIdentifier]` array of vector identifiers, 2 dimensional. The row corresponds to index 
    of the query, and the column corresponds to the k neighbors requested 
    """
    distances: np.ndarray[np.float32]
    """  
    A `numpy.typing.NDAarray[numpy.float32]` of distances as calculated by the distance metric function, 2 dimensional. 
    The row corresponds to the index of the query, and the column corresponds to the distance of the query to the 
    *k-th* neighbor 
    """


from . import defaults
from ._builder import build_disk_index, build_memory_index
from ._common import valid_dtype
from ._dynamic_memory_index import DynamicMemoryIndex
from ._files import (
    Metadata,
    tags_from_file,
    tags_to_file,
    vectors_from_file,
    vectors_metadata_from_file,
    vectors_to_file,
)
from ._static_disk_index import StaticDiskIndex
from ._static_memory_index import StaticMemoryIndex

__all__ = [
    "build_disk_index",
    "build_memory_index",
    "StaticDiskIndex",
    "StaticMemoryIndex",
    "DynamicMemoryIndex",
    "defaults",
    "DistanceMetric",
    "VectorDType",
    "QueryResponse",
    "QueryResponseBatch",
    "VectorIdentifier",
    "VectorIdentifierBatch",
    "VectorLike",
    "VectorLikeBatch",
    "Metadata",
    "vectors_metadata_from_file",
    "vectors_to_file",
    "vectors_from_file",
    "tags_to_file",
    "tags_from_file",
    "valid_dtype",
]
