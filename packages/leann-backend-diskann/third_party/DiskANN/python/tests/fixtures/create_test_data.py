# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import numpy as np


def random_vectors(rows: int, dimensions: int, dtype, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == np.float32:
        vectors = rng.random((rows, dimensions), dtype=dtype)
    elif dtype == np.uint8:
        vectors = rng.integers(
            low=0, high=256, size=(rows, dimensions), dtype=dtype
        )  # low is inclusive, high is exclusive
    elif dtype == np.int8:
        vectors = rng.integers(
            low=-128, high=128, size=(rows, dimensions), dtype=dtype
        )  # low is inclusive, high is exclusive
    else:
        raise RuntimeError("Only np.float32, np.int8, and np.uint8 are supported")
    return vectors


def write_vectors(file_handler: BinaryIO, vectors: np.ndarray):
    _ = file_handler.write(np.array(vectors.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(vectors.tobytes())


@contextmanager
def vectors_as_temp_file(vectors: np.ndarray) -> str:
    temp = NamedTemporaryFile(mode="wb", delete=False)
    write_vectors(temp, vectors)
    temp.close()
    yield temp.name
    Path(temp.name).unlink()
