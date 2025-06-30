# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np

from .create_test_data import random_vectors


def build_random_vectors_and_memory_index(
    dtype, metric, with_tags: bool = False, index_prefix: str = "ann", seed: int = 12345
):
    query_vectors: np.ndarray = random_vectors(1000, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(10000, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    if with_tags:
        rng = np.random.default_rng(seed)
        tags = np.arange(start=1, stop=10001, dtype=np.uint32)
        rng.shuffle(tags)
    else:
        tags = ""

    dap.build_memory_index(
        data=index_vectors,
        distance_metric=metric,
        index_directory=ann_dir,
        graph_degree=16,
        complexity=32,
        alpha=1.2,
        num_threads=0,
        use_pq_build=False,
        num_pq_bytes=8,
        use_opq=False,
        filter_complexity=32,
        tags=tags,
        index_prefix=index_prefix,
    )

    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        os.path.join(ann_dir, "vectors.bin"),
        tags,
    )
