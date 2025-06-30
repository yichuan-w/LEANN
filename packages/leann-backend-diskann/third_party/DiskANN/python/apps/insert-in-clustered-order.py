# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse

import diskannpy
import numpy as np
import utils


def insert_and_search(
    dtype_str,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    num_clusters,
    num_insert_threads,
    K,
    Ls,
    num_search_threads,
    gt_file,
):
    npts, ndims = utils.get_bin_metadata(indexdata_file)

    if dtype_str == "float":
        dtype = np.float32
    elif dtype_str == "int8":
        dtype = np.int8
    elif dtype_str == "uint8":
        dtype = np.uint8
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    index = diskannpy.DynamicMemoryIndex(
        distance_metric="l2",
        vector_dtype=dtype,
        dimensions=ndims,
        max_vectors=npts,
        complexity=Lb,
        graph_degree=graph_degree
    )
    queries = diskannpy.vectors_from_file(querydata_file, dtype)
    data = diskannpy.vectors_from_file(indexdata_file, dtype)

    offsets, permutation = utils.cluster_and_permute(
        dtype_str, npts, ndims, data, num_clusters
    )

    i = 0
    timer = utils.Timer()
    for c in range(num_clusters):
        cluster_index_range = range(offsets[c], offsets[c + 1])
        cluster_indices = np.array(permutation[cluster_index_range], dtype=np.uint32)
        cluster_data = data[cluster_indices, :]
        index.batch_insert(cluster_data, cluster_indices + 1, num_insert_threads)
        print('Inserted cluster', c, 'in', timer.elapsed(), 's')
    tags, dists = index.batch_search(queries, K, Ls, num_search_threads)
    print('Batch searched', queries.shape[0], 'queries in', timer.elapsed(), 's')
    res_ids = tags - 1

    if gt_file != "":
        recall = utils.calculate_recall_from_gt_file(K, res_ids, gt_file)
        print(f"recall@{K} is {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="in-mem-dynamic",
        description="Inserts points dynamically in a clustered order and search from vectors in a file.",
    )

    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-q", "--querydata_file", required=True)
    parser.add_argument("-Lb", "--Lbuild", default=50, type=int)
    parser.add_argument("-Ls", "--Lsearch", default=50, type=int)
    parser.add_argument("-R", "--graph_degree", default=32, type=int)
    parser.add_argument("-TI", "--num_insert_threads", default=8, type=int)
    parser.add_argument("-TS", "--num_search_threads", default=8, type=int)
    parser.add_argument("-C", "--num_clusters", default=32, type=int)
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("--gt_file", default="")
    args = parser.parse_args()

    insert_and_search(
        args.data_type,
        args.indexdata_file,
        args.querydata_file,
        args.Lbuild,
        args.graph_degree,  # Build args
        args.num_clusters,
        args.num_insert_threads,
        args.K,
        args.Lsearch,
        args.num_search_threads,  # search args
        args.gt_file,
    )

# An ingest optimized example with SIFT1M
# python3 ~/DiskANN/python/apps/insert-in-clustered-order.py -d float \
# -i sift_base.fbin -q sift_query.fbin --gt_file  gt100_base \
# -Lb 10 -R 30 -Ls 200 -C 32