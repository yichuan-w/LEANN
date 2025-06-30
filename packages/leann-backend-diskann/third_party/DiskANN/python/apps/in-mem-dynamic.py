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
    K,
    Ls,
    num_insert_threads,
    num_search_threads,
    gt_file,
) -> dict[str, float]:
    """

    :param dtype_str:
    :param indexdata_file:
    :param querydata_file:
    :param Lb:
    :param graph_degree:
    :param K:
    :param Ls:
    :param num_insert_threads:
    :param num_search_threads:
    :param gt_file:
    :return: Dictionary of timings.  Key is the event and value is the number of seconds the event took
    """
    timer_results: dict[str, float] = {}

    method_timer: utils.Timer = utils.Timer()

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

    tags = np.zeros(npts, dtype=np.uintc)
    timer = utils.Timer()
    for i in range(npts):
        tags[i] = i + 1
    index.batch_insert(data, tags, num_insert_threads)
    compute_seconds = timer.elapsed()
    print('batch_insert complete in', compute_seconds, 's')
    timer_results["batch_insert_seconds"] = compute_seconds

    delete_tags = np.random.choice(
        np.array(range(1, npts + 1, 1), dtype=np.uintc),
        size=int(0.5 * npts),
        replace=False
    )

    timer.reset()
    for tag in delete_tags:
        index.mark_deleted(tag)
    compute_seconds = timer.elapsed()
    timer_results['mark_deletion_seconds'] = compute_seconds
    print('mark deletion completed in', compute_seconds, 's')

    timer.reset()
    index.consolidate_delete()
    compute_seconds = timer.elapsed()
    print('consolidation completed in', compute_seconds, 's')
    timer_results['consolidation_completed_seconds'] = compute_seconds

    deleted_data = data[delete_tags - 1, :]

    timer.reset()
    index.batch_insert(deleted_data, delete_tags, num_insert_threads)
    compute_seconds = timer.elapsed()
    print('re-insertion completed in', compute_seconds, 's')
    timer_results['re-insertion_seconds'] = compute_seconds

    timer.reset()
    tags, dists = index.batch_search(queries, K, Ls, num_search_threads)
    compute_seconds = timer.elapsed()
    print('Batch searched', queries.shape[0], ' queries in ', compute_seconds, 's')
    timer_results['batch_searched_seconds'] = compute_seconds

    res_ids = tags - 1
    if gt_file != "":
        timer.reset()
        recall = utils.calculate_recall_from_gt_file(K, res_ids, gt_file)
        print(f"recall@{K} is {recall}")
        timer_results['recall_computed_seconds'] = timer.elapsed()

    timer_results['total_time_seconds'] = method_timer.elapsed()

    return timer_results


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
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("--gt_file", default="")
    parser.add_argument("--json_timings_output", required=False, default=None, help="File to write out timings to as JSON.  If not specified, timings will not be written out.")
    args = parser.parse_args()

    timings = insert_and_search(
        args.data_type,
        args.indexdata_file,
        args.querydata_file,
        args.Lbuild,
        args.graph_degree,  # Build args
        args.K,
        args.Lsearch,
        args.num_insert_threads,
        args.num_search_threads,  # search args
        args.gt_file,
    )

    if args.json_timings_output is not None:
        import json
        timings['log_file'] = args.json_timings_output
        with open(args.json_timings_output, "w") as f:
            json.dump(timings, f)

"""
An ingest optimized example with SIFT1M
source venv/bin/activate
python python/apps/in-mem-dynamic.py -d float \
-i "$HOME/data/sift/sift_base.fbin" -q "$HOME/data/sift/sift_query.fbin" --gt_file "$HOME/data/sift/gt100_base" \
-Lb 10 -R 30 -Ls 200
"""

