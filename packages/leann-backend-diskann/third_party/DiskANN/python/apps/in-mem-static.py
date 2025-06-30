# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from xml.dom.pulldom import default_bufsize

import diskannpy
import numpy as np
import utils

def build_and_search(
    metric,
    dtype_str,
    index_directory,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    K,
    Ls,
    num_threads,
    gt_file,
    index_prefix,
    search_only
) -> dict[str, float]:
    """

    :param metric:
    :param dtype_str:
    :param index_directory:
    :param indexdata_file:
    :param querydata_file:
    :param Lb:
    :param graph_degree:
    :param K:
    :param Ls:
    :param num_threads:
    :param gt_file:
    :param index_prefix:
    :param search_only:
    :return: Dictionary of timings.  Key is the event and value is the number of seconds the event took
    in wall-clock-time.
    """
    timer_results: dict[str, float] = {}

    method_timer: utils.Timer = utils.Timer()

    if dtype_str == "float":
        dtype = np.single
    elif dtype_str == "int8":
        dtype = np.byte
    elif dtype_str == "uint8":
        dtype = np.ubyte
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    # build index
    if not search_only:
        build_index_timer = utils.Timer()
        diskannpy.build_memory_index(
            data=indexdata_file,
            distance_metric=metric,
            vector_dtype=dtype,
            index_directory=index_directory,
            complexity=Lb,
            graph_degree=graph_degree,
            num_threads=num_threads,
            index_prefix=index_prefix,
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
        )
        timer_results["build_index_seconds"] = build_index_timer.elapsed()

    # ready search object
    load_index_timer = utils.Timer()
    index = diskannpy.StaticMemoryIndex(
        distance_metric=metric,
        vector_dtype=dtype,
        index_directory=index_directory,
        num_threads=num_threads,  # this can be different at search time if you would like
        initial_search_complexity=Ls,
        index_prefix=index_prefix
    )
    timer_results["load_index_seconds"] = load_index_timer.elapsed()

    queries = utils.bin_to_numpy(dtype, querydata_file)

    query_timer = utils.Timer()
    ids, dists = index.batch_search(queries, 10, Ls, num_threads)
    query_time = query_timer.elapsed()
    qps = round(queries.shape[0]/query_time, 1)
    print('Batch searched', queries.shape[0], 'in', query_time, 's @', qps, 'QPS')
    timer_results["query_seconds"] = query_time

    if gt_file != "":
        recall_timer = utils.Timer()
        recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
        print(f"recall@{K} is {recall}")
        timer_results["recall_seconds"] = recall_timer.elapsed()

    timer_results['total_time_seconds'] = method_timer.elapsed()

    return timer_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="in-mem-static",
        description="Static in-memory build and search from vectors in a file",
    )

    parser.add_argument("-m", "--metric", required=False, default="l2")
    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-id", "--index_directory", required=False, default=".")
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-q", "--querydata_file", required=True)
    parser.add_argument("-Lb", "--Lbuild", default=50, type=int)
    parser.add_argument("-Ls", "--Lsearch", default=50, type=int)
    parser.add_argument("-R", "--graph_degree", default=32, type=int)
    parser.add_argument("-T", "--num_threads", default=8, type=int)
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("-G", "--gt_file", default="")
    parser.add_argument("-ip", "--index_prefix", required=False, default="ann")
    parser.add_argument("--search_only", required=False, default=False)
    parser.add_argument("--json_timings_output", required=False, default=None, help="File to write out timings to as JSON.  If not specified, timings will not be written out.")
    args = parser.parse_args()

    timings: dict[str, float] = build_and_search(
        args.metric,
        args.data_type,
        args.index_directory.strip(),
        args.indexdata_file.strip(),
        args.querydata_file.strip(),
        args.Lbuild,
        args.graph_degree,  # Build args
        args.K,
        args.Lsearch,
        args.num_threads,  # search args
        args.gt_file,
        args.index_prefix,
        args.search_only
    )

    if args.json_timings_output is not None:
        import json
        timings['log_file'] = args.json_timings_output
        with open(args.json_timings_output, "w") as f:
            json.dump(timings, f)
