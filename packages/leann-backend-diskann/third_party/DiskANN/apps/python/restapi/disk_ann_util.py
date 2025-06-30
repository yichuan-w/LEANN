# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import os
import subprocess


def output_vectors(
    diskann_build_path: str,
    temporary_file_path: str,
    vectors: np.ndarray,
    timeout: int = 60
) -> str:
    vectors_as_tsv_path = os.path.join(temporary_file_path, "vectors.tsv")
    with open(vectors_as_tsv_path, "w") as vectors_tsv_out:
        for vector in vectors:
            as_str = "\t".join((str(component) for component in vector))
            print(as_str, file=vectors_tsv_out)
    # there is probably a clever way to have numpy write out C++ friendly floats, so feel free to remove this in
    # favor of something more sane later
    vectors_as_bin_path = os.path.join(temporary_file_path, "vectors.bin")
    tsv_to_bin_path = os.path.join(diskann_build_path, "apps", "utils", "tsv_to_bin")

    number_of_points, dimensions = vectors.shape
    args = [
        tsv_to_bin_path,
        "float",
        vectors_as_tsv_path,
        vectors_as_bin_path,
        str(dimensions),
        str(number_of_points)
    ]
    completed = subprocess.run(args, timeout=timeout)
    if completed.returncode != 0:
        raise Exception(f"Unable to convert tsv to binary using tsv_to_bin, completed_process: {completed}")
    return vectors_as_bin_path


def build_ssd_index(
    diskann_build_path: str,
    temporary_file_path: str,
    vectors: np.ndarray,
    per_process_timeout: int = 60  # this may not be long enough if you're doing something larger
):
    vectors_as_bin_path = output_vectors(diskann_build_path, temporary_file_path, vectors, timeout=per_process_timeout)

    ssd_builder_path = os.path.join(diskann_build_path, "apps", "build_disk_index")
    args = [
        ssd_builder_path,
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", vectors_as_bin_path,
        "--index_path_prefix", os.path.join(temporary_file_path, "smoke_test"),
        "-R", "64",
        "-L", "100",
        "--search_DRAM_budget", "1",
        "--build_DRAM_budget", "1",
        "--num_threads", "1",
        "--PQ_disk_bytes", "0"
    ]
    completed = subprocess.run(args, timeout=per_process_timeout)

    if completed.returncode != 0:
        command_run = " ".join(args)
        raise Exception(f"Unable to build a disk index with the command: '{command_run}'\ncompleted_process: {completed}\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
    # index is now built inside of temporary_file_path
