# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cluster", description="kmeans cluster points in a file"
    )

    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-k", "--num_clusters", type=int, required=True)
    args = parser.parse_args()

    npts, ndims = get_bin_metadata(indexdata_file)

    data = utils.bin_to_numpy(args.data_type, args.indexdata_file)

    offsets, permutation = utils.cluster_and_permute(
        args.data_type, npts, ndims, data, args.num_clusters
    )

    permuted_data = data[permutation]

    utils.numpy_to_bin(permuted_data, args.indexdata_file + ".cluster")
