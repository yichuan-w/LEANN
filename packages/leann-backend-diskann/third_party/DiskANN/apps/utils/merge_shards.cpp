// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "disk_utils.h"
#include "cached_io.h"
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc != 9)
    {
        std::cout << argv[0]
                  << " vamana_index_prefix[1] vamana_index_suffix[2] "
                     "idmaps_prefix[3] "
                     "idmaps_suffix[4] n_shards[5] max_degree[6] "
                     "output_vamana_path[7] "
                     "output_medoids_path[8]"
                  << std::endl;
        exit(-1);
    }

    std::string vamana_prefix(argv[1]);
    std::string vamana_suffix(argv[2]);
    std::string idmaps_prefix(argv[3]);
    std::string idmaps_suffix(argv[4]);
    uint64_t nshards = (uint64_t)std::atoi(argv[5]);
    uint32_t max_degree = (uint64_t)std::atoi(argv[6]);
    std::string output_index(argv[7]);
    std::string output_medoids(argv[8]);

    return diskann::merge_shards(vamana_prefix, vamana_suffix, idmaps_prefix, idmaps_suffix, nshards, max_degree,
                                 output_index, output_medoids);
}
