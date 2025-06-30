// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "utils.h"
#include "index.h"
#include "memory_mapper.h"

namespace po = boost::program_options;

template <typename T> void bfs_count(const std::string &index_path, uint32_t data_dims)
{
    using TagT = uint32_t;
    using LabelT = uint32_t;
    diskann::Index<T, TagT, LabelT> index(diskann::Metric::L2, data_dims, 0, nullptr, nullptr, 0, false, false, false,
                                          false, 0, false);
    std::cout << "Index class instantiated" << std::endl;
    index.load(index_path.c_str(), 1, 100);
    std::cout << "Index loaded" << std::endl;
    index.count_nodes_at_bfs_levels();
}

int main(int argc, char **argv)
{
    std::string data_type, index_path_prefix;
    uint32_t data_dims;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("data_dims", po::value<uint32_t>(&data_dims)->required(), "Dimensionality of the data");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    try
    {
        if (data_type == std::string("int8"))
            bfs_count<int8_t>(index_path_prefix, data_dims);
        else if (data_type == std::string("uint8"))
            bfs_count<uint8_t>(index_path_prefix, data_dims);
        if (data_type == std::string("float"))
            bfs_count<float>(index_path_prefix, data_dims);
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index BFS failed." << std::endl;
        return -1;
    }
}
