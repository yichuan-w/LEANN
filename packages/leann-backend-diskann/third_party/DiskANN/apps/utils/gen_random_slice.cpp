// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include "partition.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

template <typename T> int aux_main(char **argv)
{
    std::string base_file(argv[2]);
    std::string output_prefix(argv[3]);
    float sampling_rate = (float)(std::atof(argv[4]));
    gen_random_slice<T>(base_file, output_prefix, sampling_rate);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << argv[0]
                  << " data_type [float/int8/uint8] base_bin_file "
                     "sample_output_prefix sampling_probability"
                  << std::endl;
        exit(-1);
    }

    if (std::string(argv[1]) == std::string("float"))
    {
        aux_main<float>(argv);
    }
    else if (std::string(argv[1]) == std::string("int8"))
    {
        aux_main<int8_t>(argv);
    }
    else if (std::string(argv[1]) == std::string("uint8"))
    {
        aux_main<uint8_t>(argv);
    }
    else
        std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
    return 0;
}
