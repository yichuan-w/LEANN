// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_uint8_bin output_float_bin" << std::endl;
        exit(-1);
    }

    uint8_t *input;
    size_t npts, nd;
    diskann::load_bin<uint8_t>(argv[1], input, npts, nd);
    float *output = new float[npts * nd];
    diskann::convert_types<uint8_t, float>(input, output, npts, nd);
    diskann::save_bin<float>(argv[2], output, npts, nd);
    delete[] output;
    delete[] input;
}
