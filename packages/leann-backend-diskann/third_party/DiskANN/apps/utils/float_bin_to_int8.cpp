// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ofstream &writer, int8_t *write_buf, std::ifstream &reader, float *read_buf, size_t npts,
                   size_t ndims, float bias, float scale)
{
    reader.read((char *)read_buf, npts * ndims * sizeof(float));

    for (size_t i = 0; i < npts; i++)
    {
        for (size_t d = 0; d < ndims; d++)
        {
            write_buf[d + i * ndims] = (int8_t)((read_buf[d + i * ndims] - bias) * (254.0 / scale));
        }
    }
    writer.write((char *)write_buf, npts * ndims);
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << "  input_bin  output_tsv  bias  scale" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[1], std::ios::binary);
    uint32_t npts_u32;
    uint32_t ndims_u32;
    reader.read((char *)&npts_u32, sizeof(uint32_t));
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    size_t npts = npts_u32;
    size_t ndims = ndims_u32;
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    size_t blk_size = 131072;
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

    std::ofstream writer(argv[2], std::ios::binary);
    auto read_buf = new float[blk_size * ndims];
    auto write_buf = new int8_t[blk_size * ndims];
    float bias = (float)atof(argv[3]);
    float scale = (float)atof(argv[4]);

    writer.write((char *)(&npts_u32), sizeof(uint32_t));
    writer.write((char *)(&ndims_u32), sizeof(uint32_t));

    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert(writer, write_buf, reader, read_buf, cblk_size, ndims, bias, scale);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    writer.close();
    reader.close();
}
