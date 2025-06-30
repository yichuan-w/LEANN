// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ifstream &reader, std::ofstream &writer, uint32_t *read_buf, uint32_t *write_buf, size_t npts,
                   size_t ndims)
{
    reader.read((char *)read_buf, npts * (ndims * sizeof(uint32_t) + sizeof(uint32_t)));
    for (size_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(uint32_t));
    }
    writer.write((char *)write_buf, npts * ndims * sizeof(uint32_t));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_ivecs output_bin" << std::endl;
        exit(-1);
    }
    std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    size_t npts = fsize / ((ndims + 1) * sizeof(uint32_t));
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    size_t blk_size = 131072;
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    std::ofstream writer(argv[2], std::ios::binary);
    int npts_s32 = (int)npts;
    int ndims_s32 = (int)ndims;
    writer.write((char *)&npts_s32, sizeof(int));
    writer.write((char *)&ndims_s32, sizeof(int));
    uint32_t *read_buf = new uint32_t[npts * (ndims + 1)];
    uint32_t *write_buf = new uint32_t[npts * ndims];
    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();
}
