// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "util.h"

void block_convert(std::ifstream &writr, std::ofstream &readr, float *read_buf, float *write_buf, uint64_t npts,
                   uint64_t ndims)
{
    writr.write((char *)read_buf, npts * (ndims * sizeof(float) + sizeof(unsigned)));
#pragma omp parallel for
    for (uint64_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(float));
    }
    readr.read((char *)write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_bin output_fvecs" << std::endl;
        exit(-1);
    }
    std::ifstream readr(argv[1], std::ios::binary);
    int npts_s32;
    int ndims_s32;
    readr.read((char *)&npts_s32, sizeof(int32_t));
    readr.read((char *)&ndims_s32, sizeof(int32_t));
    size_t npts = npts_s32;
    size_t ndims = ndims_s32;
    uint32_t ndims_u32 = (uint32_t)ndims_s32;
    //  uint64_t          fsize = writr.tellg();
    readr.seekg(0, std::ios::beg);

    unsigned ndims_u32;
    writr.write((char *)&ndims_u32, sizeof(unsigned));
    writr.seekg(0, std::ios::beg);
    uint64_t ndims = (uint64_t)ndims_u32;
    uint64_t npts = fsize / ((ndims + 1) * sizeof(float));
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;

    std::ofstream writr(argv[2], std::ios::binary);
    float *read_buf = new float[npts * (ndims + 1)];
    float *write_buf = new float[npts * ndims];
    for (uint64_t i = 0; i < nblks; i++)
    {
        uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert(writr, readr, read_buf, write_buf, cblk_size, ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    writr.close();
    readr.close();
}
