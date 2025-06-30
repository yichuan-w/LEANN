// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

// Convert float types
void block_convert_float(std::ifstream &reader, std::ofstream &writer, float *read_buf, float *write_buf, size_t npts,
                         size_t ndims)
{
    reader.read((char *)read_buf, npts * (ndims * sizeof(float) + sizeof(uint32_t)));
    for (size_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(float));
    }
    writer.write((char *)write_buf, npts * ndims * sizeof(float));
}

// Convert byte types
void block_convert_byte(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf,
                        size_t npts, size_t ndims)
{
    reader.read((char *)read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(uint32_t)));
    for (size_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(uint32_t))) + sizeof(uint32_t),
               ndims * sizeof(uint8_t));
    }
    writer.write((char *)write_buf, npts * ndims * sizeof(uint8_t));
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << argv[0] << " <float/int8/uint8> input_vecs output_bin" << std::endl;
        exit(-1);
    }

    int datasize = sizeof(float);

    if (strcmp(argv[1], "uint8") == 0 || strcmp(argv[1], "int8") == 0)
    {
        datasize = sizeof(uint8_t);
    }
    else if (strcmp(argv[1], "float") != 0)
    {
        std::cout << "Error: type not supported. Use float/int8/uint8" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[2], std::ios::binary | std::ios::ate);
    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    size_t blk_size = 131072;
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    std::ofstream writer(argv[3], std::ios::binary);
    int32_t npts_s32 = (int32_t)npts;
    int32_t ndims_s32 = (int32_t)ndims;
    writer.write((char *)&npts_s32, sizeof(int32_t));
    writer.write((char *)&ndims_s32, sizeof(int32_t));

    size_t chunknpts = std::min(npts, blk_size);
    uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * datasize) + sizeof(uint32_t))];
    uint8_t *write_buf = new uint8_t[chunknpts * ndims * datasize];

    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        if (datasize == sizeof(float))
        {
            block_convert_float(reader, writer, (float *)read_buf, (float *)write_buf, cblk_size, ndims);
        }
        else
        {
            block_convert_byte(reader, writer, read_buf, write_buf, cblk_size, ndims);
        }
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();
}
