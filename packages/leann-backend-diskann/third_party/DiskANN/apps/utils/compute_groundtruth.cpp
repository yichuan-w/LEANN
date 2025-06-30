// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <unordered_map>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 10000000
#define ALIGNMENT 512

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set;
typedef std::string path;

namespace po = boost::program_options;

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct
{
    bool operator()(const pairIF &l, const pairIF &r)
    {
        return l.second < r.second;
    };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b)
{
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim)
{
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
        points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                                    matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix, // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq, // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq, // queries in Col major
                      float *ones_vec = NULL)          // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void inner_prod_to_points(const size_t dim,
                          float *dist_matrix, // Col Major, cols are queries, rows are points
                          size_t npoints, const float *const points, size_t nqueries, const float *const queries,
                          float *ones_vec = NULL) // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-1.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);

    if (ones_vec_alloc)
        delete[] ones_vec;
}

void exact_knn(const size_t dim, const size_t k,
               size_t *const closest_points,     // k * num_queries preallocated, col
                                                 // major, queries columns
               float *const dist_closest_points, // k * num_queries
                                                 // preallocated, Dist to
                                                 // corresponding closes_points
               size_t npoints,
               float *points_in, // points in Col major
               size_t nqueries, float *queries_in,
               diskann::Metric metric = diskann::Metric::L2) // queries in Col major
{
    float *points_l2sq = new float[npoints];
    float *queries_l2sq = new float[nqueries];
    compute_l2sq(points_l2sq, points_in, npoints, dim);
    compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

    float *points = points_in;
    float *queries = queries_in;

    if (metric == diskann::Metric::COSINE)
    { // we convert cosine distance as
      // normalized L2 distnace
        points = new float[npoints * dim];
        queries = new float[nqueries * dim];
#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)npoints; i++)
        {
            float norm = std::sqrt(points_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                points[i * dim + j] = points_in[i * dim + j] / norm;
            }
        }

#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)nqueries; i++)
        {
            float norm = std::sqrt(queries_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                queries[i * dim + j] = queries_in[i * dim + j] / norm;
            }
        }
        // recalculate norms after normalizing, they should all be one.
        compute_l2sq(points_l2sq, points, npoints, dim);
        compute_l2sq(queries_l2sq, queries, nqueries, dim);
    }

    std::cout << "Going to compute " << k << " NNs for " << nqueries << " queries over " << npoints << " points in "
              << dim << " dimensions using";
    if (metric == diskann::Metric::INNER_PRODUCT)
        std::cout << " MIPS ";
    else if (metric == diskann::Metric::COSINE)
        std::cout << " Cosine ";
    else
        std::cout << " L2 ";
    std::cout << "distance fn. " << std::endl;

    size_t q_batch_size = (1 << 9);
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)npoints];

    for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b)
    {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

        if (metric == diskann::Metric::L2 || metric == diskann::Metric::COSINE)
        {
            distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                             queries + (ptrdiff_t)q_b * (ptrdiff_t)dim, queries_l2sq + q_b);
        }
        else
        {
            inner_prod_to_points(dim, dist_matrix, npoints, points, q_e - q_b,
                                 queries + (ptrdiff_t)q_b * (ptrdiff_t)dim);
        }
        std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++)
        {
            maxPQIFCS point_dist;
            for (size_t p = 0; p < k; p++)
                point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
            for (size_t p = k; p < npoints; p++)
            {
                if (point_dist.top().second > dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints])
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
                if (point_dist.size() > k)
                    point_dist.pop();
            }
            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l)
            {
                closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().first;
                dist_closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().second;
                point_dist.pop();
            }
            assert(std::is_sorted(dist_closest_points + (ptrdiff_t)q * (ptrdiff_t)k,
                                  dist_closest_points + (ptrdiff_t)(q + 1) * (ptrdiff_t)k));
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
    }

    delete[] dist_matrix;

    delete[] points_l2sq;
    delete[] queries_l2sq;

    if (metric == diskann::Metric::COSINE)
    {
        delete[] points;
        delete[] queries;
    }
}

template <typename T> inline int get_num_parts(const char *filename)
{
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    std::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
    reader.close();
    uint32_t num_parts =
        (npts_i32 % PARTSIZE) == 0 ? npts_i32 / PARTSIZE : (uint32_t)std::floor(npts_i32 / PARTSIZE) + 1;
    std::cout << "Number of parts: " << num_parts << std::endl;
    return num_parts;
}

template <typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims, int part_num)
{
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    uint64_t start_id = part_num * PARTSIZE;
    uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t)npts_i32);
    npts = end_id - start_id;
    ndims = (uint64_t)ndims_i32;
    std::cout << "#pts in part = " << npts << ", #dims = " << ndims << ", size = " << npts * ndims * sizeof(T) << "B"
              << std::endl;

    reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
    T *data_T = new T[npts * ndims];
    reader.read((char *)data_T, sizeof(T) * npts * ndims);
    std::cout << "Finished reading part of the bin file." << std::endl;
    reader.close();
    data = aligned_malloc<float>(npts * ndims, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
    for (int64_t i = 0; i < (int64_t)npts; i++)
    {
        for (int64_t j = 0; j < (int64_t)ndims; j++)
        {
            float cur_val_float = (float)data_T[i * ndims + j];
            std::memcpy((char *)(data + i * ndims + j), (char *)&cur_val_float, sizeof(float));
        }
    }
    delete[] data_T;
    std::cout << "Finished converting part data to float." << std::endl;
}

template <typename T> inline void save_bin(const std::string filename, T *data, size_t npts, size_t ndims)
{
    std::ofstream writer;
    writer.exceptions(std::ios::failbit | std::ios::badbit);
    writer.open(filename, std::ios::binary | std::ios::out);
    std::cout << "Writing bin: " << filename << "\n";
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
              << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int) << "B" << std::endl;

    writer.write((char *)data, npts * ndims * sizeof(T));
    writer.close();
    std::cout << "Finished writing bin" << std::endl;
}

inline void save_groundtruth_as_one_file(const std::string filename, int32_t *data, float *distances, size_t npts,
                                         size_t ndims)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
                 "npts*dim dist-matrix) with npts = "
              << npts << ", dim = " << ndims << ", size = " << 2 * npts * ndims * sizeof(uint32_t) + 2 * sizeof(int)
              << "B" << std::endl;

    writer.write((char *)data, npts * ndims * sizeof(uint32_t));
    writer.write((char *)distances, npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Finished writing truthset" << std::endl;
}

template <typename T>
std::vector<std::vector<std::pair<uint32_t, float>>> processUnfilteredParts(const std::string &base_file,
                                                                            size_t &nqueries, size_t &npoints,
                                                                            size_t &dim, size_t &k, float *query_data,
                                                                            const diskann::Metric &metric,
                                                                            std::vector<uint32_t> &location_to_tag)
{
    float *base_data = nullptr;
    int num_parts = get_num_parts<T>(base_file.c_str());
    std::vector<std::vector<std::pair<uint32_t, float>>> res(nqueries);
    for (int p = 0; p < num_parts; p++)
    {
        size_t start_id = p * PARTSIZE;
        load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);

        size_t *closest_points_part = new size_t[nqueries * k];
        float *dist_closest_points_part = new float[nqueries * k];

        auto part_k = k < npoints ? k : npoints;
        exact_knn(dim, part_k, closest_points_part, dist_closest_points_part, npoints, base_data, nqueries, query_data,
                  metric);

        for (size_t i = 0; i < nqueries; i++)
        {
            for (size_t j = 0; j < part_k; j++)
            {
                if (!location_to_tag.empty())
                    if (location_to_tag[closest_points_part[i * k + j] + start_id] == 0)
                        continue;

                res[i].push_back(std::make_pair((uint32_t)(closest_points_part[i * part_k + j] + start_id),
                                                dist_closest_points_part[i * part_k + j]));
            }
        }

        delete[] closest_points_part;
        delete[] dist_closest_points_part;

        diskann::aligned_free(base_data);
    }
    return res;
};

template <typename T>
int aux_main(const std::string &base_file, const std::string &query_file, const std::string &gt_file, size_t k,
             const diskann::Metric &metric, const std::string &tags_file = std::string(""))
{
    size_t npoints, nqueries, dim;

    float *query_data;

    load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);
    if (nqueries > PARTSIZE)
        std::cerr << "WARNING: #Queries provided (" << nqueries << ") is greater than " << PARTSIZE
                  << ". Computing GT only for the first " << PARTSIZE << " queries." << std::endl;

    // load tags
    const bool tags_enabled = tags_file.empty() ? false : true;
    std::vector<uint32_t> location_to_tag = diskann::loadTags(tags_file, base_file);

    int *closest_points = new int[nqueries * k];
    float *dist_closest_points = new float[nqueries * k];

    std::vector<std::vector<std::pair<uint32_t, float>>> results =
        processUnfilteredParts<T>(base_file, nqueries, npoints, dim, k, query_data, metric, location_to_tag);

    for (size_t i = 0; i < nqueries; i++)
    {
        std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
        std::sort(cur_res.begin(), cur_res.end(), custom_dist);
        size_t j = 0;
        for (auto iter : cur_res)
        {
            if (j == k)
                break;
            if (tags_enabled)
            {
                std::uint32_t index_with_tag = location_to_tag[iter.first];
                closest_points[i * k + j] = (int32_t)index_with_tag;
            }
            else
            {
                closest_points[i * k + j] = (int32_t)iter.first;
            }

            if (metric == diskann::Metric::INNER_PRODUCT)
                dist_closest_points[i * k + j] = -iter.second;
            else
                dist_closest_points[i * k + j] = iter.second;

            ++j;
        }
        if (j < k)
            std::cout << "WARNING: found less than k GT entries for query " << i << std::endl;
    }

    save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points, nqueries, k);
    delete[] closest_points;
    delete[] dist_closest_points;
    diskann::aligned_free(query_data);

    return 0;
}

void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (uint32_t)npts_i32;
    dim = (uint32_t)dim_i32;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... " << std::endl;

    int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
                            // only ids, -1 is error
    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
        truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
        truthset_type = 2;

    if (truthset_type == -1)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. File should have bin format, with "
                  "npts followed by ngt followed by npts*ngt ids and optionally "
                  "followed by npts*ngt distance values; actual size: "
               << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
               << expected_file_size_just_ids;
        diskann::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *)ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1)
    {
        dists = new float[npts * dim];
        reader.read((char *)dists, npts * dim * sizeof(float));
    }
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, base_file, query_file, gt_file, tags_file;
    uint64_t K;

    try
    {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("base_file", po::value<std::string>(&base_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "File name for the writing ground truth in binary "
                           "format, please don' append .bin at end if "
                           "no filter_label or filter_label_file is provided it "
                           "will save the file with '.bin' at end."
                           "else it will save the file as filename_label.bin");
        desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");
        desc.add_options()("tags_file", po::value<std::string>(&tags_file)->default_value(std::string()),
                           "File containing the tags in binary format");

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

    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
    {
        std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
            aux_main<float>(base_file, query_file, gt_file, K, metric, tags_file);
        if (data_type == std::string("int8"))
            aux_main<int8_t>(base_file, query_file, gt_file, K, metric, tags_file);
        if (data_type == std::string("uint8"))
            aux_main<uint8_t>(base_file, query_file, gt_file, K, metric, tags_file);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Compute GT failed." << std::endl;
        return -1;
    }
}
