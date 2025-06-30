// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include <algorithm>
#include <memory>
#include <cmath>

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"
#include "embedding.pb.h" // from embedding.proto -> embedding.pb.h
#include <zmq.h>
#include <fstream>
#include <atomic>
#include <mutex>
#include <filesystem>

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#elif __APPLE__
#include "apple_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{
static std::mutex log_file_mutex;
static std::atomic<int> search_counter(0);

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                      std::shared_ptr<AlignedFileReader> &graphReader, diskann::Metric m)
    : reader(fileReader), graph_reader(graphReader), metric(m), _thread_data(nullptr)
{
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

#if 1
    // -- If not partition_read, this is the normal DiskANN approach:
    if (!_use_partition)
    {
#endif
        // (1) read each node's 4 KB from offset = get_node_sector(node_id)*4096
        alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

        // create read requests
        for (size_t i = 0; i < node_ids.size(); ++i)
        {
            auto node_id = node_ids[i];

            AlignedRead read;
            read.len = num_sectors * defaults::SECTOR_LEN;
            read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
            read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
            read_reqs.push_back(read);
        }

        reader->read(read_reqs, ctx);

        // copy reads into buffers
        for (uint32_t i = 0; i < read_reqs.size(); i++)
        {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
            if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
            {
                retval[i] = false;
                continue;
            }
#endif

            char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

            if (coord_buffers[i] != nullptr)
            {
                T *node_coords = offset_to_node_coords(node_buf);
                memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
            }

            if (nbr_buffers[i].second != nullptr)
            {
                uint32_t *node_nhood = offset_to_node_nhood(node_buf);
                auto num_nbrs = *node_nhood;
                nbr_buffers[i].first = num_nbrs;
                memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
            }
        }
        aligned_free(buf);

        if (!_use_partition)
        {
            // done with the normal path
            return retval;
        }
#if 1
    }
#endif

    {
        // 计算每个节点的分区偏移
        std::vector<std::pair<uint64_t, uint64_t>> offsets(node_ids.size());
        std::vector<bool> valid_nodes(node_ids.size(), true);

        // 按分区分组，减少重复读取相同分区
        std::map<uint32_t, std::vector<size_t>> partition_to_indices;

        // 遍历所有节点，获取其分区信息
        for (size_t i = 0; i < node_ids.size(); i++)
        {
            uint32_t node_id = node_ids[i];
            if (nbr_buffers[i].second != nullptr)
            {
                // 使用read_neighbors中的逻辑获取分区ID
                uint32_t partition_id = _id2partition[node_id];
                if (partition_id >= _num_partitions)
                {
                    valid_nodes[i] = false;
                    retval[i] = false;
                    continue;
                }

                // 将节点按分区ID分组
                partition_to_indices[partition_id].push_back(i);
            }
        }

        // 对每个分区执行一次读取
        for (const auto &pair : partition_to_indices)
        {
            uint32_t partition_id = pair.first;
            const auto &indices = pair.second;

            // 计算扇区偏移 (与read_neighbors中相同)
            uint64_t sector_offset = (partition_id + 1) * defaults::SECTOR_LEN;

            // 读取分区扇区
            char *sector_buf = nullptr;
            alloc_aligned((void **)&sector_buf, defaults::SECTOR_LEN, defaults::SECTOR_LEN);

            AlignedRead read;
            read.len = defaults::SECTOR_LEN;
            read.buf = sector_buf;
            read.offset = sector_offset;

            std::vector<AlignedRead> single_read = {read};
            graph_reader->read(single_read, ctx);

            // 处理该分区中的所有节点
            for (size_t idx : indices)
            {
                uint32_t node_id = node_ids[idx];

                // 查找节点在分区内的位置 (与read_neighbors中相同)
                const auto &part_list = _graph_partitions[partition_id];
                auto it = std::find(part_list.begin(), part_list.end(), node_id);
                if (it == part_list.end())
                {
                    retval[idx] = false;
                    continue;
                }
                size_t j = std::distance(part_list.begin(), it);

                // 计算节点在扇区内的偏移 (与read_neighbors中相同)
                uint64_t node_offset = j * _graph_node_len;
                if (node_offset + 4 > defaults::SECTOR_LEN)
                {
                    retval[idx] = false;
                    continue;
                }

                // 读取邻居数量
                char *adjacency_ptr = sector_buf + node_offset;
                uint32_t neighbor_count = *reinterpret_cast<uint32_t *>(adjacency_ptr);

                // 检查邻居数据是否超出扇区范围
                size_t needed = neighbor_count * sizeof(uint32_t);
                if (node_offset + 4 + needed > defaults::SECTOR_LEN)
                {
                    retval[idx] = false;
                    continue;
                }

                // 拷贝邻居数据
                nbr_buffers[idx].first = neighbor_count;
                memcpy(nbr_buffers[idx].second, adjacency_ptr + 4, needed);
            }

            aligned_free(sector_buf);
        }
    }

    return retval;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    size_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min((size_t)num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    uint64_t tenp_nodes = (uint64_t)(std::round(this->_num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(10 percent of total nodes:" << this->_num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    if ((_filter_to_medoid_ids.size() > 0) && (cur_level->size() < num_nodes_to_cache))
    {
        for (auto &x : _filter_to_medoid_ids)
        {
            for (auto &y : x.second)
            {
                cur_level->insert(y);
                if (cur_level->size() == num_nodes_to_cache)
                    break;
            }
            if (cur_level->size() == num_nodes_to_cache)
                break;
        }
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        for (const uint32_t &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        size_t BLOCK_SIZE = 1024;
        size_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            diskann::cout << "." << std::flush;
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // process each nhood buf
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_set.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix,
                                  const char *pq_prefix)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix, const char *pq_prefix,
                                  const char *partition_prefix)
{
#endif
    if (pq_prefix == nullptr || strcmp(pq_prefix, "") == 0)
    {
        pq_prefix = index_prefix;
    }
    if (partition_prefix != nullptr && strcmp(partition_prefix, "") != 0)
    {
        _use_partition = true;
    }
    std::string pq_table_bin = std::string(pq_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(pq_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
    std::string graph_file = std::string(partition_prefix) + "_disk_graph.index";
    std::string partition_file = std::string(partition_prefix) + "_partition.bin";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str(), graph_file.c_str(), partition_file.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str(), graph_file.c_str(), partition_file.c_str());
#endif
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::read_partition_info(const std::string &partition_bin)
{
    std::ifstream pf(partition_bin, std::ios::binary);
    if (!pf.is_open())
    {
        diskann::cout << "Cannot open partition.bin: " << partition_bin << std::endl;
        return 1;
    }
    diskann::cout << "Loading partition info from " << partition_bin << std::endl;
    uint64_t C, nd;
    READ_U64(pf, C);
    READ_U64(pf, _num_partitions);
    READ_U64(pf, nd);
    std::cout << "[partition.bin header] C=" << C << ", partition_nums=" << _num_partitions << ", nd=" << nd
              << std::endl;

    // 读取分区节点列表
    _graph_partitions.resize(_num_partitions);
    for (uint64_t i = 0; i < _num_partitions; i++)
    {
        uint32_t psize;
        READ_U32(pf, psize);
        _graph_partitions[i].resize(psize);
        pf.read(reinterpret_cast<char *>(_graph_partitions[i].data()), psize * sizeof(uint32_t));
    }
    // 读取 _id2partition[node], 大小= nd
    _id2partition.resize(nd);
    pf.read(reinterpret_cast<char *>(_id2partition.data()), nd * sizeof(uint32_t));
    pf.close();
    std::cout << "Done loading partition info.\n";

    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_graph_index(const std::string &graph_index_file)
{
    std::ifstream gf(graph_index_file, std::ios::binary);
    if (!gf.is_open())
    {
        diskann::cout << "Cannot open disk_graph.index: " << graph_index_file << std::endl;
        return 1;
    }
    diskann::cout << "Loading graph index from " << graph_index_file << std::endl;

    // (a) sector0 => read 2 ints for meta_n and meta_dim
    int meta_n, meta_dim;
    gf.read((char *)&meta_n, sizeof(int));
    gf.read((char *)&meta_dim, sizeof(int));
    diskann::cout << "[debug] meta_n=" << meta_n << ", meta_dim=" << meta_dim << "\n";

    // (b) Read uint64_t meta_n times
    std::vector<uint64_t> meta_info(meta_n);
    gf.read(reinterpret_cast<char *>(meta_info.data()), meta_n * sizeof(uint64_t));
    for (int i = 0; i < meta_n; i++)
    {
        diskann::cout << " meta_info[" << i << "]= " << meta_info[i] << "\n";
    }

    size_t file_size = get_file_size(graph_index_file);
    diskann::cout << "[disk_graph.index size] " << file_size << " bytes\n";

    uint64_t nd_in_meta = meta_info[0];
    uint64_t dim_in_meta = meta_info[1];
    uint64_t max_node_len = meta_info[3];
    uint64_t c_in_meta = meta_info[4];
    uint64_t entire_file_sz = meta_info[8];

    diskann::cout << "Based on meta_info:\n"
                  << "  nd_in_meta= " << nd_in_meta << ", dim_in_meta= " << dim_in_meta
                  << ", max_node_len= " << max_node_len << ", c_in_meta= " << c_in_meta
                  << ", entire_file_size= " << entire_file_sz << "\n";

    uint64_t dim_size = dim_in_meta * sizeof(float);

    _graph_node_len = max_node_len - dim_size;

#if 0
    assert(max_node_len == _max_node_len);
    assert(dim_size == _disk_bytes_per_point);
    assert(_graph_node_len / sizeof(float) == _max_degree + 1);
#endif

    // Compensate the losting info from old meta_info
    _max_degree = _graph_node_len / sizeof(float) - 1;
    _disk_bytes_per_point = dim_size;
    _max_node_len = max_node_len;

    diskann::cout << " => graph_node_len= " << _graph_node_len << "\n\n";

    return 0;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath, const char *graph_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath,
                                                      const char *graph_file, const char *partition_file)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    // medoids, etc.
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std::string(_disk_index_file) + "_labels_map.txt";

    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim = 0, pq_file_num_centroids = 0;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Got " << pq_file_num_centroids << " PQ centroids, loading from " << pq_table_bin << std::endl;
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    diskann::cout << "Loading index metadata from " << _disk_index_file << std::endl;
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    size_t medoid_id_on_file;
#if 1
    if (!_use_partition)
    {
#endif
        if (!index_metadata.is_open())
        {
            diskann::cout << "Error: Could not open index metadata file: " << _disk_index_file << std::endl;
            return -1;
        }

        uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                         // metadata, nc should be 1)
        READ_U32(index_metadata, nr);
        READ_U32(index_metadata, nc);

        uint64_t disk_nnodes;
        uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
        READ_U64(index_metadata, disk_nnodes);
        READ_U64(index_metadata, disk_ndims);

        if (disk_nnodes != _num_points)
        {
            diskann::cout << "Mismatch in #points for compressed data file and disk "
                             "index file: "
                          << disk_nnodes << " vs " << _num_points << std::endl;
            return -1;
        }

        READ_U64(index_metadata, medoid_id_on_file);
        READ_U64(index_metadata, _max_node_len);
        READ_U64(index_metadata, _nnodes_per_sector);
        _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

        if (_max_degree > defaults::MAX_GRAPH_DEGREE)
        {
            std::stringstream stream;
            stream << "Error loading index. Ensure that max graph degree (R) does "
                      "not exceed "
                   << defaults::MAX_GRAPH_DEGREE << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        // setting up concept of frozen points in disk index for streaming-DiskANN
        READ_U64(index_metadata, this->_num_frozen_points);
        uint64_t file_frozen_id;
        READ_U64(index_metadata, file_frozen_id);
        if (this->_num_frozen_points == 1)
            this->_frozen_location = file_frozen_id;
        if (this->_num_frozen_points == 1)
        {
            diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                          << ". Will not output it at search time." << std::endl;
        }

        READ_U64(index_metadata, this->_reorder_data_exists);
        if (this->_reorder_data_exists)
        {
            if (this->_use_disk_index_pq == false)
            {
                throw ANNException("Reordering is designed for used with disk PQ "
                                   "compression option",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            READ_U64(index_metadata, this->_reorder_data_start_sector);
            READ_U64(index_metadata, this->_ndims_reorder_vecs);
            READ_U64(index_metadata, this->_nvecs_per_sector);
        }

        diskann::cout << "Disk-Index File Meta-data: ";
        diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
        diskann::cout << ", max node len (bytes): " << _max_node_len;
        diskann::cout << ", max node degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
        delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
        // open AlignedFileReader handle to index_file
        std::string index_fname(_disk_index_file);
        reader->open(index_fname);

        diskann::cout << "Disk-Index Meta: nodes per sector: " << _nnodes_per_sector
                      << ", max node len: " << _max_node_len << ", max node degree: " << _max_degree << std::endl;

#endif

#if 1
    }
#endif

    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        if (_use_partition)
        {
            assert(false); // We do not have a valid medoid id in the partition file.
        }
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        size_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }

    if (_use_partition)
    {
        read_partition_info(partition_file);

        this->_graph_index_file = graph_file;
        graph_reader->open(this->_graph_index_file);
        load_graph_index(this->_graph_index_file);
    }

    diskann::cout << "load_from_separate_paths done." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data, QueryStats *stats,
                                                 bool USE_DEFERRED_FETCH, bool skip_search_reorder,
                                                 bool recompute_beighbor_embeddings, bool dedup_node_dis,
                                                 float prune_ratio, const bool batch_recompute, bool global_pruning)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, stats, USE_DEFERRED_FETCH, skip_search_reorder, recompute_beighbor_embeddings,
                       dedup_node_dis, prune_ratio, batch_recompute, global_pruning);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats,
                                                 bool USE_DEFERRED_FETCH, bool skip_search_reorder,
                                                 bool recompute_beighbor_embeddings, bool dedup_node_dis,
                                                 float prune_ratio, const bool batch_recompute, bool global_pruning)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, stats, USE_DEFERRED_FETCH,
                       skip_search_reorder, recompute_beighbor_embeddings, dedup_node_dis, prune_ratio, batch_recompute,
                       global_pruning);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats, bool USE_DEFERRED_FETCH, bool skip_search_reorder,
                                                 bool recompute_beighbor_embeddings, bool dedup_node_dis,
                                                 float prune_ratio, const bool batch_recompute, bool global_pruning)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, stats, USE_DEFERRED_FETCH, skip_search_reorder, recompute_beighbor_embeddings,
                       dedup_node_dis, prune_ratio, batch_recompute, global_pruning);
}

// A helper callback for cURL
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

static void *g_zmq_context = zmq_ctx_new();

struct ZmqContextManager
{
    ~ZmqContextManager()
    {
        if (g_zmq_context)
        {
            zmq_ctx_destroy(g_zmq_context);
            g_zmq_context = nullptr;
        }
    }
};
static ZmqContextManager g_zmq_manager;

bool fetch_embeddings_zmq(const std::vector<uint32_t> &node_ids, std::vector<std::vector<float>> &out_embeddings,
                          int zmq_port)
{
    // 1. Protobuf 序列化：创建请求消息
    protoembedding::NodeEmbeddingRequest req_proto;
    for (const auto id : node_ids)
    {
        req_proto.add_node_ids(id);
    }
    std::string req_str;
    if (!req_proto.SerializeToString(&req_str))
    {
        std::cerr << "ZMQ_FETCH_ERROR: Failed to serialize NodeEmbeddingRequest.\n";
        return false;
    }

    // 2. 使用线程本地(thread_local)的 Socket，实现连接复用
    // 每个线程将拥有自己独立的、持久化的 Socket
    thread_local void *tl_socket = nullptr;

    // 如果当前线程的 Socket 还未创建，则初始化并连接
    if (tl_socket == nullptr)
    {
        // 从全局 Context 创建 Socket
        tl_socket = zmq_socket(g_zmq_context, ZMQ_REQ);
        if (!tl_socket)
        {
            std::cerr << "ZMQ_FETCH_ERROR: zmq_socket() failed: " << zmq_strerror(zmq_errno()) << "\n";
            return false;
        }

        int timeout = 30000; // 30 秒超时
        zmq_setsockopt(tl_socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(tl_socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

        std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
        if (zmq_connect(tl_socket, endpoint.c_str()) != 0)
        {
            std::cerr << "ZMQ_FETCH_ERROR: zmq_connect() to " << endpoint << " failed: " << zmq_strerror(zmq_errno())
                      << "\n";
            zmq_close(tl_socket);
            tl_socket = nullptr; // 重置为空指针，以便下次调用时可以尝试重建
            return false;
        }
    }

    // 3. 使用已建立的连接发送请求
    if (zmq_send(tl_socket, req_str.data(), req_str.size(), 0) < 0)
    {
        std::cerr << "ZMQ_FETCH_ERROR: zmq_send() failed: " << zmq_strerror(zmq_errno()) << "\n";
        zmq_close(tl_socket); // 连接可能已失效，关闭它
        tl_socket = nullptr;  // 重置，强制下次重建
        return false;
    }

    // 4. 接收响应
    zmq_msg_t response_msg;
    zmq_msg_init(&response_msg);
    bool success = true;

    if (zmq_msg_recv(&response_msg, tl_socket, 0) < 0)
    {
        std::cerr << "ZMQ_FETCH_ERROR: zmq_msg_recv() failed: " << zmq_strerror(zmq_errno()) << "\n";
        zmq_close(tl_socket); // 同样，接收超时后连接也可能无效
        tl_socket = nullptr;  // 重置，强制下次重建
        success = false;
    }
    else
    {
        // 5. Protobuf 反序列化并提取数据
        protoembedding::NodeEmbeddingResponse resp_proto;
        if (!resp_proto.ParseFromArray(zmq_msg_data(&response_msg), static_cast<int>(zmq_msg_size(&response_msg))))
        {
            std::cerr << "ZMQ_FETCH_ERROR: Failed to parse NodeEmbeddingResponse from server.\n";
            success = false;
        }
        else
        {
            if (resp_proto.dimensions_size() == 2)
            {
                int batch_size = resp_proto.dimensions(0);
                int embedding_dim = resp_proto.dimensions(1);
                const std::string &emb_data = resp_proto.embeddings_data();
                size_t expected_bytes = (size_t)batch_size * embedding_dim * sizeof(float);

                if (batch_size >= 0 && emb_data.size() == expected_bytes)
                {
                    out_embeddings.resize(batch_size);
                    if (batch_size > 0)
                    {
                        const float *float_data = reinterpret_cast<const float *>(emb_data.data());
                        for (int i = 0; i < batch_size; ++i)
                        {
                            out_embeddings[i].resize(embedding_dim);
                            std::memcpy(out_embeddings[i].data(), float_data + (size_t)i * embedding_dim,
                                        embedding_dim * sizeof(float));
                        }
                    }
                }
                else
                {
                    std::cerr << "ZMQ_FETCH_ERROR: Embedding data size mismatch. Expected " << expected_bytes
                              << " bytes, got " << emb_data.size() << ".\n";
                    success = false;
                }
            }
            else
            {
                std::cerr << "ZMQ_FETCH_ERROR: Server response has invalid dimensions size.\n";
                success = false;
            }
        }
    }

    // 6. 清理消息对象，但保持 Socket 和 Context 开放以备下次复用
    zmq_msg_close(&response_msg);

    return success;
}

/**
 * fetch_embeddings_http: Function for backward compatibility, now uses ZMQ exclusively
 */
bool fetch_embeddings_http(const std::vector<uint32_t> &node_ids, std::vector<std::vector<float>> &out_embeddings)
{
    // Use ZMQ implementation exclusively
    return fetch_embeddings_zmq(node_ids, out_embeddings, 5555);
}

//! Should be aligned with utils.h::prepare_base_for_inner_products
void preprocess_fetched_embeddings(std::vector<std::vector<float>> &embeddings, diskann::Metric metric,
                                   float max_base_norm, uint32_t data_dim)
{
    for (auto &emb : embeddings)
    {
        // Ensure embedding has correct size
        if (emb.size() < data_dim - 1)
        {
            // Pad with zeros if needed
            emb.resize(data_dim - 1, 0);
        }

        if (metric == diskann::Metric::INNER_PRODUCT)
        {
            // For inner product, apply same preprocessing as in prepare_base_for_inner_products

            // Calculate original norm
            float norm_sq = 0;
            for (size_t i = 0; i < data_dim - 1; i++)
            {
                norm_sq += emb[i] * emb[i];
            }

            // Normalize by max_base_norm (same as in index construction)
            for (size_t i = 0; i < data_dim - 1; i++)
            {
                emb[i] /= max_base_norm;
            }

            // Add the extra coordinate for MIPS->L2 conversion
            float res = 1 - (norm_sq / (max_base_norm * max_base_norm));
            res = res <= 0 ? 0 : std::sqrt(res);
            emb.resize(data_dim, res);
        }
        else if (metric == diskann::Metric::COSINE)
        {
            // For cosine similarity, just normalize the vector
            float norm = 0;
            for (auto val : emb)
            {
                norm += val * val;
            }
            norm = std::sqrt(norm);

            if (norm > 0)
            {
                for (size_t i = 0; i < emb.size(); i++)
                {
                    emb[i] /= norm;
                }
            }
        }
        // For L2, no preprocessing needed
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats, bool USE_DEFERRED_FETCH, bool skip_search_reorder,
                                                 bool recompute_beighbor_embeddings, const bool dedup_node_dis,
                                                 float prune_ratio, const bool batch_recompute, bool global_pruning)
{
    // printf("cached_beam_search\n");
    // diskann::cout << "cached_beam_search" << std::endl;
    // diskann out prune_ratio
    prune_ratio = 1 - prune_ratio;
    diskann::cout << "reserve ratio: " << prune_ratio << std::endl;
    // prune_ratio = 0.8;
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // Add cache hit tracking variables
    uint64_t total_nodes_requested = 0;
    uint64_t total_nodes_from_cache = 0;

    // normalization step. for cosine, we simply normalize the query
    // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
    // convert MIPS to L2 search
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    size_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> PQ chunk centers distances
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);
    // Preprocess Distance b/w Query Vector and Centroids
    //            Chunk 1 | Chunk 2 | Chunk 3
    // Centroid 1  d[1][1]  d[1][2]  d[1][3]
    // Centroid 2
    // Centroid 3
    // Centroid 4
    // Centroid 5
    // Centroid 6
    // Centroid 7
    // Centroid 8

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    std::map<int, float> node_distances;

    // Lambda to batch compute query<->node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists, aligned_query_T, recompute_beighbor_embeddings, data_buf,
                          &node_distances, &total_nodes_requested, &total_nodes_from_cache,
                          dedup_node_dis](const uint32_t *ids, const uint64_t n_ids, float *dists_out) {
        // Vector[0], {3, 6, 2}
        // Distance = d[3][1] + d[6][2] + d[2][3]
        // recompute_beighbor_embeddings = true;
        if (!recompute_beighbor_embeddings)
        {
            diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
            diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
        }
        else
        {
            // Fetch the embeddings from the embedding server using n_ids
            std::vector<uint32_t> node_ids;

            // Update total nodes requested counter
            total_nodes_requested += n_ids;

            // Build a map from node_id to original position for O(1) lookup
            // Handle deduplication if enabled
            std::vector<bool> cached_node_idx(n_ids, false);
            if (dedup_node_dis)
            {
                // First pass: use cached distances where available
                for (size_t i = 0; i < n_ids; i++)
                {
                    if (node_distances.find(ids[i]) != node_distances.end())
                    {
                        // Use cached distance
                        dists_out[i] = node_distances[ids[i]];
                        cached_node_idx[i] = true;
                        total_nodes_from_cache++; // Count cache hits
                    }
                    else
                    {
                        // Not in cache, need to compute
                        node_ids.push_back(ids[i]);
                    }
                }

                // If all distances are cached, we can return early
                if (node_ids.empty())
                    return;
            }
            else
            {
                node_ids = std::vector<uint32_t>(ids, ids + n_ids);
            }

            // Fetch embeddings from the embedding server
            std::vector<std::vector<float>> embeddings;
            bool success = fetch_embeddings_http(node_ids, embeddings);

            if (!success || embeddings.size() != node_ids.size())
            {
                diskann::cout << "Failed to fetch embeddings from the embedding server" << std::endl;
                // Fallback to PQ-based distance computation if fetching fails
                diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
                diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
                return;
            }

            // Preprocess the fetched embeddings to match the format used in diskann
            preprocess_fetched_embeddings(embeddings, this->metric, this->_max_base_norm, this->_data_dim);

            // Compute distances for fetched embeddings
            if (dedup_node_dis)
            {
                // Process each node that needs computation
                uint32_t idx = 0;
                for (size_t i = 0; i < n_ids; i++)
                {
                    if (cached_node_idx[i])
                    {
                        continue;
                    }
                    // Prepare embedding for distance computation
                    embeddings[idx].resize(this->_aligned_dim, 0);
                    memcpy(data_buf, embeddings[idx].data(), this->_aligned_dim * sizeof(T));

                    // Compute distance
                    float distance =
                        this->_dist_cmp->compare(aligned_query_T, data_buf, static_cast<uint32_t>(this->_aligned_dim));

                    // Store results
                    dists_out[i] = distance;
                    node_distances[node_ids[i]] = distance;
                    idx++;
                }
            }
            else
            {
                // Without deduplication, embeddings match the original order
                for (size_t i = 0; i < n_ids; i++)
                {
                    // Prepare embedding for distance computation
                    embeddings[i].resize(this->_aligned_dim, 0);
                    memcpy(data_buf, embeddings[i].data(), this->_aligned_dim * sizeof(T));

                    // Compute distance
                    float distance =
                        this->_dist_cmp->compare(aligned_query_T, data_buf, static_cast<uint32_t>(this->_aligned_dim));

                    // Store results
                    dists_out[i] = distance;
                }
            }
        }
    };

    // Add logic of global pruning
    // Using a priority queue to record the PQ distance - use min heap for nearest neighbors
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        aq_priority_queue;
    tsl::robin_set<size_t> &visited = query_scratch->visited;

    // TODO: implement this function
    // 1. Based on some heristic to prune the node_nbrs and nnbrs that is not promising
    // 1.1 heruistic 1: use higher compression PQ to prune the node_nbrs and nnbrs that is not promising in path
    // /powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/compressed_2/
    // 1.2 heruistic 2: use a lightweight reranker to rerank the node_nbrs and nnbrs that is not promising
    auto prune_node_nbrs = [this, pq_coord_scratch, pq_dists, recompute_beighbor_embeddings, dedup_node_dis,
                            prune_ratio, global_pruning, &aq_priority_queue,
                            &visited](uint32_t *&node_nbrs, uint64_t &nnbrs) {
        if (!recompute_beighbor_embeddings)
        {
            return;
        }
        if (nnbrs <= 10)
        {
            // Don't prune if there are very few neighbors
            return;
        }

        // Allocate space for distance calculations
        float *dists_out = new float[nnbrs];

        // Compute distances using PQ directly instead of compute_dists
        diskann::aggregate_coords(node_nbrs, nnbrs, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, nnbrs, this->_n_chunks, pq_dists, dists_out);

        if (global_pruning)
        {
            // Add the distance and node_id to the priority queue
            for (uint64_t i = 0; i < nnbrs; i++)
            {
                aq_priority_queue.push(std::make_pair(dists_out[i], node_nbrs[i]));
            }
            // select all ratio=prune_ratio in aq_priority_queue but need to check if the node_id is already visited,
            // dont need to pop
            std::vector<std::pair<float, uint32_t>> promising_nodes;

            std::vector<std::pair<float, uint32_t>> roll_back_nodes;
            // 1. visit top prune_ratio*length of aq_priority_queue nodes in aq_priority_queue and put the node_id not
            // visited into a vector
            uint64_t original_size = aq_priority_queue.size();
            for (uint64_t i = 0; i < prune_ratio * original_size; i++)
            {
                auto top_node = aq_priority_queue.top();
                roll_back_nodes.push_back(top_node);
                aq_priority_queue.pop();
                if (visited.find(top_node.second) == visited.end())
                {
                    float distance = top_node.first;
                    uint32_t node_id = top_node.second;
                    promising_nodes.push_back(std::make_pair(distance, node_id));
                }
            }
            // push all roll_back_nodes back to aq_priority_queue
            for (uint64_t i = 0; i < roll_back_nodes.size(); i++)
            {
                aq_priority_queue.push(roll_back_nodes[i]);
            }

            // 2. assing the node_id and distance to node_nbrs and nnbrs
            for (uint64_t i = 0; i < promising_nodes.size(); i++)
            {
                node_nbrs[i] = promising_nodes[i].second;
            }
            nnbrs = promising_nodes.size();
            // then return corresponding node_nbrs and nnbrs

            delete[] dists_out;
            return;
        }
        // Create a vector of pairs (node_id, distance)
        std::vector<std::pair<uint32_t, float>> scored_nbrs;
        scored_nbrs.reserve(nnbrs);

        for (uint64_t i = 0; i < nnbrs; i++)
        {
            scored_nbrs.emplace_back(node_nbrs[i], dists_out[i]);
        }

        // Sort by distance (lower is better)
        std::sort(scored_nbrs.begin(), scored_nbrs.end(),
                  [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
                      return a.second < b.second;
                  });

        // Keep only the top portion of neighbors based on prune_ratio (or at least 10)
        uint64_t new_nnbrs = std::max<uint64_t>(10UL, static_cast<uint64_t>(nnbrs * prune_ratio));
        if (new_nnbrs < nnbrs)
        {
            // Update the original node_nbrs array with pruned neighbors
            for (uint64_t i = 0; i < new_nnbrs; i++)
            {
                node_nbrs[i] = scored_nbrs[i].first;
            }

            // Update the count of neighbors
            nnbrs = new_nnbrs;
        }

        // Free the allocated memory
        delete[] dists_out;
    };
    Timer query_timer, io_timer, cpu_timer;

    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;
    std::vector<T *> points_to_compute; // Store points for later embedding computation

#if 0
    std::vector<Neighbor> exact_dist_retset;
    std::vector<std::vector<float>> exact_embeddings;
#endif

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        if (_filter_to_medoid_ids.find(filter_label) != _filter_to_medoid_ids.end())
        {
            const auto &medoid_ids = _filter_to_medoid_ids[filter_label];
            for (uint64_t cur_m = 0; cur_m < medoid_ids.size(); cur_m++)
            {
                // for filtered index, we dont store global centroid data as for unfiltered index, so we use PQ distance
                // as approximation to decide closest medoid matching the query filter.
                compute_dists(&medoid_ids[cur_m], 1, dist_scratch);
                float cur_expanded_dist = dist_scratch[0];
                if (cur_expanded_dist < best_dist)
                {
                    best_medoid = medoid_ids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        }
        else
        {
            throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    float *batched_dists = nullptr;
    if (batch_recompute)
    {
        batched_dists = new float[_max_degree * beam_width + 5];
    }

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            auto iter = _nhood_cache.find(nbr.id);
            if (iter != _nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
            if (this->_count_visited_nodes)
            {
                reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
            }
        }

        std::vector<AlignedRead> graph_read_reqs;
        std::map<uint32_t, int> node_offsets; // id -> offset
        std::map<uint32_t, std::vector<uint32_t>> node_nbrs_ori;
        std::map<uint32_t, std::vector<float>> node_cords;

        // read nhoods of frontier ids
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;

            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
#if 1
                if (!_use_partition)
                {
#endif
                    frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                                                    num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
#if 1
                }
#endif
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }

            if (_use_partition)
            {
                sector_scratch_idx = 0;
                for (auto &frontier_nhood : frontier_nhoods)
                {
                    uint32_t node_id = frontier_nhood.first;
                    uint32_t partition_id = _id2partition[node_id];
                    if (partition_id >= _num_partitions)
                    {
                        diskann::cout << "Warning: partition_id is invalid: " << partition_id << std::endl;
                        assert(false);
                    }

                    std::vector<uint32_t> part_list = _graph_partitions[partition_id];
                    auto it = std::find(part_list.begin(), part_list.end(), node_id);
                    if (it == part_list.end())
                    {
                        diskann::cerr << "Error: node " << node_id << " not found in partition " << partition_id
                                      << std::endl;
                        assert(false);
                    }
                    size_t j = std::distance(part_list.begin(), it);
                    node_offsets[node_id] = j;

                    uint64_t sector_offset = (partition_id + 1) * defaults::SECTOR_LEN;
                    // ! Keep it same with frontier_nhood.second
                    char *sector_buffer = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;

                    AlignedRead partition_read;
                    partition_read.len = defaults::SECTOR_LEN;
                    partition_read.buf = sector_buffer;
                    partition_read.offset = sector_offset;

                    graph_read_reqs.emplace_back(partition_read);
                }
            }

            io_timer.reset();
#if 1
            if (!_use_partition)
            {
#endif
#ifdef USE_BING_INFRA
                reader->read(frontier_read_reqs, ctx,
                             true); // asynhronous reader for Bing.
#else
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
#endif
#if 1
            }
#endif

#if 0
            for (auto &[node_id, disk_buf] : frontier_nhoods)
            {
                char *node_disk_buf = offset_to_node(disk_buf, node_id);
                uint32_t *nhood_buf = offset_to_node_nhood(node_disk_buf);
                uint32_t neighbor_count = *nhood_buf;
                node_nbrs_ori[node_id] = std::vector<uint32_t>(nhood_buf + 1, nhood_buf + 1 + neighbor_count);
                node_cords[node_id] =
                    std::vector<float>(offset_to_node_coords(node_disk_buf),
                                       offset_to_node_coords(node_disk_buf) + _disk_bytes_per_point / sizeof(float));
            }
#endif
            if (_use_partition)
            {
                graph_reader->read(graph_read_reqs, ctx);
            }

            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            auto global_cache_iter = _coord_cache.find(cached_nhood.first);
            uint32_t node_id = cached_nhood.first;
            T *node_fp_coords_copy = global_cache_iter->second;
            float cur_expanded_dist;
            float exact_expanded_dist = 0;

            if (skip_search_reorder)
            {
                compute_dists(&node_id, 1, dist_scratch);
                cur_expanded_dist = dist_scratch[0];
            }
            else if (USE_DEFERRED_FETCH)
            {
                cur_expanded_dist = 0.0f;
            }
            else if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                        query_float, (uint8_t *)node_fp_coords_copy);
            }
            full_retset.push_back(Neighbor(node_id, cur_expanded_dist));

#if 0
            if (!_use_disk_index_pq)
            {
                exact_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    exact_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    exact_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)node_fp_coords_copy);
            }
            exact_dist_retset.push_back(Neighbor(node_id, exact_expanded_dist));
            exact_embeddings.push_back(std::vector<float>(node_fp_coords_copy, node_fp_coords_copy + _aligned_dim));
#endif

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are
        // reads in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(reader, ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
        std::vector<uint32_t> batched_node_ids;

        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            uint32_t node_id = frontier_nhood.first;
            char *disk_buf = frontier_nhood.second;
            char *node_disk_buf = offset_to_node(disk_buf, node_id);

            float cur_expanded_dist;

            // If skip_reorder is true, compute both PQ distance and exact distance
            if (skip_search_reorder)
            {
                compute_dists(&node_id, 1, dist_scratch);
                cur_expanded_dist = dist_scratch[0];
            }
            else if (USE_DEFERRED_FETCH)
            {
                cur_expanded_dist = 0.0f;
            }
            else if (recompute_beighbor_embeddings && dedup_node_dis && _use_partition)
            {
                // For _use_partition = True, we must rely on node_distances to get the distance
                // Since we are using graph-structure only reading.
                // ! Use node_distances to get the distance
                cur_expanded_dist = node_distances[node_id];
            }
            else
            {
#if 0
                if (node_cords.find(node_id) == node_cords.end())
                {
                    diskann::cout << "Warning: node " << node_id << " not found in node_cords" << std::endl;
                    diskann::cout << "Are you using deferred fetch for detached graph?" << std::endl;
                    assert(false);
                }
                // ! As for DEBUG mode and partition_read = True, we are overriding the node_disk_buf
                // ! with our graph-structure only reading. So we need to use node_cords to get the correct
                // ! coordinates.
                T *node_fp_coords = reinterpret_cast<T *>(node_cords[node_id].data());
                // T *node_fp_coords = offset_to_node_coords(node_disk_buf);
#endif
                T *node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                if (!_use_disk_index_pq)
                {
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                }
                else
                {
                    if (metric == diskann::Metric::INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
                }
            }
            full_retset.push_back(Neighbor(node_id, cur_expanded_dist));

#if 0
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float exact_expanded_dist = 0;
            if (!_use_disk_index_pq)
            {
                exact_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    exact_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    exact_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            exact_dist_retset.push_back(Neighbor(node_id, exact_expanded_dist));
            exact_embeddings.push_back(std::vector<float>(data_buf, data_buf + _aligned_dim));
#endif

            uint32_t *node_nbrs;
            uint64_t nnbrs;

            if (!_use_partition)
            {
                auto node_buf = offset_to_node_nhood(node_disk_buf);
                nnbrs = (uint64_t)(*node_buf);
                node_nbrs = (node_buf + 1);
            }

#if 0
            auto node_nbrs_vec = node_nbrs_ori[node_id];
            nnbrs = node_nbrs_vec.size();
            node_nbrs = node_nbrs_vec.data();
#endif
            if (_use_partition)
            {
                char *sector_buffer = frontier_nhood.second;
                int j = node_offsets[node_id];
                uint64_t node_offset = j * _graph_node_len;
                if (node_offset + 4 > defaults::SECTOR_LEN)
                {
                    diskann::cerr << "Error: node offset out of range: " << node_offset << " (+4) > "
                                  << defaults::SECTOR_LEN << " for node " << node_id << std::endl;
                    assert(false);
                }

                char *adjacency_ptr = sector_buffer + node_offset;
                uint32_t neighbor_count = *reinterpret_cast<uint32_t *>(adjacency_ptr);

                if (neighbor_count > 10000)
                {
                    diskann::cerr << "Error: suspicious neighbor count: " << neighbor_count << " for node " << node_id
                                  << std::endl;
                    assert(false);
                }

                size_t needed = neighbor_count * sizeof(uint32_t);
                if (node_offset + 4 + needed > defaults::SECTOR_LEN)
                {
                    diskann::cerr << "Error: neighbor data out of range: " << (node_offset + 4 + needed) << " > "
                                  << defaults::SECTOR_LEN << " for node " << node_id << std::endl;
                    assert(false);
                }

#if 0
                if (neighbor_count != nnbrs)
                {
                    diskann::cout << "Warning: neighbor_count != nnbrs: " << neighbor_count << " != " << nnbrs
                                  << std::endl;
                    assert(false);
                }
#endif

                nnbrs = neighbor_count;

#if 0
                uint32_t *our_node_nbrs = (uint32_t *)(adjacency_ptr + 4);
                for (uint32_t i = 0; i < nnbrs; i++)
                {
                    if (our_node_nbrs[i] != node_nbrs[i])
                    {
                        diskann::cout << "Warning: our_node_nbrs[" << i << "] != node_nbrs[" << i
                                      << "]: " << our_node_nbrs[i] << " != " << node_nbrs[i] << std::endl;
                        assert(false);
                    }
                }
#endif

                node_nbrs = reinterpret_cast<uint32_t *>(adjacency_ptr + 4);
            }

            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            // have a function to prune the node_nbrs and nnbrs

            // prune_node_nbrs(node_nbrs, nnbrs);

            if (!batch_recompute)
            {
                prune_node_nbrs(node_nbrs, nnbrs);
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr)
                {
                    stats->n_cmps += (uint32_t)nnbrs;
                    stats->cpu_us += (float)cpu_timer.elapsed();
                }

                cpu_timer.reset();
                // process prefetch-ed nhood
                for (uint64_t m = 0; m < nnbrs; ++m)
                {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second)
                    {
                        if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                            continue;

                        if (use_filter && !(point_has_label(id, filter_label)) &&
                            (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                            continue;
                        cmps++;
                        float dist = dist_scratch[m];
                        if (stats != nullptr)
                        {
                            stats->n_cmps++;
                        }

                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }

                if (stats != nullptr)
                {
                    stats->cpu_us += (float)cpu_timer.elapsed();
                }
            }
            else
            {
                // add all the node_nbrs to the batch_requests
                batched_node_ids.insert(batched_node_ids.end(), node_nbrs, node_nbrs + nnbrs);
            }
        }

        if (batch_recompute)
        {
            auto nnbrs = batched_node_ids.size();
            uint32_t *batched_data_ptr = batched_node_ids.data(); // Get pointer to data
            prune_node_nbrs(batched_data_ptr, nnbrs);             // Prune using the pointer, nnbrs is updated

            compute_dists(batched_data_ptr, nnbrs, batched_dists); // Compute dists for the pruned set
            // ! Not sure if dist_scratch has enough space

            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = batched_node_ids[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = batched_dists[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
        // }
        // }
        hops++;
    }

    delete[] batched_dists;

    diskann::cout << "Graph traversal completed, hops: " << hops << std::endl;

    if (USE_DEFERRED_FETCH)
    {
        diskann::cout << "hops: " << hops << std::endl;

        std::vector<uint32_t> node_ids;
        node_ids.reserve(full_retset.size());
        for (auto &nr : full_retset)
        {
            node_ids.push_back(nr.id);
        }

        Timer fetch_timer;
        std::vector<std::vector<float>> real_embeddings;
        bool success = fetch_embeddings_http(node_ids, real_embeddings);
        if (!success)
        {
            throw ANNException("Failed to fetch embeddings", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        diskann::cout << "Fetched " << real_embeddings.size() << " embeddings in " << fetch_timer.elapsed() << " us"
                      << std::endl;

        // compute real-dist
        Timer compute_timer;
        // preprocess the real embedding to match the format of  nomarlized version of diskann
        preprocess_fetched_embeddings(real_embeddings, metric, _max_base_norm, this->_data_dim);

#if 0
        assert(real_embeddings.size() == full_retset.size());
        assert(real_embeddings.size() == exact_dist_retset.size());
        assert(real_embeddings.size() == exact_embeddings.size());
#endif

        for (int i = 0; i < real_embeddings.size(); i++)
        {
            // padding real_embeddings[i] to _aligned_dim
            real_embeddings[i].resize(_aligned_dim, 0);
#if 0
            // compare real_embeddings[i] with exact_embeddings[i]
            if (real_embeddings[i].size() != exact_embeddings[i].size())
            {
                diskann::cout << "real_embeddings[i].size(): " << real_embeddings[i].size() << std::endl;
                diskann::cout << "exact_embeddings[i].size(): " << exact_embeddings[i].size() << std::endl;

                // dumping to files
                std::ofstream diff_file("./diff_embeddings.txt");
                diff_file << "real_embeddings[i].size(): " << real_embeddings[i].size() << std::endl;
                diff_file << "exact_embeddings[i].size(): " << exact_embeddings[i].size() << std::endl;
                for (int j = 0; j < real_embeddings[i].size(); j++)
                {
                    diff_file << real_embeddings[i][j] << " ";
                }
                diff_file << std::endl;
                for (int j = 0; j < exact_embeddings[i].size(); j++)
                {
                    diff_file << exact_embeddings[i][j] << " ";
                }
                diff_file << std::endl;
                assert(false);
            }
            for (int j = 0; j < real_embeddings[i].size(); j++)
            {
                if (abs(real_embeddings[i][j] - exact_embeddings[i][j]) > 5e-4)
                {
                    diskann::cout << "Difference found at node_id: " << full_retset[i].id << " and dimension: " << j
                                  << std::endl;
                    diskann::cout << "real_embeddings[i][j]: " << real_embeddings[i][j] << std::endl;
                    diskann::cout << "exact_embeddings[i][j]: " << exact_embeddings[i][j] << std::endl;
                    assert(false);
                }
            }
#endif

            float dist;
            assert(!_use_disk_index_pq);
            memcpy(data_buf, real_embeddings[i].data(), real_embeddings[0].size() * sizeof(T));
            dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);

            full_retset[i].distance = dist;

#if 0
            if (abs(dist - exact_dist_retset[i].distance) > 5e-4)
            {
                diskann::cout << "Difference found at node_id: " << full_retset[i].id << std::endl;
                diskann::cout << "dist: " << dist << std::endl;
                diskann::cout << "exact_dist_retset[i].distance: " << exact_dist_retset[i].distance << std::endl;
                assert(false);
            }
#endif
        }
        diskann::cout << "compute_timer.elapsed(): " << compute_timer.elapsed() << std::endl;
    }

    std::sort(full_retset.begin(), full_retset.end());

// Compare PQ results with exact results when skip_search_reorder is true
#if 0
    if (skip_search_reorder)
    {
        // Sort the exact distance results
        std::sort(exact_dist_retset.begin(), exact_dist_retset.end());

        // Create a map to find positions of IDs in the PQ-sorted list
        std::unordered_map<uint32_t, size_t> pq_positions;
        for (size_t i = 0; i < full_retset.size(); i++)
        {
            pq_positions[full_retset[i].id] = i;
        }

        int current_search_id = search_counter.fetch_add(1);
        int thread_id = omp_get_thread_num();

        std::lock_guard<std::mutex> lock(log_file_mutex);

        std::ofstream log_file("./top3_positions_log.txt", std::ios::app);
        // Write header if file is empty
        log_file.seekp(0, std::ios::end);
        if (log_file.tellp() == 0)
        {
            diskann::cout << "Saved top3 distributions to " << std::filesystem::canonical("./top3_positions_log.txt")
                          << std::endl;
            log_file << "Search#,ThreadID,FullSetSize,Rank,ID,PQ_Rank,PQ_Distance,Exact_Distance" << std::endl;
        }

        // Log the top-k results from exact distance sorting and their positions in PQ-sorted list
        size_t top_k = std::min((size_t)k_search, exact_dist_retset.size());
        for (size_t i = 0; i < top_k; i++)
        {
            uint32_t id = exact_dist_retset[i].id;
            float exact_dist = exact_dist_retset[i].distance;

            // Find this ID's position in the PQ-sorted list
            size_t pq_pos = pq_positions.count(id) ? pq_positions[id] : full_retset.size();
            float pq_dist = (pq_pos < full_retset.size()) ? full_retset[pq_pos].distance : -1;

            log_file << current_search_id << "," << thread_id << "," << full_retset.size() << "," << i + 1 << "," << id
                     << "," << pq_pos + 1 << "," << pq_dist << "," << exact_dist << std::endl;
        }

        log_file.close();
    }
#endif

    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            // MULTISECTORFIX
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                       defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, true); // async reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            // MULTISECTORFIX
            auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }

    // After search is complete, print cache hit rate statistics
    if (recompute_beighbor_embeddings && dedup_node_dis && total_nodes_requested > 0)
    {
        float cache_hit_rate = static_cast<float>(total_nodes_from_cache) / total_nodes_requested * 100.0f;
        diskann::cout << "Node distance cache statistics:" << std::endl;
        diskann::cout << "  Total nodes requested: " << total_nodes_requested << std::endl;
        diskann::cout << "  Nodes served from cache: " << total_nodes_from_cache << std::endl;
        diskann::cout << "  Cache hit rate: " << cache_hit_rate << "%" << std::endl;
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
