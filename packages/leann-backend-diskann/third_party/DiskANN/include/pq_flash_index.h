// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

template <typename T, typename LabelT = uint32_t> class PQFlashIndex
{
  public:
    DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                   std::shared_ptr<AlignedFileReader> &graphReader,
                                   diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix,
                               const char *pq_prefix = nullptr);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix, const char *pq_prefix = nullptr,
                               const char *partition_prefix = nullptr);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath, const char *graph_file);
#else
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath,
                                                   const char *graph_file, const char *partition_file);
#endif

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                   uint64_t l_search, uint64_t beamwidth,
                                                                   uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                   std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                   uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                   uint32_t num_threads,
                                                                   std::vector<uint32_t> &node_list);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                            const bool shuffle = false);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr,
                                              const bool USE_DEFERRED_FETCH = false,
                                              const bool skip_search_reorder = false,
                                              const bool recompute_beighbor_embeddings = false,
                                              const bool dedup_node_dis = false, float prune_ratio = 0,
                                              const bool batch_recompute = false, bool global_pruning = false);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr,
                                              const bool USE_DEFERRED_FETCH = false,
                                              const bool skip_search_reorder = false,
                                              const bool recompute_beighbor_embeddings = false,
                                              const bool dedup_node_dis = false, float prune_ratio = 0,
                                              const bool batch_recompute = false, bool global_pruning = false);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr, const bool USE_DEFERRED_FETCH = false,
                                              const bool skip_search_reorder = false,
                                              const bool recompute_beighbor_embeddings = false,
                                              const bool dedup_node_dis = false, float prune_ratio = 0,
                                              const bool batch_recompute = false, bool global_pruning = false);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr, const bool USE_DEFERRED_FETCH = false,
                                              const bool skip_search_reorder = false,
                                              const bool recompute_beighbor_embeddings = false,
                                              const bool dedup_node_dis = false, float prune_ratio = 0,
                                              const bool batch_recompute = false, bool global_pruning = false);

    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &filter_label);

    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT uint64_t get_data_dim();

    std::shared_ptr<AlignedFileReader> &reader;

    DISKANN_DLLEXPORT diskann::Metric get_metric();

    //
    // node_ids: input list of node_ids to be read
    // coord_buffers: pointers to pre-allocated buffers that coords need to copied to. If null, dont copy.
    // nbr_buffers: pre-allocated buffers to copy neighbors into
    //
    // returns a vector of bool one for each node_id: true if read is success, else false
    //
    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

    DISKANN_DLLEXPORT std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);
    DISKANN_DLLEXPORT uint64_t get_num_points();

  protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

  private:
    DISKANN_DLLEXPORT inline bool point_has_label(uint32_t point_id, LabelT label_id);
    std::unordered_map<std::string, LabelT> load_label_map(std::basic_istream<char> &infile);
    DISKANN_DLLEXPORT void parse_label_file(std::basic_istream<char> &infile, size_t &num_pts_labels);
    DISKANN_DLLEXPORT void get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                   uint32_t &num_total_labels);
    DISKANN_DLLEXPORT void generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                  const uint32_t nthreads);
    void reset_stream_for_reading(std::basic_istream<char> &infile);

    // sector # on disk where node_id is present with in the graph part
    DISKANN_DLLEXPORT uint64_t get_node_sector(uint64_t node_id);

    // ptr to start of the node
    DISKANN_DLLEXPORT char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    DISKANN_DLLEXPORT uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    DISKANN_DLLEXPORT int load_graph_index(const std::string &graph_index_file);

    DISKANN_DLLEXPORT int read_partition_info(const std::string &partition_bin);

    DISKANN_DLLEXPORT int read_neighbors(const std::string &graph_index_file, uint64_t target_node_id);

    // index info for multi-node sectors
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    //
    // index info for multi-sector nodes
    // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
    // offset in sector: [0]
    //
    // Common info
    // coords start at ofsset
    // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
    // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)

    uint64_t _max_node_len = 0;
    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    uint64_t _max_degree = 0;
    uint64_t _C = 0;
    // Data used for searching with re-order vectors
    uint64_t _ndims_reorder_vecs = 0;
    uint64_t _reorder_data_start_sector = 0;
    uint64_t _nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float _max_base_norm = 0.0f;

    // data info
    uint64_t _num_points = 0;
    uint64_t _num_frozen_points = 0;
    uint64_t _frozen_location = 0;
    uint64_t _data_dim = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _disk_bytes_per_point = 0; // Number of bytes

    std::string _disk_index_file;
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;

    // PQ data
    // _n_chunks = # of chunks ndims is split into
    // data: char * _n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
    uint8_t *data = nullptr;
    uint64_t _n_chunks;
    FixedChunkPQTable _pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool _use_disk_index_pq = false;
    uint64_t _disk_pq_n_chunks = 0;
    FixedChunkPQTable _disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *_medoids = nullptr;
    // defaults to 1
    size_t _num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *_centroid_data = nullptr;

    // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
    T *_coord_cache_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;

    // thread-specific scratch
    ConcurrentQueue<SSDThreadData<T> *> _thread_data;
    uint64_t _max_nthreads;
    bool _load_flag = false;
    bool _count_visited_nodes = false;
    bool _reorder_data_exists = false;
    uint64_t _reoreder_data_offset = 0;

    // filter support
    uint32_t *_pts_to_label_offsets = nullptr;
    uint32_t *_pts_to_label_counts = nullptr;
    LabelT *_pts_to_labels = nullptr;
    std::unordered_map<LabelT, std::vector<uint32_t>> _filter_to_medoid_ids;
    bool _use_universal_label = false;
    LabelT _universal_filter_label;
    tsl::robin_set<uint32_t> _dummy_pts;
    tsl::robin_set<uint32_t> _has_dummy_pts;
    tsl::robin_map<uint32_t, uint32_t> _dummy_to_real_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;

  private:
    bool _use_partition = false;

    std::shared_ptr<AlignedFileReader> graph_reader; // Graph file reader
    std::string _graph_index_file;                   // Graph file path
    uint64_t _graph_node_len;                        // Graph node length
    uint64_t _emb_node_len;                          // Embedding node length

    // Partition related data structures
    uint64_t _num_partitions;                             // Number of partitions
    std::vector<std::vector<uint32_t>> _graph_partitions; // Partition information
    std::vector<uint32_t> _id2partition;                  // ID to partition mapping

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = defaults::SECTOR_LEN;
    char *getHeaderBytes();
#endif
};
} // namespace diskann
