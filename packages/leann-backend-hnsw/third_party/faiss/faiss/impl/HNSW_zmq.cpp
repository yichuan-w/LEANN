/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>

#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

#include <fcntl.h>
#include <msgpack.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <sstream> // For msgpack stringstream buffer
#include <thread>

#include "HNSW_zmq.h"

namespace faiss {

namespace {
std::string experimental_disk_storage_path;
off_t experimental_disk_data_offset;
int experimental_block_size;
std::vector<bool> experimental_is_in_top_degree_set;
} // namespace

void setup_experimental_top_degree_disk_read(
        const std::string& degree_path,
        float top_percent,
        const std::string& storage_path,
        off_t data_offset,
        idx_t ntotal) {
    FAISS_THROW_IF_NOT_FMT(
            data_offset >= 0, "Data offset (%ld) invalid.", data_offset);
    FAISS_THROW_IF_NOT_MSG(
            top_percent >= 0.0f && top_percent <= 100.0f,
            "top_percent invalid.");
    FAISS_THROW_IF_NOT_FMT(
            ntotal > 0, "ntotal (%ld) must be positive.", ntotal);

    // 3. Determine threshold and build the boolean set
    if (abs(top_percent - 0.0f) <= 1e-6) {
        return;
    }

    // 1. Get block size
    struct stat file_stat;
    int block_size = 4096;
    if (stat(storage_path.c_str(), &file_stat) == 0) {
        block_size = (file_stat.st_blksize > 0) ? file_stat.st_blksize : 4096;
    } else {
        // Fail fast on stat error as block size is critical for O_DIRECT
        FAISS_THROW_FMT(
                "Setup Error: Cannot stat storage file %s: %s",
                storage_path.c_str(),
                strerror(errno));
    }
    FAISS_THROW_IF_NOT(block_size > 0);

    // 2. Load degree distribution temporarily
    std::vector<int> degrees;
    std::ifstream degree_file(degree_path);
    if (degree_file.is_open()) {
        std::string line;
        if (ntotal > 0)
            degrees.reserve(ntotal);
        while (std::getline(degree_file, line)) {
            if (line.empty())
                continue;
            try {
                degrees.push_back(std::stoi(line));
            } catch (...) { /* ignore */
            }
        }
        degree_file.close();
    } else {
        FAISS_THROW_FMT(
                "Setup Error: Degree file not found: %s", degree_path.c_str());
    }
    FAISS_THROW_IF_NOT_FMT(
            !degrees.empty(), "Degree file %s empty.", degree_path.c_str());
    if ((idx_t)degrees.size() != ntotal) {
        FAISS_THROW_FMT(
                "Setup Error: Degree file size (%zu) != ntotal (%ld).",
                degrees.size(),
                ntotal);
    }

    std::vector<int> sorted_degrees = degrees;
    std::sort(
            sorted_degrees.begin(), sorted_degrees.end(), std::greater<int>());

    float percentile = top_percent / 100.0f;
    size_t threshold_idx = std::min(
            (size_t)(sorted_degrees.size() * percentile) - 1,
            sorted_degrees.size() - 1); // Clamp index

    int degree_threshold =
            sorted_degrees[threshold_idx]; // Store threshold for info

    std::vector<bool> is_in_top_degree_set(ntotal, false);
    size_t top_node_count = 0;
    for (idx_t i = 0; i < ntotal; ++i) {
        if (degrees[i] >= degree_threshold) {
            is_in_top_degree_set[i] = true;
            top_node_count++;
        }
    }
    experimental_disk_storage_path = storage_path;
    experimental_disk_data_offset = data_offset;
    experimental_block_size = block_size;
    experimental_is_in_top_degree_set = is_in_top_degree_set;

    printf("ZmqDC Setup: Disk logic Top %.2f%% (deg>=%d). %zu nodes. Offset=%ld, BlkSize=%d\n",
           top_percent,
           degree_threshold,
           top_node_count,
           data_offset,
           block_size);
}

bool fetch_embeddings_zmq(
        const std::vector<uint32_t>& node_ids,
        std::vector<std::vector<float>>& out_embeddings,
        int zmq_port = 5557); // Default port kept

float read_disk_and_compute_local_ip(idx_t i, size_t d, const float* query) {
    size_t vector_bytes = d * sizeof(float);
    std::vector<float> vec_buffer(d);
    void* aligned_buffer = nullptr;
    size_t block_size = experimental_block_size;

    int fd;
#ifdef __linux__
    // O_DIRECT is Linux-specific
    fd =
            open(experimental_disk_storage_path.c_str(),
                 O_RDONLY | O_CLOEXEC | O_DIRECT);
#else
    // macOS doesn't have O_DIRECT
    fd = open(experimental_disk_storage_path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    if (fd == -1) {
        assert(false);
    }

    off_t desired_start =
            experimental_disk_data_offset + (off_t)i * vector_bytes;
    off_t desired_end = desired_start + vector_bytes;
    off_t aligned_start = (desired_start / block_size) * block_size;
    off_t aligned_end =
            ((desired_end + block_size - 1) / block_size) * block_size;
    size_t bytes_to_read_aligned = aligned_end - aligned_start;

    assert(bytes_to_read_aligned > 0);
    if (posix_memalign(&aligned_buffer, block_size, bytes_to_read_aligned) !=
        0) {
        assert(false);
    }

    ssize_t bytes_read =
            pread(fd, aligned_buffer, bytes_to_read_aligned, aligned_start);
    close(fd);

    assert(bytes_read == (ssize_t)bytes_to_read_aligned);
    size_t internal_offset = desired_start - aligned_start;
    // Check bounds before memcpy
    assert(internal_offset + vector_bytes <= bytes_to_read_aligned);
    memcpy(vec_buffer.data(),
           (char*)aligned_buffer + internal_offset,
           vector_bytes);
    // For IP metric, we need to return negative IP
    // fetch embeddings to check
    // std::vector<std::vector<float>> out_embeddings;
    // bool success = fetch_embeddings_zmq(
    //         std::vector<uint32_t>{(uint32_t)i}, out_embeddings, 5557);
    // assert(success);
    // assert(out_embeddings.size() == 1);
    // assert(out_embeddings[0].size() == d);
    // for (auto i = 0; i < out_embeddings.size(); i++) {
    //     for (auto j = 0; j < out_embeddings[i].size(); j++) {
    //         if (abs(out_embeddings[i][j] - vec_buffer[j]) > 1e-3) {
    //             printf("ERROR: disk and remote fetched embeddings mismatch
    //             for id %ld, error: %f\n",
    //                    i,
    //                    abs(out_embeddings[i][j] - vec_buffer[j]));
    //         }
    //     }
    //     printf("\n");
    // }
    float distance = -fvec_inner_product(query, vec_buffer.data(), d);
    free(aligned_buffer);
    return distance;
}

// --- MessagePack Data Structures (Define simple structs for serialization) ---
struct EmbeddingRequestMsgpack {
    std::vector<uint32_t> node_ids;
    MSGPACK_DEFINE_ARRAY(node_ids); // Use array format [ [ids] ]
};

struct EmbeddingResponseMsgpack {
    // Store dimensions as separate fields for clarity with msgpack map
    // Or keep as vector [batch_size, dim] if using array format
    // Let's use array format for simplicity matching MSGPACK_DEFINE_ARRAY
    std::vector<uint32_t> dimensions; // [batch_size, embedding_dim]
    // Store flat embedding data as raw bytes or vector<float>
    // Using vector<float> is easier to handle with msgpack-c directly
    std::vector<float>
            embeddings_data; // Flattened [batch_size * embedding_dim]
    // Optional: Add missing_ids if needed
    // std::vector<uint32_t> missing_ids;

    MSGPACK_DEFINE_ARRAY(dimensions, embeddings_data); // [ [dims], [data] ]
};

struct DistanceRequestMsgpack {
    std::vector<uint32_t> node_ids;
    std::vector<float> query_vector;
    MSGPACK_DEFINE_ARRAY(node_ids, query_vector); // [ [ids], [query_vector] ]
};

struct DistanceResponseMsgpack {
    std::vector<float> distances;    // Direct distances between query and nodes
    MSGPACK_DEFINE_ARRAY(distances); // [ [distances] ]
};

// --- ZMQ Fetch Function (Using MessagePack) ---
bool fetch_embeddings_zmq(
        const std::vector<uint32_t>& node_ids,
        std::vector<std::vector<float>>& out_embeddings,
        int zmq_port) // Default port kept
{
    EmbeddingRequestMsgpack req_msgpack;
    req_msgpack.node_ids = node_ids;

    std::stringstream buffer;
    try {
        msgpack::pack(buffer, req_msgpack);
    } catch (const std::exception& e) {
        std::cerr << "MessagePack pack failed: " << e.what() << std::endl;
        return false;
    }
    std::string req_str = buffer.str();

    void* context = zmq_ctx_new();
    if (!context) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_ctx_new failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        return false;
    }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_socket failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_ctx_destroy(context);
        return false;
    }
    int timeout = 30000;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_connect failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (zmq_send(socket, req_str.data(), req_str.size(), 0) < 0) { /*...*/
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_msg_recv failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    zmq_msg_t response;
    zmq_msg_init(&response);
    if (zmq_msg_recv(&response, socket, 0) < 0) { /*...*/
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_msg_recv failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    EmbeddingResponseMsgpack resp_msgpack;
    const char* resp_data = static_cast<const char*>(zmq_msg_data(&response));
    size_t resp_size = zmq_msg_size(&response);
    // printf("[fetch_zmq] Raw response bytes (first %d): ",
    //        (int)std::min((size_t)64, resp_size));

    msgpack::object_handle oh = msgpack::unpack(resp_data, resp_size);
    msgpack::object obj = oh.get();
    obj.convert(resp_msgpack); // Convert msgpack object to our struct
    // for (size_t k = 0; k < std::min((size_t)64, resp_size); ++k)
    //     printf("%02x ", (unsigned char)resp_data[k]);
    // printf("\n");

    // --- Print parsed values BEFORE NaN check ---
    // printf("[fetch_zmq] Parsed response. Dimensions: %d x %d. Data floats:
    // %zu\n",
    //        resp_msgpack.dimensions.empty() ? 0 : resp_msgpack.dimensions[0],
    //        resp_msgpack.dimensions.size() < 2 ? 0 :
    //        resp_msgpack.dimensions[1], resp_msgpack.embeddings_data.size());
    // printf("[fetch_zmq] Parsed embeddings_data (first %d floats): ",
    //        (int)std::min((size_t)10, resp_msgpack.embeddings_data.size()));
    // bool parse_contains_nan = false;
    // for (size_t k = 0;
    //      k < std::min((size_t)10, resp_msgpack.embeddings_data.size());
    //      ++k) {
    //     printf("%.6f ", resp_msgpack.embeddings_data[k]);
    //     if (std::isnan(resp_msgpack.embeddings_data[k]))
    //         parse_contains_nan = true;
    // }
    // printf("%s\n",
    //        parse_contains_nan ? "!!! CONTAINS NaN AFTER PARSE !!!"
    //                           : "(Checked first 10 for NaN)");

    if (resp_msgpack.dimensions.size() != 2) {
        // std::cerr << "Server response has invalid dimensions size: "
        //           << resp_msgpack.dimensions.size() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }
    int batch_size = resp_msgpack.dimensions[0];
    int embedding_dim = resp_msgpack.dimensions[1];

    // Handle empty response
    if (batch_size == 0) {
        out_embeddings.clear();
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return true; // Successful communication, no data returned
    }

    size_t expected_floats = (size_t)batch_size * embedding_dim;
    if (resp_msgpack.embeddings_data.size() != expected_floats) {
        // std::cerr << "Embedding data size mismatch: Got "
        //           << resp_msgpack.embeddings_data.size() << " floats,
        //           expected "
        //           << expected_floats << " (" << batch_size << "x"
        //           << embedding_dim << ")" << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    bool received_nan = false;
    for (float val : resp_msgpack.embeddings_data) {
        if (std::isnan(val)) {
            received_nan = true;
            break;
        }
    }
    if (received_nan) {
        // fprintf(stderr,
        //         "!!! [fetch_zmq] ERROR: Final check confirms NaN values in
        //         parsed embeddings_data! First requested ID: %u !!!\n",
        //         node_ids.empty() ? 0 : node_ids[0]);
        return false; // Decide whether to fail here
    } else {
        // printf("[fetch_zmq] Final check confirms embeddings data appears
        // clean (no NaNs checked).\n"); // Can be verbose
    }

    out_embeddings.clear();
    out_embeddings.resize(batch_size);
    const float* flat_data_ptr = resp_msgpack.embeddings_data.data();
    for (int i = 0; i < batch_size; i++) {
        out_embeddings[i].assign(
                flat_data_ptr + (size_t)i * embedding_dim,
                flat_data_ptr + ((size_t)i + 1) * embedding_dim);
    }

    zmq_msg_close(&response);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return true;
}

const float* ZmqDistanceComputer::get_vector_zmq(idx_t id) {
    std::vector<uint32_t> ids_to_fetch = {(uint32_t)id};
    std::vector<std::vector<float>>
            fetched_embeddings; // fetch_embeddings_zmq expects this
                                // structure

    if (!fetch_embeddings_zmq(ids_to_fetch, fetched_embeddings, zmq_port)) {
        // fprintf(stderr,
        //         "!!! ERROR get_vector_zmq: fetch_embeddings_zmq call
        //         failed for ID %ld !!!\n", (long)id);
        // Fill member with NaN to indicate failure?
        std::fill(
                last_fetched_zmq_vector.begin(),
                last_fetched_zmq_vector.end(),
                std::numeric_limits<float>::quiet_NaN());
        return nullptr; // Indicate failure upstream
    }
    if (fetched_embeddings.empty() || fetched_embeddings[0].size() != d) {
        // fprintf(stderr,
        //         "!!! ERROR get_vector_zmq: fetch_embeddings_zmq returned
        //         incorrect data for ID %ld !!!\n", (long)id);
        std::fill(
                last_fetched_zmq_vector.begin(),
                last_fetched_zmq_vector.end(),
                std::numeric_limits<float>::quiet_NaN());
        return nullptr;
    }

    // --- Copy fetched data to member variable ---
    // fetched_embeddings[0] contains the vector data
    FAISS_ASSERT(fetched_embeddings[0].size() == d);
    memcpy(last_fetched_zmq_vector.data(),
           fetched_embeddings[0].data(),
           d * sizeof(float));

    // ---- Addition: Increment fetch count on success ----
    fetch_count++;
    // ---- End Addition ----

    // --- Log values RIGHT BEFORE returning pointer ---
    const float* return_ptr = last_fetched_zmq_vector.data();
    // bool has_nan_before_return = false;
    // printf("DEBUG get_vector_zmq: Fetched ID %ld. Values BEFORE return
    // (ptr %p) [0..%d]: ",
    //        (long)id,
    //        (void*)return_ptr,
    //        (int)std::min((size_t)4, d - 1));
    // for (size_t k = 0; k < std::min((size_t)5, d); ++k) {
    //     printf("%.6f ", return_ptr[k]);
    // if (std::isnan(return_ptr[k]) || std::isinf(return_ptr[k]))
    //         has_nan_before_return = true;
    // }
    // printf("%s\n",
    //        has_nan_before_return ? "!!! HAS NaN/Inf BEFORE RETURN !!!"
    //                              : "(OK Before Return)");
    // -------------------------------------------

    return return_ptr; // Return pointer to member data
}

// --- ZMQ Distance Calculation Function (Using MessagePack) ---
bool fetch_distances_zmq(
        const std::vector<uint32_t>& node_ids,
        const float* query_vector,
        size_t query_dim,
        std::vector<float>& out_distances,
        int zmq_port = 5557) {
    DistanceRequestMsgpack req_msgpack;
    req_msgpack.node_ids = node_ids;

    // Copy query vector
    req_msgpack.query_vector.resize(query_dim);
    memcpy(req_msgpack.query_vector.data(),
           query_vector,
           query_dim * sizeof(float));

    std::stringstream buffer;
    try {
        msgpack::pack(buffer, req_msgpack);
    } catch (const std::exception& e) {
        std::cerr << "MessagePack pack failed for distance request: "
                  << e.what() << std::endl;
        return false;
    }
    std::string req_str = buffer.str();

    void* context = zmq_ctx_new();
    if (!context) {
        return false;
    }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
        zmq_ctx_destroy(context);
        return false;
    }
    int timeout = 30000;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (zmq_send(socket, req_str.data(), req_str.size(), 0) < 0) {
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    zmq_msg_t response;
    zmq_msg_init(&response);
    if (zmq_msg_recv(&response, socket, 0) < 0) {
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    DistanceResponseMsgpack resp_msgpack;
    const char* resp_data = static_cast<const char*>(zmq_msg_data(&response));
    size_t resp_size = zmq_msg_size(&response);

    try {
        msgpack::object_handle oh = msgpack::unpack(resp_data, resp_size);
        msgpack::object obj = oh.get();
        obj.convert(resp_msgpack); // Convert msgpack object to our struct
    } catch (const std::exception& e) {
        std::cerr << "MessagePack unpack failed for distance response: "
                  << e.what() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (resp_msgpack.distances.size() != node_ids.size()) {
        std::cerr << "Distance response size mismatch: Got "
                  << resp_msgpack.distances.size() << " distances, expected "
                  << node_ids.size() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    // Copy distances to output vector
    out_distances = resp_msgpack.distances;

    zmq_msg_close(&response);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return true;
}

void ZmqDistanceComputer::distances_batch(
        const std::vector<idx_t>& ids,
        std::vector<float>& distances_out) {
    // Resize output vector
    distances_out.resize(ids.size());

    // Separate nodes into disk-read and remote-read groups
    std::vector<uint32_t> remote_nodes;
    std::vector<size_t> remote_orig_indices;
    std::vector<uint32_t> disk_nodes;
    std::vector<size_t> disk_orig_indices;

    for (size_t j = 0; j < ids.size(); ++j) {
        idx_t id = ids[j];
        if (experimental_is_in_top_degree_set.empty()) {
            remote_nodes.push_back(id);
            remote_orig_indices.push_back(j);
        } else {
            assert(id >= 0 &&
                   (size_t)id < experimental_is_in_top_degree_set.size());
            if (experimental_is_in_top_degree_set[id]) {
                // Mark for disk read
                disk_nodes.push_back(id);
                disk_orig_indices.push_back(j);
            } else {
                // Mark for remote read
                remote_nodes.push_back(id);
                remote_orig_indices.push_back(j);
            }
        }
    }

    // Process remote nodes via ZMQ if any
    if (!remote_nodes.empty()) {
        // Call the original ZMQ batch function
        std::vector<float> fetched_distances;
        bool success = fetch_distances_zmq(
                remote_nodes, query.data(), d, fetched_distances, zmq_port);
        assert(success);
        assert(fetched_distances.size() == remote_nodes.size());

        for (size_t j = 0; j < remote_nodes.size(); ++j) {
            distances_out[remote_orig_indices[j]] = fetched_distances[j];
        }
        fetch_count += remote_nodes.size(); // Count these as fetches
    }

    // timing
    std::chrono::steady_clock::time_point disk_start =
            std::chrono::steady_clock::now();

    // Process disk nodes locally
    for (size_t j = 0; j < disk_nodes.size(); ++j) {
        float disk_dist =
                read_disk_and_compute_local_ip(disk_nodes[j], d, query.data());

        // sanity check with remote fetched ones
        // if (abs(disk_dist - distances_out[disk_orig_indices[j]]) > 1e-4) {
        //     printf("ERROR: disk and remote fetched distances mismatch for id
        //     %ld, error: %f\n",
        //            disk_nodes[j],
        //            abs(disk_dist - distances_out[disk_orig_indices[j]]));
        //     // assert(false);
        // }
        distances_out[disk_orig_indices[j]] = disk_dist;
    }
    fetch_disk_cache_counts += disk_nodes.size();

    // timing
    std::chrono::steady_clock::time_point disk_end =
            std::chrono::steady_clock::now();
    std::chrono::duration<double> disk_duration = disk_end - disk_start;
    // printf("ZmqDC Distances Batch Time: %f seconds\n",
    // disk_duration.count());
}

// --- Implementation of new experimental methods ---

} // namespace faiss