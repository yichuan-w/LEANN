// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/program_options.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include "filter_utils.h"
#include <omp.h>
#ifndef _WINDOWS
#include <sys/uio.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "utils.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;
typedef std::tuple<std::vector<std::vector<uint32_t>>, uint64_t> stitch_indices_return_values;

/*
 * Inline function to display progress bar.
 */
inline void print_progress(double percentage)
{
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

/*
 * Inline function to generate a random integer in a range.
 */
inline size_t random(size_t range_from, size_t range_to)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<size_t> distr(range_from, range_to);
    return distr(generator);
}

/*
 * function to handle command line parsing.
 *
 * Arguments are merely the inputs from the command line.
 */
void handle_args(int argc, char **argv, std::string &data_type, path &input_data_path, path &final_index_path_prefix,
                 path &label_data_path, std::string &universal_label, uint32_t &num_threads, uint32_t &R, uint32_t &L,
                 uint32_t &stitched_R, float &alpha)
{
    po::options_description desc{
        program_options_utils::make_program_description("build_stitched_index", "Build a stitched DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("index_path_prefix",
                                       po::value<std::string>(&final_index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&input_data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_data_path)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("stitched_R", po::value<uint32_t>(&stitched_R)->default_value(100),
                                       "Degree to prune final graph down to");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            exit(0);
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        throw;
    }
}

/*
 * Custom index save to write the in-memory index to disk.
 * Also writes required files for diskANN API -
 *  1. labels_to_medoids
 *  2. universal_label
 *  3. data (redundant for static indices)
 *  4. labels (redundant for static indices)
 */
void save_full_index(path final_index_path_prefix, path input_data_path, uint64_t final_index_size,
                     std::vector<std::vector<uint32_t>> stitched_graph,
                     tsl::robin_map<std::string, uint32_t> entry_points, std::string universal_label,
                     path label_data_path)
{
    // aux. file 1
    auto saving_index_timer = std::chrono::high_resolution_clock::now();
    std::ifstream original_label_data_stream;
    original_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    original_label_data_stream.open(label_data_path, std::ios::binary);
    std::ofstream new_label_data_stream;
    new_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    new_label_data_stream.open(final_index_path_prefix + "_labels.txt", std::ios::binary);
    new_label_data_stream << original_label_data_stream.rdbuf();
    original_label_data_stream.close();
    new_label_data_stream.close();

    // aux. file 2
    std::ifstream original_input_data_stream;
    original_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    original_input_data_stream.open(input_data_path, std::ios::binary);
    std::ofstream new_input_data_stream;
    new_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    new_input_data_stream.open(final_index_path_prefix + ".data", std::ios::binary);
    new_input_data_stream << original_input_data_stream.rdbuf();
    original_input_data_stream.close();
    new_input_data_stream.close();

    // aux. file 3
    std::ofstream labels_to_medoids_writer;
    labels_to_medoids_writer.exceptions(std::ios::badbit | std::ios::failbit);
    labels_to_medoids_writer.open(final_index_path_prefix + "_labels_to_medoids.txt");
    for (auto iter : entry_points)
        labels_to_medoids_writer << iter.first << ", " << iter.second << std::endl;
    labels_to_medoids_writer.close();

    // aux. file 4 (only if we're using a universal label)
    if (universal_label != "")
    {
        std::ofstream universal_label_writer;
        universal_label_writer.exceptions(std::ios::badbit | std::ios::failbit);
        universal_label_writer.open(final_index_path_prefix + "_universal_label.txt");
        universal_label_writer << universal_label << std::endl;
        universal_label_writer.close();
    }

    // main index
    uint64_t index_num_frozen_points = 0, index_num_edges = 0;
    uint32_t index_max_observed_degree = 0, index_entry_point = 0;
    const size_t METADATA = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    for (auto &point_neighbors : stitched_graph)
    {
        index_max_observed_degree = std::max(index_max_observed_degree, (uint32_t)point_neighbors.size());
    }

    std::ofstream stitched_graph_writer;
    stitched_graph_writer.exceptions(std::ios::badbit | std::ios::failbit);
    stitched_graph_writer.open(final_index_path_prefix, std::ios_base::binary);

    stitched_graph_writer.write((char *)&final_index_size, sizeof(uint64_t));
    stitched_graph_writer.write((char *)&index_max_observed_degree, sizeof(uint32_t));
    stitched_graph_writer.write((char *)&index_entry_point, sizeof(uint32_t));
    stitched_graph_writer.write((char *)&index_num_frozen_points, sizeof(uint64_t));

    size_t bytes_written = METADATA;
    for (uint32_t node_point = 0; node_point < stitched_graph.size(); node_point++)
    {
        uint32_t current_node_num_neighbors = (uint32_t)stitched_graph[node_point].size();
        std::vector<uint32_t> current_node_neighbors = stitched_graph[node_point];
        stitched_graph_writer.write((char *)&current_node_num_neighbors, sizeof(uint32_t));
        bytes_written += sizeof(uint32_t);
        for (const auto &current_node_neighbor : current_node_neighbors)
        {
            stitched_graph_writer.write((char *)&current_node_neighbor, sizeof(uint32_t));
            bytes_written += sizeof(uint32_t);
        }
        index_num_edges += current_node_num_neighbors;
    }

    if (bytes_written != final_index_size)
    {
        std::cerr << "Error: written bytes does not match allocated space" << std::endl;
        throw;
    }

    stitched_graph_writer.close();

    std::chrono::duration<double> saving_index_time = std::chrono::high_resolution_clock::now() - saving_index_timer;
    std::cout << "Stitched graph written in " << saving_index_time.count() << " seconds" << std::endl;
    std::cout << "Stitched graph average degree: " << ((float)index_num_edges) / ((float)(stitched_graph.size()))
              << std::endl;
    std::cout << "Stitched graph max degree: " << index_max_observed_degree << std::endl << std::endl;
}

/*
 * Unions the per-label graph indices together via the following policy:
 *  - any two nodes can only have at most one edge between them -
 *
 * Returns the "stitched" graph and its expected file size.
 */
template <typename T>
stitch_indices_return_values stitch_label_indices(
    path final_index_path_prefix, uint32_t total_number_of_points, label_set all_labels,
    tsl::robin_map<std::string, uint32_t> labels_to_number_of_points,
    tsl::robin_map<std::string, uint32_t> &label_entry_points,
    tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id_map)
{
    size_t final_index_size = 0;
    std::vector<std::vector<uint32_t>> stitched_graph(total_number_of_points);

    auto stitching_index_timer = std::chrono::high_resolution_clock::now();
    for (const auto &lbl : all_labels)
    {
        path curr_label_index_path(final_index_path_prefix + "_" + lbl);
        std::vector<std::vector<uint32_t>> curr_label_index;
        uint64_t curr_label_index_size;
        uint32_t curr_label_entry_point;

        std::tie(curr_label_index, curr_label_index_size) =
            diskann::load_label_index(curr_label_index_path, labels_to_number_of_points[lbl]);
        curr_label_entry_point = (uint32_t)random(0, curr_label_index.size());
        label_entry_points[lbl] = label_id_to_orig_id_map[lbl][curr_label_entry_point];

        for (uint32_t node_point = 0; node_point < curr_label_index.size(); node_point++)
        {
            uint32_t original_point_id = label_id_to_orig_id_map[lbl][node_point];
            for (auto &node_neighbor : curr_label_index[node_point])
            {
                uint32_t original_neighbor_id = label_id_to_orig_id_map[lbl][node_neighbor];
                std::vector<uint32_t> curr_point_neighbors = stitched_graph[original_point_id];
                if (std::find(curr_point_neighbors.begin(), curr_point_neighbors.end(), original_neighbor_id) ==
                    curr_point_neighbors.end())
                {
                    stitched_graph[original_point_id].push_back(original_neighbor_id);
                    final_index_size += sizeof(uint32_t);
                }
            }
        }
    }

    const size_t METADATA = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    final_index_size += (total_number_of_points * sizeof(uint32_t) + METADATA);

    std::chrono::duration<double> stitching_index_time =
        std::chrono::high_resolution_clock::now() - stitching_index_timer;
    std::cout << "stitched graph generated in memory in " << stitching_index_time.count() << " seconds" << std::endl;

    return std::make_tuple(stitched_graph, final_index_size);
}

/*
 * Applies the prune_neighbors function from src/index.cpp to
 * every node in the stitched graph.
 *
 * This is an optional step, hence the saving of both the full
 * and pruned graph.
 */
template <typename T>
void prune_and_save(path final_index_path_prefix, path full_index_path_prefix, path input_data_path,
                    std::vector<std::vector<uint32_t>> stitched_graph, uint32_t stitched_R,
                    tsl::robin_map<std::string, uint32_t> label_entry_points, std::string universal_label,
                    path label_data_path, uint32_t num_threads)
{
    size_t dimension, number_of_label_points;
    auto diskann_cout_buffer = diskann::cout.rdbuf(nullptr);
    auto std_cout_buffer = std::cout.rdbuf(nullptr);
    auto pruning_index_timer = std::chrono::high_resolution_clock::now();

    diskann::get_bin_metadata(input_data_path, number_of_label_points, dimension);

    diskann::Index<T> index(diskann::Metric::L2, dimension, number_of_label_points, nullptr, nullptr, 0, false, false,
                            false, false, 0, false);

    // not searching this index, set search_l to 0
    index.load(full_index_path_prefix.c_str(), num_threads, 1);

    std::cout << "parsing labels" << std::endl;

    index.prune_all_neighbors(stitched_R, 750, 1.2);
    index.save((final_index_path_prefix).c_str());

    diskann::cout.rdbuf(diskann_cout_buffer);
    std::cout.rdbuf(std_cout_buffer);
    std::chrono::duration<double> pruning_index_time = std::chrono::high_resolution_clock::now() - pruning_index_timer;
    std::cout << "pruning performed in " << pruning_index_time.count() << " seconds\n" << std::endl;
}

/*
 * Delete all temporary artifacts.
 * In the process of creating the stitched index, some temporary artifacts are
 * created:
 * 1. the separate bin files for each labels' points
 * 2. the separate diskANN indices built for each label
 * 3. the '.data' file created while generating the indices
 */
void clean_up_artifacts(path input_data_path, path final_index_path_prefix, label_set all_labels)
{
    for (const auto &lbl : all_labels)
    {
        path curr_label_input_data_path(input_data_path + "_" + lbl);
        path curr_label_index_path(final_index_path_prefix + "_" + lbl);
        path curr_label_index_path_data(curr_label_index_path + ".data");

        if (std::remove(curr_label_index_path.c_str()) != 0)
            throw;
        if (std::remove(curr_label_input_data_path.c_str()) != 0)
            throw;
        if (std::remove(curr_label_index_path_data.c_str()) != 0)
            throw;
    }
}

int main(int argc, char **argv)
{
    // 1. handle cmdline inputs
    std::string data_type;
    path input_data_path, final_index_path_prefix, label_data_path;
    std::string universal_label;
    uint32_t num_threads, R, L, stitched_R;
    float alpha;

    auto index_timer = std::chrono::high_resolution_clock::now();
    handle_args(argc, argv, data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label,
                num_threads, R, L, stitched_R, alpha);

    path labels_file_to_use = final_index_path_prefix + "_label_formatted.txt";
    path labels_map_file = final_index_path_prefix + "_labels_map.txt";

    convert_labels_string_to_int(label_data_path, labels_file_to_use, labels_map_file, universal_label);

    // 2. parse label file and create necessary data structures
    std::vector<label_set> point_ids_to_labels;
    tsl::robin_map<std::string, uint32_t> labels_to_number_of_points;
    label_set all_labels;

    std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) =
        diskann::parse_label_file(labels_file_to_use, universal_label);

    // 3. for each label, make a separate data file
    tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id_map;
    uint32_t total_number_of_points = (uint32_t)point_ids_to_labels.size();

#ifndef _WINDOWS
    if (data_type == "uint8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files<uint8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "int8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files<int8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "float")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files<float>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else
        throw;
#else
    if (data_type == "uint8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<uint8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "int8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<int8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "float")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<float>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else
        throw;
#endif

    // 4. for each created data file, create a vanilla diskANN index
    if (data_type == "uint8")
        diskann::generate_label_indices<uint8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                                 num_threads);
    else if (data_type == "int8")
        diskann::generate_label_indices<int8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                                num_threads);
    else if (data_type == "float")
        diskann::generate_label_indices<float>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                               num_threads);
    else
        throw;

    // 5. "stitch" the indices together
    std::vector<std::vector<uint32_t>> stitched_graph;
    tsl::robin_map<std::string, uint32_t> label_entry_points;
    uint64_t stitched_graph_size;

    if (data_type == "uint8")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<uint8_t>(final_index_path_prefix, total_number_of_points, all_labels,
                                          labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else if (data_type == "int8")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<int8_t>(final_index_path_prefix, total_number_of_points, all_labels,
                                         labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else if (data_type == "float")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<float>(final_index_path_prefix, total_number_of_points, all_labels,
                                        labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else
        throw;
    path full_index_path_prefix = final_index_path_prefix + "_full";
    // 5a. save the stitched graph to disk
    save_full_index(full_index_path_prefix, input_data_path, stitched_graph_size, stitched_graph, label_entry_points,
                    universal_label, labels_file_to_use);

    // 6. run a prune on the stitched index, and save to disk
    if (data_type == "uint8")
        prune_and_save<uint8_t>(final_index_path_prefix, full_index_path_prefix, input_data_path, stitched_graph,
                                stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else if (data_type == "int8")
        prune_and_save<int8_t>(final_index_path_prefix, full_index_path_prefix, input_data_path, stitched_graph,
                               stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else if (data_type == "float")
        prune_and_save<float>(final_index_path_prefix, full_index_path_prefix, input_data_path, stitched_graph,
                              stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else
        throw;

    std::chrono::duration<double> index_time = std::chrono::high_resolution_clock::now() - index_timer;
    std::cout << "pruned/stitched graph generated in " << index_time.count() << " seconds" << std::endl;

    clean_up_artifacts(input_data_path, final_index_path_prefix, all_labels);
}
