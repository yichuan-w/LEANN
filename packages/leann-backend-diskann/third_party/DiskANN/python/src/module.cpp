// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "defaults.h"
#include "distance.h"

#include "builder.h"
#include "dynamic_memory_index.h"
#include "static_disk_index.h"
#include "static_memory_index.h"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

struct Variant
{
    std::string disk_builder_name;
    std::string memory_builder_name;
    std::string dynamic_memory_index_name;
    std::string static_memory_index_name;
    std::string static_disk_index_name;
};

const Variant FloatVariant{"build_disk_float_index", "build_memory_float_index", "DynamicMemoryFloatIndex",
                           "StaticMemoryFloatIndex", "StaticDiskFloatIndex"};

const Variant UInt8Variant{"build_disk_uint8_index", "build_memory_uint8_index", "DynamicMemoryUInt8Index",
                           "StaticMemoryUInt8Index", "StaticDiskUInt8Index"};

const Variant Int8Variant{"build_disk_int8_index", "build_memory_int8_index", "DynamicMemoryInt8Index",
                          "StaticMemoryInt8Index", "StaticDiskInt8Index"};

template <typename T> inline void add_variant(py::module_ &m, const Variant &variant)
{
    m.def(variant.disk_builder_name.c_str(), &diskannpy::build_disk_index<T>, "distance_metric"_a, "data_file_path"_a,
          "index_prefix_path"_a, "complexity"_a, "graph_degree"_a, "final_index_ram_limit"_a, "indexing_ram_budget"_a,
          "num_threads"_a, "pq_disk_bytes"_a, "codebook_prefix"_a = "");

    m.def(variant.memory_builder_name.c_str(), &diskannpy::build_memory_index<T>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "complexity"_a, "alpha"_a, "num_threads"_a,
          "use_pq_build"_a, "num_pq_bytes"_a, "use_opq"_a, "use_tags"_a = false, "filter_labels_file"_a = "",
          "universal_label"_a = "", "filter_complexity"_a = 0);

    py::class_<diskannpy::StaticMemoryIndex<T>>(m, variant.static_memory_index_name.c_str())
        .def(py::init<const diskann::Metric, const std::string &, const size_t, const size_t, const uint32_t,
                      const uint32_t>(),
             "distance_metric"_a, "index_path"_a, "num_points"_a, "dimensions"_a, "num_threads"_a,
             "initial_search_complexity"_a)
        .def("search", &diskannpy::StaticMemoryIndex<T>::search, "query"_a, "knn"_a, "complexity"_a)
        .def("search_with_filter", &diskannpy::StaticMemoryIndex<T>::search_with_filter, "query"_a, "knn"_a,
             "complexity"_a, "filter"_a)
        .def("batch_search", &diskannpy::StaticMemoryIndex<T>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "complexity"_a, "num_threads"_a);

    py::class_<diskannpy::DynamicMemoryIndex<T>>(m, variant.dynamic_memory_index_name.c_str())
        .def(py::init<const diskann::Metric, const size_t, const size_t, const uint32_t, const uint32_t, const bool,
                      const uint32_t, const float, const uint32_t, const uint32_t, const uint32_t, const uint32_t,
                      const uint32_t, const bool>(),
             "distance_metric"_a, "dimensions"_a, "max_vectors"_a, "complexity"_a, "graph_degree"_a,
             "saturate_graph"_a = diskann::defaults::SATURATE_GRAPH,
             "max_occlusion_size"_a = diskann::defaults::MAX_OCCLUSION_SIZE, "alpha"_a = diskann::defaults::ALPHA,
             "num_threads"_a = diskann::defaults::NUM_THREADS,
             "filter_complexity"_a = diskann::defaults::FILTER_LIST_SIZE,
             "num_frozen_points"_a = diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC, "initial_search_complexity"_a = 0,
             "search_threads"_a = 0, "concurrent_consolidation"_a = true)
        .def("search", &diskannpy::DynamicMemoryIndex<T>::search, "query"_a, "knn"_a, "complexity"_a)
        .def("load", &diskannpy::DynamicMemoryIndex<T>::load, "index_path"_a)
        .def("batch_search", &diskannpy::DynamicMemoryIndex<T>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "complexity"_a, "num_threads"_a)
        .def("batch_insert", &diskannpy::DynamicMemoryIndex<T>::batch_insert, "vectors"_a, "ids"_a, "num_inserts"_a,
             "num_threads"_a)
        .def("save", &diskannpy::DynamicMemoryIndex<T>::save, "save_path"_a = "", "compact_before_save"_a = false)
        .def("insert", &diskannpy::DynamicMemoryIndex<T>::insert, "vector"_a, "id"_a)
        .def("mark_deleted", &diskannpy::DynamicMemoryIndex<T>::mark_deleted, "id"_a)
        .def("consolidate_delete", &diskannpy::DynamicMemoryIndex<T>::consolidate_delete)
        .def("num_points", &diskannpy::DynamicMemoryIndex<T>::num_points);

    py::class_<diskannpy::StaticDiskIndex<T>>(m, variant.static_disk_index_name.c_str())
        .def(py::init<const diskann::Metric, const std::string &, const uint32_t, const size_t, const uint32_t,
                      const std::string &, const std::string &>(),
             "distance_metric"_a, "index_path_prefix"_a, "num_threads"_a, "num_nodes_to_cache"_a,
             "cache_mechanism"_a = 1, "pq_prefix"_a = "", "partition_prefix"_a)
        .def("cache_bfs_levels", &diskannpy::StaticDiskIndex<T>::cache_bfs_levels, "num_nodes_to_cache"_a)
        .def("search", &diskannpy::StaticDiskIndex<T>::search, "query"_a, "knn"_a, "complexity"_a, "beam_width"_a,
             "USE_DEFERRED_FETCH"_a = false, "skip_search_reorder"_a = false, "recompute_beighbor_embeddings"_a = false,
             "dedup_node_dis"_a = false, "prune_ratio"_a = 0, "batch_recompute"_a = false, "global_pruning"_a = false)
        .def("batch_search", &diskannpy::StaticDiskIndex<T>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "complexity"_a, "beam_width"_a, "num_threads"_a, "USE_DEFERRED_FETCH"_a = false,
             "skip_search_reorder"_a = false, "recompute_beighbor_embeddings"_a = false, "dedup_node_dis"_a = false,
             "prune_ratio"_a = 0, "batch_recompute"_a = false, "global_pruning"_a = false);
}

PYBIND11_MODULE(_diskannpy, m)
{
    m.doc() = "DiskANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    // let's re-export our defaults
    py::module_ default_values = m.def_submodule(
        "defaults",
        "A collection of the default values used for common diskann operations. `GRAPH_DEGREE` and `COMPLEXITY` are not"
        " set as defaults, but some semi-reasonable default values are selected for your convenience. We urge you to "
        "investigate their meaning and adjust them for your use cases.");

    default_values.attr("ALPHA") = diskann::defaults::ALPHA;
    default_values.attr("NUM_THREADS") = diskann::defaults::NUM_THREADS;
    default_values.attr("MAX_OCCLUSION_SIZE") = diskann::defaults::MAX_OCCLUSION_SIZE;
    default_values.attr("FILTER_COMPLEXITY") = diskann::defaults::FILTER_LIST_SIZE;
    default_values.attr("NUM_FROZEN_POINTS_STATIC") = diskann::defaults::NUM_FROZEN_POINTS_STATIC;
    default_values.attr("NUM_FROZEN_POINTS_DYNAMIC") = diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC;
    default_values.attr("SATURATE_GRAPH") = diskann::defaults::SATURATE_GRAPH;
    default_values.attr("GRAPH_DEGREE") = diskann::defaults::MAX_DEGREE;
    default_values.attr("COMPLEXITY") = diskann::defaults::BUILD_LIST_SIZE;
    default_values.attr("PQ_DISK_BYTES") = (uint32_t)0;
    default_values.attr("USE_PQ_BUILD") = false;
    default_values.attr("NUM_PQ_BYTES") = (uint32_t)0;
    default_values.attr("USE_OPQ") = false;

    add_variant<float>(m, FloatVariant);
    add_variant<uint8_t>(m, UInt8Variant);
    add_variant<int8_t>(m, Int8Variant);

    py::enum_<diskann::Metric>(m, "Metric")
        .value("L2", diskann::Metric::L2)
        .value("INNER_PRODUCT", diskann::Metric::INNER_PRODUCT)
        .value("COSINE", diskann::Metric::COSINE)
        .export_values();
}
