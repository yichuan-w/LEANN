// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "index.h"
#include "parameters.h"

namespace py = pybind11;

namespace diskannpy
{

template <typename DT>
class DynamicMemoryIndex
{
  public:
    DynamicMemoryIndex(diskann::Metric m, size_t dimensions, size_t max_vectors, uint32_t complexity,
                       uint32_t graph_degree, bool saturate_graph, uint32_t max_occlusion_size, float alpha,
                       uint32_t num_threads, uint32_t filter_complexity, uint32_t num_frozen_points,
                       uint32_t initial_search_complexity, uint32_t initial_search_threads,
                       bool concurrent_consolidation);

    void load(const std::string &index_path);
    int insert(const py::array_t<DT, py::array::c_style | py::array::forcecast> &vector, DynamicIdType id);
    py::array_t<int> batch_insert(py::array_t<DT, py::array::c_style | py::array::forcecast> &vectors,
                                  py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &ids, int32_t num_inserts,
                                  int num_threads = 0);
    int mark_deleted(DynamicIdType id);
    void save(const std::string &save_path, bool compact_before_save = false);
    NeighborsAndDistances<DynamicIdType> search(py::array_t<DT, py::array::c_style | py::array::forcecast> &query, uint64_t knn,
                                      uint64_t complexity);
    NeighborsAndDistances<DynamicIdType> batch_search(py::array_t<DT, py::array::c_style | py::array::forcecast> &queries,
                                            uint64_t num_queries, uint64_t knn, uint64_t complexity,
                                            uint32_t num_threads);
    void consolidate_delete();
    size_t num_points();


  private:
    const uint32_t _initial_search_complexity;
    const diskann::IndexWriteParameters _write_parameters;
    diskann::Index<DT, DynamicIdType, filterT> _index;
};

}; // namespace diskannpy