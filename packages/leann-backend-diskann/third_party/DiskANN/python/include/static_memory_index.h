// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "index.h"

namespace py = pybind11;

namespace diskannpy
{

template <typename DT> class StaticMemoryIndex
{
  public:
    StaticMemoryIndex(diskann::Metric m, const std::string &index_prefix, size_t num_points, size_t dimensions,
                      uint32_t num_threads, uint32_t initial_search_complexity);

    NeighborsAndDistances<StaticIdType> search(py::array_t<DT, py::array::c_style | py::array::forcecast> &query,
                                               uint64_t knn, uint64_t complexity);

    NeighborsAndDistances<StaticIdType> search_with_filter(
        py::array_t<DT, py::array::c_style | py::array::forcecast> &query, uint64_t knn, uint64_t complexity,
        filterT filter);

    NeighborsAndDistances<StaticIdType> batch_search(
        py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn,
        uint64_t complexity, uint32_t num_threads);

  private:
    diskann::Index<DT, StaticIdType, filterT> _index;
};
} // namespace diskannpy