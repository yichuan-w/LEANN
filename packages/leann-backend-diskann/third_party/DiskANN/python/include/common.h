// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace diskannpy
{

typedef uint32_t filterT;

typedef uint32_t StaticIdType;
typedef uint32_t DynamicIdType;

template <class IdType> using NeighborsAndDistances = std::pair<py::array_t<IdType>, py::array_t<float>>;

}; // namespace diskannpy
