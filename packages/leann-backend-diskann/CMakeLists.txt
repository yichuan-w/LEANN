# Wrapper so USE_TCMALLOC is OFF before DiskANN's option() runs. DiskANN's
# FetchContent(tcmalloc) pulls protobuf/re2/fuzztest and fails on Linux CI when
# Abseil CMake targets are missing (e.g. ubuntu arm64 with apt protobuf only).
cmake_minimum_required(VERSION 3.24)
project(leann_backend_diskann_wrapper NONE)

set(USE_TCMALLOC OFF CACHE BOOL "Use tcmalloc from gperftools" FORCE)

add_subdirectory(third_party/DiskANN)
