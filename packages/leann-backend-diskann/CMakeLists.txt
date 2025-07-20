# packages/leann-backend-diskann/CMakeLists.txt (simplified version)

cmake_minimum_required(VERSION 3.20)
project(leann_backend_diskann_wrapper)

# Tell CMake to directly enter the DiskANN submodule and execute its own CMakeLists.txt
# DiskANN will handle everything itself, including compiling Python bindings
add_subdirectory(src/third_party/DiskANN)
