# packages/leann-backend-diskann/CMakeLists.txt (最终简化版)

cmake_minimum_required(VERSION 3.20)
project(leann_backend_diskann_wrapper)

# 告诉 CMake 直接进入 DiskANN 子模块并执行它自己的 CMakeLists.txt
# DiskANN 会自己处理所有事情，包括编译 Python 绑定
add_subdirectory(src/third_party/DiskANN)
