# 最终简化版
cmake_minimum_required(VERSION 3.24)
project(leann_backend_hnsw_wrapper)

set(FAISS_ENABLE_PYTHON ON CACHE BOOL "" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)

add_subdirectory(third_party/faiss)