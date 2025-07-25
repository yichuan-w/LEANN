cmake_minimum_required(VERSION 3.24)
project(leann_backend_hnsw_wrapper)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

# Set OpenMP path for macOS
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

# Use system ZeroMQ instead of building from source
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Add cppzmq headers
include_directories(third_party/cppzmq)

# Configure msgpack-c - disable boost dependency
set(MSGPACK_USE_BOOST OFF CACHE BOOL "" FORCE)
add_compile_definitions(MSGPACK_NO_BOOST)
include_directories(third_party/msgpack-c/include)

# Find Python before configuring Faiss
# Check if we're in scikit-build environment
if(DEFINED SKBUILD)
    message(STATUS "Building with scikit-build, using provided Python")
    # scikit-build-core provides Python_* variables
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

# Print Python information for debugging
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")

# Faiss configuration - streamlined build
set(FAISS_ENABLE_PYTHON ON CACHE BOOL "" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)

# Disable additional SIMD versions to speed up compilation
set(FAISS_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)

# Additional optimization options from INSTALL.md
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # Static library is faster to build

# Avoid building demos and benchmarks
set(BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHS OFF CACHE BOOL "" FORCE)

# NEW: Tell Faiss to only build the generic version
set(FAISS_BUILD_GENERIC ON CACHE BOOL "" FORCE)
set(FAISS_BUILD_AVX2 OFF CACHE BOOL "" FORCE)
set(FAISS_BUILD_AVX512 OFF CACHE BOOL "" FORCE)

# IMPORTANT: Disable building AVX versions to speed up compilation
set(FAISS_BUILD_AVX_VERSIONS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/faiss)