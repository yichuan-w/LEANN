cmake_minimum_required(VERSION 3.24)
project(leann_backend_hnsw_wrapper)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

# Set OpenMP path for macOS
if(APPLE)
    # Detect Homebrew installation path (Apple Silicon vs Intel)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(HOMEBREW_PREFIX "/opt/homebrew")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(HOMEBREW_PREFIX "/usr/local")
    else()
        message(FATAL_ERROR "Could not find libomp installation. Please install with: brew install libomp")
    endif()

    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${HOMEBREW_PREFIX}/opt/libomp/lib/libomp.dylib")

    # Force use of system libc++ to avoid version mismatch
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")

    # Set minimum macOS version for better compatibility
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
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

# Faiss configuration - streamlined build
set(FAISS_ENABLE_PYTHON ON CACHE BOOL "" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)

# Disable x86-specific SIMD optimizations (important for ARM64 compatibility)
set(FAISS_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_SSE4_1 OFF CACHE BOOL "" FORCE)

# ARM64-specific configuration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Configuring Faiss for ARM64 architecture")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Use SVE optimization level for ARM64 Linux (as seen in Faiss conda build)
        set(FAISS_OPT_LEVEL "sve" CACHE STRING "" FORCE)
        message(STATUS "Setting FAISS_OPT_LEVEL to 'sve' for ARM64 Linux")
    else()
        # Use generic optimization for other ARM64 platforms (like macOS)
        set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)
        message(STATUS "Setting FAISS_OPT_LEVEL to 'generic' for ARM64 ${CMAKE_SYSTEM_NAME}")
    endif()

    # ARM64 compatibility: Faiss submodule has been modified to fix x86 header inclusion
    message(STATUS "Using ARM64-compatible Faiss submodule")
endif()

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
