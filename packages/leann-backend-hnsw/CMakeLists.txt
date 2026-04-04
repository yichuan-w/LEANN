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

# Find ZMQ via pkg-config on all platforms.
# Windows CI installs pkgconfiglite + zeromq via vcpkg and exports PKG_CONFIG_PATH.
find_package(PkgConfig REQUIRED)

# On ARM64 macOS, ensure pkg-config finds ARM64 Homebrew packages first
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

# This creates PkgConfig::ZMQ target automatically with correct properties
if(TARGET PkgConfig::ZMQ)
    message(STATUS "Found and configured ZMQ target: PkgConfig::ZMQ")
else()
    message(FATAL_ERROR "pkg_check_modules did not create IMPORTED target for ZMQ.")
endif()

# Add cppzmq headers
include_directories(SYSTEM third_party/cppzmq)

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
        # Default to generic: many ARM64 Linux CI runners (and some consumer
        # devices) don't implement SVE, and Faiss built with SVE can SIGILL at
        # runtime. Opt in with `-DLEANN_FAISS_ARM64_SVE=ON`.
        option(LEANN_FAISS_ARM64_SVE "Enable SVE in Faiss on ARM64 Linux (requires SVE at runtime)" OFF)
        if(LEANN_FAISS_ARM64_SVE)
            set(FAISS_OPT_LEVEL "sve" CACHE STRING "" FORCE)
            message(STATUS "Setting FAISS_OPT_LEVEL to 'sve' for ARM64 Linux (LEANN_FAISS_ARM64_SVE=ON)")
        else()
            set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)
            message(STATUS "Setting FAISS_OPT_LEVEL to 'generic' for ARM64 Linux")
        endif()
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
