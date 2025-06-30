# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.18...3.22)

set(CMAKE_CXX_STANDARD 17)

if (PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE _numpy_include
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)

# pybind11_add_module(diskannpy MODULE src/diskann_bindings.cpp)
# the following is fairly synonymous with pybind11_add_module, but we need more target_link_libraries
# see https://pybind11.readthedocs.io/en/latest/compiling.html#advanced-interface-library-targets for more details
add_library(_diskannpy MODULE
        src/module.cpp
        src/builder.cpp
        src/dynamic_memory_index.cpp
        src/static_memory_index.cpp
        src/static_disk_index.cpp
)

target_include_directories(_diskannpy AFTER PRIVATE include)

if (MSVC)
    target_compile_options(_diskannpy PRIVATE /U_WINDLL)
endif()

target_link_libraries(
        _diskannpy
        PRIVATE
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
        ${PROJECT_NAME}
        ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
        ${DISKANN_ASYNC_LIB}
)

pybind11_extension(_diskannpy)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(_diskannpy)
endif()

set_target_properties(_diskannpy PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

# generally, the VERSION_INFO flag is set by pyproject.toml, by way of setup.py.
# attempts to locate the version within CMake fail because the version has to be available
# to pyproject.toml for the sdist to work after we build it.

if(NOT VERSION_INFO)
    set(VERSION_INFO "0.0.0dev")
endif()
target_compile_definitions(_diskannpy PRIVATE VERSION_INFO="${VERSION_INFO}")

# Add a post-build command to automatically copy the compiled Python module
if(UPDATE_EDITABLE_INSTALL)
add_custom_command(
TARGET _diskannpy
POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy 
        ${CMAKE_CURRENT_BINARY_DIR}/_diskannpy.cpython-*.so
        ${CMAKE_SOURCE_DIR}/python/src/
COMMENT "Copying Python module to python/src directory"
)
endif()