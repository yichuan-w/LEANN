# packages/leann-backend-diskann/CMakeLists.txt (simplified version)

cmake_minimum_required(VERSION 3.20)
project(leann_backend_diskann_wrapper)

# Find Python - scikit-build-core should provide this
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Print Python information for debugging
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")

# Pass Python information to DiskANN through cache variables
set(Python_EXECUTABLE ${Python_EXECUTABLE} CACHE FILEPATH "Python executable" FORCE)
set(Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE PATH "Python include dirs" FORCE)
set(Python_LIBRARIES ${Python_LIBRARIES} CACHE FILEPATH "Python libraries" FORCE)
set(Python_VERSION ${Python_VERSION} CACHE STRING "Python version" FORCE)
set(Python_FOUND ${Python_FOUND} CACHE BOOL "Python found" FORCE)

# Also set Python3 variables for compatibility
set(Python3_EXECUTABLE ${Python_EXECUTABLE} CACHE FILEPATH "Python3 executable" FORCE)
set(Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE PATH "Python3 include dirs" FORCE)
set(Python3_LIBRARIES ${Python_LIBRARIES} CACHE FILEPATH "Python3 libraries" FORCE)
set(Python3_VERSION ${Python_VERSION} CACHE STRING "Python3 version" FORCE)
set(Python3_FOUND ${Python_FOUND} CACHE BOOL "Python3 found" FORCE)
set(Python3_Development_FOUND TRUE CACHE BOOL "Python3 development found" FORCE)

# Set Python finding strategy
set(Python_FIND_VIRTUALENV ONLY CACHE STRING "" FORCE)
set(Python3_FIND_VIRTUALENV ONLY CACHE STRING "" FORCE)

# Tell CMake to directly enter the DiskANN submodule and execute its own CMakeLists.txt
# DiskANN will handle everything itself, including compiling Python bindings
add_subdirectory(src/third_party/DiskANN)
