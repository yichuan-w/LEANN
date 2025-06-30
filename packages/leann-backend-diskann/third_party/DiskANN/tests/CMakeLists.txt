# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

find_package(Boost COMPONENTS unit_test_framework)

# For Windows, fall back to nuget version if find_package didn't find it.
if (MSVC AND NOT Boost_FOUND)
    set(DISKANN_BOOST_INCLUDE "${DISKANN_MSVC_PACKAGES}/boost/lib/native/include")
    # Multi-threaded static library.
    set(UNIT_TEST_FRAMEWORK_LIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB ${UNIT_TEST_FRAMEWORK_LIB_PATTERN})

    set(UNIT_TEST_FRAMEWORK_DLIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-gd-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB ${UNIT_TEST_FRAMEWORK_DLIB_PATTERN})

    if (EXISTS ${DISKANN_BOOST_INCLUDE} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB})
        set(Boost_FOUND ON)
        set(Boost_INCLUDE_DIR ${DISKANN_BOOST_INCLUDE})
        add_library(Boost::unit_test_framework STATIC IMPORTED)
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_RELEASE "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB}")
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_DEBUG "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB}")
        message(STATUS "Falling back to using Boost from the nuget package")
    else()
        message(WARNING "Couldn't find Boost. Was looking for ${DISKANN_BOOST_INCLUDE} and ${UNIT_TEST_FRAMEWORK_LIB_PATTERN}")
    endif()
endif()

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()


set(DISKANN_UNIT_TEST_SOURCES main.cpp index_write_parameters_builder_tests.cpp)

add_executable(${PROJECT_NAME}_unit_tests ${DISKANN_SOURCES} ${DISKANN_UNIT_TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_unit_tests ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::unit_test_framework)

add_test(NAME ${PROJECT_NAME}_unit_tests COMMAND ${PROJECT_NAME}_unit_tests)

