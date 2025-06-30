// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include "parameters.h"

BOOST_AUTO_TEST_SUITE(IndexWriteParametersBuilder_tests)

BOOST_AUTO_TEST_CASE(test_build)
{
    uint32_t search_list_size = rand();
    uint32_t max_degree = rand();
    float alpha = (float)rand();
    uint32_t filter_list_size = rand();
    uint32_t max_occlusion_size = rand();
    bool saturate_graph = true;

    diskann::IndexWriteParametersBuilder builder(search_list_size, max_degree);

    builder.with_alpha(alpha)
        .with_filter_list_size(filter_list_size)
        .with_max_occlusion_size(max_occlusion_size)
        .with_num_threads(0)
        .with_saturate_graph(saturate_graph);

    {
        auto parameters = builder.build();

        BOOST_TEST(search_list_size == parameters.search_list_size);
        BOOST_TEST(max_degree == parameters.max_degree);
        BOOST_TEST(alpha == parameters.alpha);
        BOOST_TEST(filter_list_size == parameters.filter_list_size);
        BOOST_TEST(max_occlusion_size == parameters.max_occlusion_size);
        BOOST_TEST(saturate_graph == parameters.saturate_graph);

        BOOST_TEST(parameters.num_threads > (uint32_t)0);
    }

    {
        uint32_t num_threads = rand() + 1;
        saturate_graph = false;
        builder.with_num_threads(num_threads).with_saturate_graph(saturate_graph);

        auto parameters = builder.build();

        BOOST_TEST(search_list_size == parameters.search_list_size);
        BOOST_TEST(max_degree == parameters.max_degree);
        BOOST_TEST(alpha == parameters.alpha);
        BOOST_TEST(filter_list_size == parameters.filter_list_size);
        BOOST_TEST(max_occlusion_size == parameters.max_occlusion_size);
        BOOST_TEST(saturate_graph == parameters.saturate_graph);

        BOOST_TEST(num_threads == parameters.num_threads);
    }
}

BOOST_AUTO_TEST_SUITE_END()
