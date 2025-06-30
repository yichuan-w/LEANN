// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <cstdlib>
#include <codecvt>
#include <boost/program_options.hpp>

#include <cpprest/http_client.h>
#include <restapi/common.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

using namespace diskann;
namespace po = boost::program_options;

template <typename T>
void query_loop(const std::string &ip_addr_port, const std::string &query_file, const unsigned nq, const unsigned Ls,
                const unsigned k_value)
{
    web::http::client::http_client client(U(ip_addr_port));

    T *data;
    size_t npts = 1, ndims = 128, rounded_dim = 128;
    diskann::load_aligned_bin<T>(query_file, data, npts, ndims, rounded_dim);

    for (unsigned i = 0; i < nq; ++i)
    {
        T *vec = data + i * rounded_dim;
        web::http::http_request http_query(methods::POST);
        web::json::value queryJson = web::json::value::object();
        queryJson[QUERY_ID_KEY] = i;
        queryJson[K_KEY] = k_value;
        queryJson[L_KEY] = Ls;
        for (size_t i = 0; i < ndims; ++i)
        {
            queryJson[VECTOR_KEY][i] = web::json::value::number(vec[i]);
        }
        http_query.set_body(queryJson);

        client.request(http_query)
            .then([](web::http::http_response response) -> pplx::task<utility::string_t> {
                if (response.status_code() == status_codes::OK)
                {
                    return response.extract_string();
                }
                std::cerr << "Query failed" << std::endl;
                return pplx::task_from_result(utility::string_t());
            })
            .then([](pplx::task<utility::string_t> previousTask) {
                try
                {
                    std::cout << previousTask.get() << std::endl;
                }
                catch (http_exception const &e)
                {
                    std::wcout << e.what() << std::endl;
                }
            })
            .wait();
    }
}

int main(int argc, char *argv[])
{
    std::string data_type, query_file, address;
    uint32_t num_queries;
    uint32_t l_search, k_value;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("address", po::value<std::string>(&address)->required(), "Web server address");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "File containing the queries to search");
        desc.add_options()("num_queries,Q", po::value<uint32_t>(&num_queries)->required(),
                           "Number of queries to search");
        desc.add_options()("l_search", po::value<uint32_t>(&l_search)->required(), "Value of L");
        desc.add_options()("k_value,K", po::value<uint32_t>(&k_value)->default_value(10), "Value of K (default 10)");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    if (data_type == std::string("float"))
    {
        query_loop<float>(address, query_file, num_queries, l_search, k_value);
    }
    else if (data_type == std::string("int8"))
    {
        query_loop<int8_t>(address, query_file, num_queries, l_search, k_value);
    }
    else if (data_type == std::string("uint8"))
    {
        query_loop<uint8_t>(address, query_file, num_queries, l_search, k_value);
    }
    else
    {
        std::cerr << "Unsupported type " << argv[2] << std::endl;
        return -1;
    }

    return 0;
}