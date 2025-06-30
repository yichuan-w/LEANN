// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <codecvt>
#include <limits>

#include <restapi/server.h>

namespace diskann
{

Server::Server(web::uri &uri, std::vector<std::unique_ptr<diskann::BaseSearch>> &multi_searcher,
               const std::string &typestring)
    : _multi_search(multi_searcher.size() > 1 ? true : false)
{
    for (auto &searcher : multi_searcher)
        _multi_searcher.push_back(std::move(searcher));

    _listener = std::unique_ptr<web::http::experimental::listener::http_listener>(
        new web::http::experimental::listener::http_listener(uri));
    if (typestring == std::string("float"))
    {
        _listener->support(std::bind(&Server::handle_post<float>, this, std::placeholders::_1));
    }
    else if (typestring == std::string("int8_t"))
    {
        _listener->support(web::http::methods::POST,
                           std::bind(&Server::handle_post<int8_t>, this, std::placeholders::_1));
    }
    else if (typestring == std::string("uint8_t"))
    {
        _listener->support(web::http::methods::POST,
                           std::bind(&Server::handle_post<uint8_t>, this, std::placeholders::_1));
    }
    else
    {
        throw "Unsupported type in server constuctor";
    }
}

Server::~Server()
{
}

pplx::task<void> Server::open()
{
    return _listener->open();
}
pplx::task<void> Server::close()
{
    return _listener->close();
}

diskann::SearchResult Server::aggregate_results(const unsigned K, const std::vector<diskann::SearchResult> &results)
{
    if (_multi_search)
    {
        auto best_indices = new unsigned[K];
        auto best_distances = new float[K];
        auto best_partitions = new unsigned[K];
        auto best_tags = results[0].tags_enabled() ? new std::string[K] : nullptr;

        auto numsearchers = _multi_searcher.size();
        std::vector<size_t> pos(numsearchers, 0);

        for (size_t k = 0; k < K; ++k)
        {
            float best_distance = std::numeric_limits<float>::max();
            unsigned best_partition = 0;

            for (size_t i = 0; i < numsearchers; ++i)
            {
                if (results[i].get_distances()[pos[i]] < best_distance)
                {
                    best_distance = results[i].get_distances()[pos[i]];
                    best_partition = i;
                }
            }
            best_distances[k] = best_distance;
            best_indices[k] = results[best_partition].get_indices()[pos[best_partition]];
            best_partitions[k] = best_partition;
            if (results[best_partition].tags_enabled())
                best_tags[k] = results[best_partition].get_tags()[pos[best_partition]];
            std::cout << best_partition << " " << pos[best_partition] << std::endl;
            pos[best_partition]++;
        }

        unsigned int total_time = 0;
        for (size_t i = 0; i < numsearchers; ++i)
            total_time += results[i].get_time();
        diskann::SearchResult result =
            SearchResult(K, total_time, best_indices, best_distances, best_tags, best_partitions);

        delete[] best_indices;
        delete[] best_distances;
        delete[] best_partitions;
        delete[] best_tags;

        return result;
    }
    else
    {
        return results[0];
    }
}

template <class T> void Server::handle_post(web::http::http_request message)
{
    message.extract_string(true)
        .then([=](utility::string_t body) {
            int64_t queryId = -1;
            unsigned int K = 0;
            try
            {
                T *queryVector = nullptr;
                unsigned int dimensions = 0;
                unsigned int Ls;
                parseJson(body, K, queryId, queryVector, dimensions, Ls);

                auto startTime = std::chrono::high_resolution_clock::now();
                std::vector<diskann::SearchResult> results;

                for (auto &searcher : _multi_searcher)
                    results.push_back(searcher->search(queryVector, dimensions, (unsigned int)K, Ls));
                diskann::SearchResult result = aggregate_results(K, results);
                diskann::aligned_free(queryVector);
                web::json::value response = prepareResponse(queryId, K);
                response[INDICES_KEY] = idsToJsonArray(result);
                response[DISTANCES_KEY] = distancesToJsonArray(result);
                if (result.tags_enabled())
                    response[TAGS_KEY] = tagsToJsonArray(result);
                if (result.partitions_enabled())
                    response[PARTITION_KEY] = partitionsToJsonArray(result);

                response[TIME_TAKEN_KEY] = std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::high_resolution_clock::now() - startTime)
                                               .count();

                std::cout << "Responding to: " << queryId << std::endl;
                return std::make_pair(web::http::status_codes::OK, response);
            }
            catch (const std::exception &ex)
            {
                std::cerr << "Exception while processing query: " << queryId << ":" << ex.what() << std::endl;
                web::json::value response = prepareResponse(queryId, K);
                response[ERROR_MESSAGE_KEY] = web::json::value::string(ex.what());
                return std::make_pair(web::http::status_codes::InternalError, response);
            }
            catch (...)
            {
                std::cerr << "Uncaught exception while processing query: " << queryId;
                web::json::value response = prepareResponse(queryId, K);
                response[ERROR_MESSAGE_KEY] = web::json::value::string(UNKNOWN_ERROR);
                return std::make_pair(web::http::status_codes::InternalError, response);
            }
        })
        .then([=](std::pair<short unsigned int, web::json::value> response_status) {
            try
            {
                message.reply(response_status.first, response_status.second).wait();
            }
            catch (const std::exception &ex)
            {
                std::cerr << "Exception while processing reply: " << ex.what() << std::endl;
            };
        });
}

web::json::value Server::prepareResponse(const int64_t &queryId, const int k)
{
    web::json::value response = web::json::value::object();
    response[QUERY_ID_KEY] = queryId;
    response[K_KEY] = k;

    return response;
}

template <class T>
void Server::parseJson(const utility::string_t &body, unsigned int &k, int64_t &queryId, T *&queryVector,
                       unsigned int &dimensions, unsigned &Ls)
{
    std::cout << body << std::endl;
    web::json::value val = web::json::value::parse(body);
    web::json::array queryArr = val.at(VECTOR_KEY).as_array();
    queryId = val.has_field(QUERY_ID_KEY) ? val.at(QUERY_ID_KEY).as_number().to_int64() : -1;
    Ls = val.has_field(L_KEY) ? val.at(L_KEY).as_number().to_uint32() : DEFAULT_L;
    k = val.at(K_KEY).as_integer();

    if (k <= 0 || k > Ls)
    {
        throw new std::invalid_argument("Num of expected NN (k) must be greater than zero and less than or "
                                        "equal to Ls.");
    }
    if (queryArr.size() == 0)
    {
        throw new std::invalid_argument("Query vector has zero elements.");
    }

    dimensions = static_cast<unsigned int>(queryArr.size());
    unsigned new_dim = ROUND_UP(dimensions, 8);
    diskann::alloc_aligned((void **)&queryVector, new_dim * sizeof(T), 8 * sizeof(T));
    memset(queryVector, 0, new_dim * sizeof(float));
    for (size_t i = 0; i < queryArr.size(); i++)
    {
        queryVector[i] = (float)queryArr[i].as_double();
    }
}

template <typename T>
web::json::value Server::toJsonArray(const std::vector<T> &v, std::function<web::json::value(const T &)> valConverter)
{
    web::json::value rslts = web::json::value::array();
    for (size_t i = 0; i < v.size(); i++)
    {
        auto jsonVal = valConverter(v[i]);
        rslts[i] = jsonVal;
    }
    return rslts;
}

web::json::value Server::idsToJsonArray(const diskann::SearchResult &result)
{
    web::json::value idArray = web::json::value::array();
    auto ids = result.get_indices();
    for (size_t i = 0; i < ids.size(); i++)
    {
        auto idVal = web::json::value::number(ids[i]);
        idArray[i] = idVal;
    }
    std::cout << "Vector size: " << ids.size() << std::endl;
    return idArray;
}

web::json::value Server::distancesToJsonArray(const diskann::SearchResult &result)
{
    web::json::value distArray = web::json::value::array();
    auto distances = result.get_distances();
    for (size_t i = 0; i < distances.size(); i++)
    {
        distArray[i] = web::json::value::number(distances[i]);
    }
    return distArray;
}

web::json::value Server::tagsToJsonArray(const diskann::SearchResult &result)
{
    web::json::value tagArray = web::json::value::array();
    auto tags = result.get_tags();
    for (size_t i = 0; i < tags.size(); i++)
    {
        tagArray[i] = web::json::value::string(tags[i]);
    }
    return tagArray;
}

web::json::value Server::partitionsToJsonArray(const diskann::SearchResult &result)
{
    web::json::value partitionArray = web::json::value::array();
    auto partitions = result.get_partitions();
    for (size_t i = 0; i < partitions.size(); i++)
    {
        partitionArray[i] = web::json::value::number(partitions[i]);
    }
    return partitionArray;
}
}; // namespace diskann