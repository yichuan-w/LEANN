// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <restapi/server.h>
#include <restapi/in_memory_search.h>
#include <codecvt>
#include <iostream>

std::unique_ptr<Server> g_httpServer(nullptr);
std::unique_ptr<diskann::InMemorySearch> g_inMemorySearch(nullptr);

void setup(const utility::string_t &address)
{
    web::http::uri_builder uriBldr(address);
    auto uri = uriBldr.to_uri();

    std::wcout << L"Attempting to start server on " << uri.to_string() << std::endl;

    g_httpServer = std::unique_ptr<Server>(new Server(uri, g_inMemorySearch));
    g_httpServer->open().wait();

    ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t &address)
{
    g_httpServer->close().wait();
}

void loadIndex(const char *indexFile, const char *baseFile, const char *idsFile)
{
    auto nsgSearch = new diskann::InMemorySearch(baseFile, indexFile, idsFile, diskann::L2);
    g_inMemorySearch = std::unique_ptr<diskann::InMemorySearch>(nsgSearch);
}

std::wstring getHostingAddress(const char *hostNameAndPort)
{
    wchar_t buffer[4096];
    mbstowcs_s(nullptr, buffer, sizeof(buffer) / sizeof(buffer[0]), hostNameAndPort,
               sizeof(buffer) / sizeof(buffer[0]));
    return std::wstring(buffer);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "Usage: nsg_server <ip_addr_and_port> <index_file> "
                     "<base_file> <ids_file> "
                  << std::endl;
        exit(1);
    }

    auto address = getHostingAddress(argv[1]);
    loadIndex(argv[2], argv[3], argv[4]);
    while (1)
    {
        try
        {
            setup(address);
            std::cout << "Type 'exit' (case-sensitive) to exit" << std::endl;
            std::string line;
            std::getline(std::cin, line);
            if (line == "exit")
            {
                teardown(address);
                exit(0);
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Exception occurred: " << ex.what() << std::endl;
            std::cerr << "Restarting HTTP server";
            teardown(address);
        }
        catch (...)
        {
            std::cerr << "Unknown exception occurreed" << std::endl;
            std::cerr << "Restarting HTTP server";
            teardown(address);
        }
    }
}
