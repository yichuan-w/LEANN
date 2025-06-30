// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define MAX_IO_DEPTH 128

#include <vector>
#include <atomic>

#ifdef __linux__
#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
#include <malloc.h>
typedef io_context_t IOContext;
#elif __APPLE__
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>

struct IOContext
{
    int fd;
    dispatch_io_t channel;
    dispatch_queue_t queue;
    dispatch_group_t grp;
};
#elif _WINDOWS
#include <Windows.h>
#include <minwinbase.h>
#include <malloc.h>

#ifndef USE_BING_INFRA
struct IOContext
{
    HANDLE fhandle = NULL;
    HANDLE iocp = NULL;
    std::vector<OVERLAPPED> reqs;
};
#else
#include "IDiskPriorityIO.h"
#include <atomic>
// TODO: Caller code is very callous about copying IOContext objects
// all over the place. MUST verify that it won't cause leaks/logical
// errors.
// Because of such callous copying, we have to use ptr->atomic instead
// of atomic, as atomic is not copyable.
struct IOContext
{
    enum Status
    {
        READ_WAIT = 0,
        READ_SUCCESS,
        READ_FAILED,
        PROCESS_COMPLETE
    };

    std::shared_ptr<ANNIndex::IDiskPriorityIO> m_pDiskIO = nullptr;
    std::shared_ptr<std::vector<ANNIndex::AsyncReadRequest>> m_pRequests;
    std::shared_ptr<std::vector<Status>> m_pRequestsStatus;

    // waitonaddress on this memory to wait for IO completion signal
    // reader should signal this memory after IO completion
    // TODO: WindowsAlignedFileReader can be modified to take advantage of this
    //   and can largely share code with the file reader for Bing.
    mutable volatile long m_completeCount = 0;

    IOContext()
        : m_pRequestsStatus(new std::vector<Status>()), m_pRequests(new std::vector<ANNIndex::AsyncReadRequest>())
    {
        (*m_pRequestsStatus).reserve(MAX_IO_DEPTH);
        (*m_pRequests).reserve(MAX_IO_DEPTH);
    }
};
#endif

#endif

#include <cstdio>
#include <mutex>
#include <thread>
#include "tsl/robin_map.h"
#include "utils.h"

// NOTE :: all 3 fields must be 512-aligned
struct AlignedRead
{
    uint64_t offset; // where to read from
    uint64_t len;    // how much to read
    void *buf;       // where to read into

    AlignedRead() : offset(0), len(0), buf(nullptr)
    {
    }

    AlignedRead(uint64_t offset, uint64_t len, void *buf) : offset(offset), len(len), buf(buf)
    {
        assert(IS_512_ALIGNED(offset));
        assert(IS_512_ALIGNED(len));
        assert(IS_512_ALIGNED(buf));
        // assert(malloc_usable_size(buf) >= len);
    }
};

class AlignedFileReader
{
  protected:
    tsl::robin_map<std::thread::id, IOContext> ctx_map;
    std::mutex ctx_mut;

  public:
    // returns the thread-specific context
    // returns (io_context_t)(-1) if thread is not registered
    virtual IOContext &get_ctx() = 0;

    virtual ~AlignedFileReader(){};

    // register thread-id for a context
    virtual void register_thread() = 0;
    // de-register thread-id for a context
    virtual void deregister_thread() = 0;
    virtual void deregister_all_threads() = 0;

    // Open & close ops
    // Blocking calls
    virtual void open(const std::string &fname) = 0;
    virtual void close() = 0;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    virtual void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) = 0;

#ifdef USE_BING_INFRA
    // wait for completion of one request in a batch of requests
    virtual void wait(IOContext &ctx, int &completedIndex) = 0;
#endif
};
