#pragma once
#ifdef __APPLE__
#include "aligned_file_reader.h"

class AppleAlignedFileReader : public AlignedFileReader
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;

  public:
    AppleAlignedFileReader();
    ~AppleAlignedFileReader();

    IOContext &get_ctx();

    void register_thread();
    void deregister_thread();
    void deregister_all_threads();

    void open(const std::string &fname);
    void close();

    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false);
};
#endif
