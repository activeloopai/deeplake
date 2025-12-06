#pragma once

#include "array.hpp"

#include <filesystem>
#include <fstream>

namespace nd {

/**
 * @brief Writes big `array` into file incrementally by chunks. This class is helpful, to fetch and cache
 * big arrays which do not fit into physical memory. The array meta information - shape and dtype should be known
 * upfront.
 */
class array_stream_writer
{
public:
    array_stream_writer(icm::shape sh, dtype d, bool is_dynamic,
                        std::ostream& stream);

    array_stream_writer(const array_stream_writer&) = delete;
    array_stream_writer& operator=(const array_stream_writer&) = delete;
    array_stream_writer(array_stream_writer&&) = delete;
    array_stream_writer& operator=(array_stream_writer&&) = delete;
    ~array_stream_writer() = default;

public:
    void add_chunk(const array& a);
    void finalize();

private:
    void write_header();

private:
    std::ostream& stream_;
    icm::shape shape_;
    int64_t current_volume_written_ = 0;
    int64_t volume_;
    dtype dtype_;
    bool is_dynamic_;
};

}
