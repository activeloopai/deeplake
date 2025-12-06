#pragma once

#include "../array.hpp"

#include <base/memory_buffer.hpp>
#include <format/read_result.hpp>
#include <format/serializable.hpp>
#include <format/buffer.hpp>

namespace format {

template <>
struct serializable<nd::array>
{
    static read_result<nd::array> read(const base::memory_buffer& buffer, int64_t offset);

    static buffer_t write(const nd::array& a);
};

} /// format namespace
