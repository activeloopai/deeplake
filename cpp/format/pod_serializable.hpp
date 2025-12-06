#pragma once

#include "buffer.hpp"
#include "read_result.hpp"
#include "serializable.hpp"

#include <base/memory_buffer.hpp>
#include <base/span_cast.hpp>
#include <base/type_traits.hpp>

#include <type_traits>

namespace format {

/**
 * @brief Generic serialization for pod types.
 *
 * @tparam T Type.
 */
template <typename T>
requires((std::is_standard_layout_v<T> && std::is_trivial_v<T>) || base::arithmetic<T>)
struct serializable<T>
{
    inline static read_result<T> read(const base::memory_buffer& bytes, int64_t offset)
    {
        auto o = base::span_cast<const T>(bytes.span().subspan(offset))[0];
        return {o, offset + static_cast<int64_t>(sizeof(T))};
    }

    inline static int64_t output_size(const T&) noexcept
    {
        return static_cast<int64_t>(sizeof(T));
    }

    inline static void write(const T& o, buffer_t& bytes, int64_t offset)
    {
        auto view = base::span_cast<T>(buffer_view_t(bytes.data() + offset, sizeof(T)));
        view[0] = o;
    }
};

} // namespace format
