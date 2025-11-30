#pragma once

#include <cstdint>
#include <tuple>

namespace format {

/**
 * @brief Represents the result of the read operation:
 * - The object
 * - The new offset of the buffer
 *
 * @tparam T
 */
template <typename T>
struct read_result
{
    T object;
    int64_t offset;

    operator std::tuple<T&, int64_t&>() // to use with std::tie
    {
        return {object, offset};
    }
};

} // namespace format