#pragma once

#include <base/format.hpp>

#include <cstdint>
#include <utility>

namespace icm {

template <typename T>
requires std::is_integral_v<T> && std::is_signed_v<T>
inline T parse_negative_index(T index, T size)
{
    if (index < -size || index >= size) {
        throw std::out_of_range(
            fmt::format("Index {} is out of range. Allowed range is from {} to {}.", index, -size, size - 1));
    }
    if (index < 0) {
        return size + index;
    }
    return index;
}

template <typename T>
requires std::is_integral_v<T> && std::is_signed_v<T>
inline std::pair<T, T> parse_negative_index(T start, T stop, T size)
{
    if (size != std::numeric_limits<T>::max()) {
        size += 1;
    }
    auto start_index = parse_negative_index(start, size);
    auto stop_index = parse_negative_index(stop, size);
    if (start_index > stop_index) {
        std::swap(start_index, stop_index);
    }
    return {start_index, stop_index};
}

} // namespace icm
