#pragma once

#include "assert.hpp"

#include <span>
#include <string_view>

namespace base {

template <typename To, typename From>
requires(sizeof(To) % sizeof(From) == 0 || sizeof(From) % sizeof(To) == 0)
inline std::span<To> span_cast(std::span<From> source)
{
    auto s = source.size() * sizeof(From) / sizeof(To);
    return std::span<To>(reinterpret_cast<To*>(source.data()), s);
}

inline std::span<const char> span_cast(std::string_view source)
{
    return std::span<const char>(source.data(), source.size());
}

template <typename T>
requires(sizeof(T) == sizeof(char))
inline std::string_view string_view_cast(std::span<const T> source)
{
    return std::string_view(span_cast<const char>(source).data(), source.size());
}

}
