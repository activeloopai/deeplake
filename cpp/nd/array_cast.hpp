#pragma once

#include "array.hpp"

namespace nd {

template <typename T>
const T* dynamic_array_cast(const array& a)
{
    return a.dynamic_cast_<T>();
}

template <typename T>
const T* static_array_cast(const array& a)
{
    return a.static_cast_<T>();
}

} // namespace nd
