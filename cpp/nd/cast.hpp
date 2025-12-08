#pragma once

/**
 * @file cast.hpp
 * @brief Definition of cast operators.
 */

#include "array.hpp"
#include "dtype.hpp"

namespace nd {

array cast(dtype t, array a);

template <dtype t>
inline array cast(array a)
{
    return cast(t, std::move(a));
}

}
