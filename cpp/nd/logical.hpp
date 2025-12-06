#pragma once

/**
 * @file logical.hpp
 * @brief Definitions of all logical operators.
 */

#include "array.hpp"

#include <utility>

namespace nd {

array logical_and(array f, array s);

array logical_or(array f, array s);

array logical_not(array f);

inline array operator&&(array f, array s)
{
    return logical_and(std::move(f), std::move(s));
}

inline array operator||(array f, array s)
{
    return logical_or(std::move(f), std::move(s));
}

inline array operator!(array f)
{
    return logical_not(std::move(f));
}

}
