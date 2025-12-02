#pragma once

/**
 * @file arithmetic.hpp
 * @brief Definitions of arithmetic operators.
 */

#include "array.hpp"

#include <utility>

namespace nd {

array plus(array f, array s);

array minus(array f, array s);

array multiply(array f, array s);

array divide(array f, array s);

array percent(array f, array s);

array unary_minus(array f);

inline array operator+(array f, array s)
{
    return plus(std::move(f), std::move(s));
}

inline array operator-(array f, array s)
{
    return minus(std::move(f), std::move(s));
}

inline array operator*(array f, array s)
{
    return multiply(std::move(f), std::move(s));
}

inline array operator/(array f, array s)
{
    return divide(std::move(f), std::move(s));
}

inline array operator%(array f, array s)
{
    return percent(std::move(f), std::move(s));
}

inline array operator-(array f)
{
    return unary_minus(std::move(f));
}

}
