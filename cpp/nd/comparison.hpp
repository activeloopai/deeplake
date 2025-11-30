#pragma once

/**
 * @file comparison.hpp
 * @brief Definitions of comparison operators.
 */

#include "array.hpp"

#include <utility>

namespace nd
{

array equal(array f, array s);

array isclose(array a, array b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);

array not_equal(array f, array s);

array less(array f, array s);

array greater(array f, array s);

array less_equal(array f, array s);

array greater_equal(array f, array s);

inline array operator==(array f, array s)
{
    return equal(std::move(f), std::move(s));
}

inline array operator!=(array f, array s)
{
    return not_equal(std::move(f), std::move(s));
}

inline array operator<(array f, array s)
{
    return less(std::move(f), std::move(s));
}

inline array operator<=(array f, array s)
{
    return less_equal(std::move(f), std::move(s));
}

inline array operator>(array f, array s)
{
    return greater(std::move(f), std::move(s));
}

inline array operator>=(array f, array s)
{
    return greater_equal(std::move(f), std::move(s));
}

} // namespace nd
