#pragma once

/**
 * @file generators.hpp
 * @brief declaration of functionalities to generate `nd::array`.
 */

#include "array.hpp"
#include "comparison.hpp"
#include <base/assert.hpp>

#include <cmath>

namespace nd
{

/**
 * @brief single dimension array filled with integral sequence starting from `start` and ending at `stop`.
 *
 * @param start
 * @param stop
 * @return array
 */

template <typename T>
requires std::is_arithmetic_v<T>
array arange(T start, T stop, T step = 1)
{
    auto compare = [](T x, T y) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(x - y) <
                   100 * std::numeric_limits<T>::epsilon();
        } else {
            return x == y;
        }
    };

    ASSERT_MESSAGE(step != 0, "nd::arange step cannot be zero");
    ASSERT_MESSAGE((stop - start) * step >= 0, "nd::arange step direction is not compatible with start and stop");

    icm::vector<T> a;
    while (!compare(start, stop) &&
           ((std::isless(start, stop) && step > 0) || (std::isgreater(start, stop) && step < 0))) {
        a.push_back(start);
        start += step;
    }

    return adapt(std::move(a));
}

} // namespace nd
