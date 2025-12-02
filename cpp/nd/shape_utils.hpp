#pragma once

/**
 * @file shape_utils.hpp
 * @brief Definitions of `shapes_equal` and `shape_interval` utilities.
 */

#include "array.hpp"
#include "shape_interval.hpp"

#include <icm/small_vector.hpp>

namespace nd {

template <typename T, typename U>
requires(!std::is_same_v<T, array>)
bool shapes_equal(const T& s1, const U& s2)
{
    if (s1.size() != s2.size()) {
        return false;
    }
    for (auto i = 0; i < s1.size(); ++i) {
        if (s1[i] != s2[i]) {
            return false;
        }
    }
    return true;
}


inline bool shapes_equal(const array& a, const array& b)
{
    auto const& s1 = a.shape();
    auto const& s2 = b.shape();
    return shapes_equal(s1, s2);
}

shape_interval_t<icm::shape::value_type> shape_interval(const array& a);

}
