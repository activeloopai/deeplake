#pragma once

/**
 * @file small_vector.hpp
 * @brief Definition of small_vector template.
 */

#include <boost/container/small_vector.hpp>

#include <cstddef>

namespace icm {

template <typename T, std::size_t N = 4>
using small_vector = boost::container::small_vector<T, N>;

}
