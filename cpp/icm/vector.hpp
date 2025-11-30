#pragma once

/**
 * @file vector.hpp
 * @brief Definition of vector template.
 */

#include <boost/container/vector.hpp>

namespace icm {

template <typename T, typename A = void, typename Options = void> using vector = boost::container::vector<T, A, Options>;

}
