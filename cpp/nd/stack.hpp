#pragma once

#include "array.hpp"

#include <memory>

namespace nd {

/**
 * @brief Stacks two arrays vertically (by axis 0), so that second becomes continuation of the first.
 * 
 * @param a First array.
 * @param b Second array. 
 * @return array Result array.
 */
array vstack(array a, array b);

/**
 * @brief Stacks list of arrays vertically (by axis 0), so that second becomes continuation of the first.
 * 
 * @param a List of arrays.
 * @return array Result array.
 */
array vstack(std::vector<array>&& a, std::shared_ptr<void>&& owner = nullptr);

}
