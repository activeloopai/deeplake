#pragma once

/**
 * @file eval.hpp
 * @brief Definition of `eval` function.
 */

#include "array.hpp"

namespace nd {

array eval(array arr);

void copy_data(const array& arr, std::span<uint8_t> buffer);

}
