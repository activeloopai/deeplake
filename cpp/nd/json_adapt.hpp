#pragma once

/**
 * @file adapt.hpp
 * @brief Definitions and implementations of `adapt`, `adapt_shape`, `empty`, `dynamic_empty`, `dynamic` functions.
 */

#include <icm/const_json.hpp>

namespace nd {

class array;

array adapt(const icm::const_json& j);

}
