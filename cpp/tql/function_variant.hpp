#pragma once

#include <nd/array.hpp>

#include <functional>
#include <variant>

namespace tql {

/**
 * @brief This declaration is used as a common api for user defined and external functions.
 */
using function_variant = std::variant<std::function<nd::array()>,
                                      std::function<nd::array(const nd::array&)>,
                                      std::function<nd::array(const nd::array&, const nd::array&)>>;
}
