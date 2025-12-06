#pragma once

#include <icm/const_json.hpp>

#include <string>
#include <variant>

namespace query_core {

using order_result = std::variant<int, float, std::string, icm::const_json>;

using numeric_variant = std::variant<int, float>;

} // namespace query_core
