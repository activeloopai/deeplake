#pragma once

#include <cstdint>
#include <string_view>

namespace nd {

enum class unary_operator_type : uint8_t
{
    not_,
    minus,
    is_null,
};

enum class binary_operator_type : uint8_t
{
    plus,
    minus,
    asterisk,
    slash,
    percent,

    equals,
    not_equals,
    less,
    less_eq,
    greater,
    greater_eq,
    and_,
    or_,
    in,
    like,
    not_like,
    ilike,
};

std::string_view binary_operator_type_to_string(binary_operator_type type);

enum class ternary_operator_type : uint8_t
{
    between,
};

} // namespace nd
