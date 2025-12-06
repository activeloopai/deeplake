#pragma once

#include "exceptions.hpp"

namespace nd {

template <typename F>
auto switch_numeric_dtype(dtype t, F f)
{
    switch (t) {
    case nd::dtype::boolean:
        return f.template operator()<bool>();
    case nd::dtype::uint8:
        return f.template operator()<uint8_t>();
    case nd::dtype::uint16:
        return f.template operator()<uint16_t>();
    case nd::dtype::uint32:
        return f.template operator()<uint32_t>();
    case nd::dtype::uint64:
        return f.template operator()<uint64_t>();
    case nd::dtype::int8:
        return f.template operator()<int8_t>();
    case nd::dtype::int16:
        return f.template operator()<int16_t>();
    case nd::dtype::int32:
        return f.template operator()<int32_t>();
    case nd::dtype::int64:
        return f.template operator()<int64_t>();
    case nd::dtype::bfloat16:
        return f.template operator()<base::bfloat16>();
    case nd::dtype::float16:
        return f.template operator()<base::half>();
    case nd::dtype::float32:
        return f.template operator()<float>();
    case nd::dtype::float64:
        return f.template operator()<double>();
    case nd::dtype::string:
        throw non_numeric_dtype(dtype_to_str(nd::dtype::string));
    case nd::dtype::object:
        throw non_numeric_dtype(dtype_to_str(nd::dtype::object));
    case nd::dtype::byte:
        throw non_numeric_dtype(dtype_to_str(nd::dtype::byte));
    case nd::dtype::unknown:
    default:
        throw unknown_dtype();
    }
}

template <typename F>
auto switch_dtype(dtype t, F f)
{
    switch (t) {
    case nd::dtype::boolean:
        return f.template operator()<bool>();
    case nd::dtype::uint8:
        return f.template operator()<uint8_t>();
    case nd::dtype::uint16:
        return f.template operator()<uint16_t>();
    case nd::dtype::uint32:
        return f.template operator()<uint32_t>();
    case nd::dtype::uint64:
        return f.template operator()<uint64_t>();
    case nd::dtype::int8:
        return f.template operator()<int8_t>();
    case nd::dtype::int16:
        return f.template operator()<int16_t>();
    case nd::dtype::int32:
        return f.template operator()<int32_t>();
    case nd::dtype::int64:
        return f.template operator()<int64_t>();
    case nd::dtype::bfloat16:
        return f.template operator()<base::bfloat16>();
    case nd::dtype::float16:
        return f.template operator()<base::half>();
    case nd::dtype::float32:
        return f.template operator()<float>();
    case nd::dtype::float64:
        return f.template operator()<double>();
    case nd::dtype::string:
        return f.template operator()<std::string_view>();
    case nd::dtype::object:
        return f.template operator()<nd::dict>();
    case nd::dtype::byte:
        return f.template operator()<std::span<const uint8_t>>();
    case nd::dtype::unknown:
    default:
        throw unknown_dtype();
    }
}

inline size_t dtype_bytes(dtype type)
{
    if (dtype_is_numeric(type)) {
        return switch_numeric_dtype(type, []<typename T>() {
            return sizeof(T);
        });
    }
    return 1;
}

}
