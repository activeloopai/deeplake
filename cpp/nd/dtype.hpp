#pragma once

/**
 * @file dtype.hpp
 * @brief Definition of the `dtype` enum and related utilities.
 */

#include <base/f16.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace icm {
class const_json;
class json;
} // namespace icm

namespace nd {

class dict;

enum class dtype : uint8_t
{
    boolean,
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    string,
    object,
    byte,
    bfloat16,
    float16,
    unknown
};

std::string_view dtype_to_str(dtype d);
dtype dtype_from_str(std::string_view s);

dtype dtype_from_json(const icm::json&);
dtype dtype_from_json(const icm::const_json&);

//////////////
/// dtype_enum
//////////////
template<typename T>
struct dtype_enum;

template<>
struct dtype_enum<bool> {
    constexpr static auto value = dtype::boolean;
};

template<>
struct dtype_enum<char> {
    constexpr static auto value = dtype::int8;
};

template<>
struct dtype_enum<uint8_t> {
    constexpr static auto value = dtype::uint8;
};

template<>
struct dtype_enum<uint16_t> {
    constexpr static auto value = dtype::uint16;
};

template<>
struct dtype_enum<uint32_t> {
    constexpr static auto value = dtype::uint32;
};

template<>
struct dtype_enum<uint64_t> {
    constexpr static auto value = dtype::uint64;
};

template<>
struct dtype_enum<int8_t> {
    constexpr static auto value = dtype::int8;
};

template<>
struct dtype_enum<int16_t> {
    constexpr static auto value = dtype::int16;
};

template<>
struct dtype_enum<int32_t> {
    constexpr static auto value = dtype::int32;
};

template<>
struct dtype_enum<int64_t> {
    constexpr static auto value = dtype::int64;
};

template<>
struct dtype_enum<base::bfloat16> {
    constexpr static auto value = dtype::bfloat16;
};

template<>
struct dtype_enum<base::half> {
    constexpr static auto value = dtype::float16;
};

template<>
struct dtype_enum<float> {
    constexpr static auto value = dtype::float32;
};

template<>
struct dtype_enum<double> {
    constexpr static auto value = dtype::float64;
};

template<>
struct dtype_enum<std::string_view> {
    constexpr static auto value = dtype::string;
};

template<>
struct dtype_enum<std::string> {
    constexpr static auto value = dtype::string;
};

template<>
struct dtype_enum<dict> {
    constexpr static auto value = dtype::object;
};

template<>
struct dtype_enum<std::span<const uint8_t>> {
    constexpr static auto value = dtype::byte;
};

template <typename T>
constexpr auto dtype_enum_v = dtype_enum<std::remove_cvref_t<T>>::value;

template <typename... Types>
struct active_type_helper;

template <typename... Types>
struct active_type_helper<std::variant<Types...>> {
    template <typename Variant>
    static constexpr auto get(const Variant& var) {
        return std::visit([](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            return dtype_enum_v<T>;
        }, var);
    }
};

template <typename... Types>
struct dtype_enum<std::variant<Types...>> {
    template <typename Variant>
    static constexpr auto value(const Variant& var) {
        return active_type_helper<std::variant<Types...>>::get(var);
    }
};

template <typename... Types>
constexpr auto dtype_enum_v<std::variant<Types...>> = dtype_enum<std::variant<Types...>>::template value<std::variant<Types...>>;

//////////////
/// dtype_type
//////////////
template<dtype t>
struct dtype_type;

template<>
struct dtype_type<dtype::boolean> {
    using type = bool;
};

template<>
struct dtype_type<dtype::uint8> {
    using type = uint8_t;
};

template<>
struct dtype_type<dtype::uint16> {
    using type = uint16_t;
};

template<>
struct dtype_type<dtype::uint32> {
    using type = uint32_t;
};

template<>
struct dtype_type<dtype::uint64> {
    using type = uint64_t;
};

template<>
struct dtype_type<dtype::int8> {
    using type = int8_t;
};

template<>
struct dtype_type<dtype::int16> {
    using type = int16_t;
};

template<>
struct dtype_type<dtype::int32> {
    using type = int32_t;
};

template<>
struct dtype_type<dtype::int64> {
    using type = int64_t;
};

template<>
struct dtype_type<dtype::bfloat16> {
    using type = base::bfloat16;
};

template<>
struct dtype_type<dtype::float16> {
    using type = base::half;
};

template<>
struct dtype_type<dtype::float32> {
    using type = float;
};

template<>
struct dtype_type<dtype::float64> {
    using type = double;
};

template<>
struct dtype_type<dtype::string> {
    using type = std::string_view;
};

template <>
struct dtype_type<dtype::object> {
    using type = dict;
};

template <>
struct dtype_type<dtype::byte> {
    using type = std::span<const uint8_t>;
};

template <dtype t>
using dtype_type_t = typename dtype_type<t>::type;

dtype common_dtype(dtype f, dtype s);
dtype float_dtype(dtype dt);

constexpr bool dtype_is_numeric(dtype t)
{
    return t != dtype::string && t != dtype::object && t != dtype::byte && t != dtype::unknown;
}

constexpr bool can_cast_dtype(dtype from, dtype to)
{
    return (from == to) ||
           (from != dtype::string && to != dtype::string && from != dtype::object && to != dtype::object &&
            from != dtype::byte && to != dtype::byte && from != dtype::unknown && to != dtype::unknown);
}

constexpr bool can_compare(dtype f, dtype s)
{
    return !((dtype_is_numeric(f) && s == dtype::string) || (dtype_is_numeric(s) && f == dtype::string));
}

} // namespace nd

#include "switch_dtype.hpp"
