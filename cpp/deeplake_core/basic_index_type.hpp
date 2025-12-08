#pragma once

#include "deeplake_index_type.hpp"

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <string>

namespace deeplake_core {

struct text_index_tag
{
};

struct json_index_tag
{
};

struct numeric_index_tag
{
};

struct embeddings_matrix_index_tag
{
};

struct embedding_index_tag
{
};

template <typename Tag>
class basic_index_type
{
public:
    constexpr inline basic_index_type(deeplake_index_type::type value = deeplake_index_type::type::none) noexcept
        : value_(value)
    {
    }

    deeplake_index_type::type get_type() const noexcept
    {
        return value_;
    }

    consteval static std::string_view type_name() noexcept;

    bool operator==(const basic_index_type& other) const noexcept = default;

    static basic_index_type from_json(const icm::const_json& json)
    {
        return deeplake_index_type::from_string(json.get<std::string>());
    }

    static basic_index_type from_legacy_json(const icm::const_json& json)
    {
        return from_json(json);
    }

    icm::json to_json() const
    {
        return icm::json(to_string());
    }

    std::string_view to_string() const noexcept
    {
        return deeplake_index_type::to_string(value_);
    }

private:
    deeplake_index_type::type value_;
};

// Specializations for type_name()
template <>
consteval std::string_view basic_index_type<text_index_tag>::type_name() noexcept
{
    return "text_index";
}

template <>
consteval std::string_view basic_index_type<json_index_tag>::type_name() noexcept
{
    return "json_index";
}

template <>
consteval std::string_view basic_index_type<numeric_index_tag>::type_name() noexcept
{
    return "numeric_index";
}

template <>
consteval std::string_view basic_index_type<embeddings_matrix_index_tag>::type_name() noexcept
{
    return "embeddings_matrix_index";
}

template <>
inline std::string_view basic_index_type<embeddings_matrix_index_tag>::to_string() const noexcept
{
    return "EmbeddingsMatrixIndexType";
}

template <>
inline icm::json basic_index_type<embeddings_matrix_index_tag>::to_json() const
{
    return icm::json("pooled_quantized");
}

template <>
constexpr inline basic_index_type<embeddings_matrix_index_tag>::basic_index_type(
    deeplake_index_type::type value) noexcept
    : value_(value == deeplake_index_type::type::none ? deeplake_index_type::type::pooled_quantized : value)
{
}

template <>
consteval std::string_view basic_index_type<embedding_index_tag>::type_name() noexcept
{
    return "embedding_index";
}

template <>
inline icm::json basic_index_type<embedding_index_tag>::to_json() const
{
    return icm::json(deeplake_index_type::to_string(value_));
}

template <>
constexpr inline basic_index_type<embedding_index_tag>::basic_index_type(deeplake_index_type::type value) noexcept
    : value_(value == deeplake_index_type::type::none ? deeplake_index_type::type::clustered : value)
{
}

template <>
inline basic_index_type<embedding_index_tag>
basic_index_type<embedding_index_tag>::from_legacy_json(const icm::const_json& json)
{
    const auto s = json.get<std::string>();
    if (s == "binary_quantized" || s == "quantized") {
        return deeplake_index_type::type::clustered_quantized;
    }
    return deeplake_index_type::type::clustered;
}

using text_index_type = basic_index_type<text_index_tag>;
using json_index_type = basic_index_type<json_index_tag>;
using numeric_index_type = basic_index_type<numeric_index_tag>;
using embeddings_matrix_index_type = basic_index_type<embeddings_matrix_index_tag>;

} // namespace deeplake_core
