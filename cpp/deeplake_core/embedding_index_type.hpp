#pragma once

#include "basic_index_type.hpp"

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <string_view>

namespace deeplake_core {

class embedding_index_type
{
public:
    /// @name Types - For backward compatibility
    /// @{
    enum class type {
        clustered = static_cast<int>(deeplake_index_type::type::clustered),
        clustered_quantized = static_cast<int>(deeplake_index_type::type::clustered_quantized)
    };

    enum class quantization_type {
        none,
        binary_quantized
    };
    /// @}

public:
    constexpr inline embedding_index_type(type value = type::clustered) noexcept
        : impl_(static_cast<deeplake_index_type::type>(value))
    {
    }

    constexpr inline embedding_index_type(quantization_type value) noexcept
        : impl_(value == quantization_type::binary_quantized ? deeplake_index_type::type::clustered_quantized
                                                              : deeplake_index_type::type::clustered)
    {
    }

    constexpr inline embedding_index_type(deeplake_index_type::type value) noexcept
        : impl_(value)
    {
    }

    constexpr static std::string_view type_name() noexcept
    {
        return "embedding_index";
    }

    bool operator==(const embedding_index_type& other) const noexcept = default;

    static embedding_index_type from_legacy_json(const icm::const_json& json)
    {
        return embedding_index_type(basic_index_type<embedding_index_tag>::from_legacy_json(json));
    }

    static embedding_index_type from_json(const icm::const_json& json)
    {
        return embedding_index_type(basic_index_type<embedding_index_tag>::from_json(json));
    }

    icm::json to_json() const
    {
        return impl_.to_json();
    }

    std::string_view to_string() const noexcept
    {
        return impl_.to_string();
    }

    deeplake_index_type::type get_type() const noexcept
    {
        return impl_.get_type();
    }

    // Conversion operator to basic_index_type
    operator basic_index_type<embedding_index_tag>() const
    {
        return impl_;
    }

private:
    explicit embedding_index_type(basic_index_type<embedding_index_tag> impl)
        : impl_(impl)
    {
    }

    basic_index_type<embedding_index_tag> impl_;
};

} // namespace deeplake_core
