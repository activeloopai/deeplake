#pragma once

#include <base/assert.hpp>

#include <algorithm>
#include <string>
#include <string_view>

namespace deeplake_core {

struct deeplake_index_type
{
public:
    enum class type
    {
        none,
        inverted_index,
        bm25,
        exact_text,
        pooled_quantized,
        clustered,
        clustered_quantized
    };

    static std::string_view to_string(type value) noexcept
    {
        switch (value) {
        case type::none:
            return "none";
        case type::inverted_index:
            return "inverted_index";
        case type::bm25:
            return "bm25";
        case type::exact_text:
            return "exact_text";
        case type::pooled_quantized:
            return "pooled_quantized";
        case type::clustered:
            return "clustered";
        case type::clustered_quantized:
            return "clustered_quantized";
        default:
            break;
        }
        ASSERT_MESSAGE(false, "Unknown index type");
        return "unknown";
    }

    static type from_string(std::string_view s) noexcept
    {
        std::string lower_s(s);
        std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), [](unsigned char c) {
            return std::tolower(c);
        });

        if (lower_s == "inverted" || lower_s == "inverted_index") {
            return type::inverted_index;
        } else if (lower_s == "bm25") {
            return type::bm25;
        } else if (lower_s == "exact_text") {
            return type::exact_text;
        } else if (lower_s == "pooled_quantized") {
            return type::pooled_quantized;
        } else if (lower_s == "clustered") {
            return type::clustered;
        } else if (lower_s == "clustered_quantized") {
            return type::clustered_quantized;
        }
        return type::none;
    }
};

} // namespace deeplake_core
