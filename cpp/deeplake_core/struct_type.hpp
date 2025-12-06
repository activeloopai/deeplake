#pragma once

#include "schema.hpp"

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/array.hpp>

#include <memory>
#include <variant>

namespace deeplake_core {

class convert_context;

class struct_type
{
public:
    struct_type() = delete;

    explicit struct_type(deeplake_core::schema&& s)
        : schema_(std::make_shared<deeplake_core::schema>(std::move(s)))
    {
    }

    static struct_type from_json(const icm::const_json& j);

    [[nodiscard]] icm::json to_json() const;

    base::htype htype() const noexcept
    {
        return base::htype::struct_;
    }

    constexpr bool is_link() const noexcept
    {
        return false;
    }

    nd::type data_type() const;

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    bool operator==(const struct_type& other) const noexcept;

private:
    std::shared_ptr<deeplake_core::schema> schema_;
};

} // namespace deeplake_core