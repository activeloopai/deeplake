#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class class_label_type
{
public:
    class_label_type(const nd::type& type)
        : type_(type)
    {
    }

    static class_label_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::label;
    }

    inline bool is_link() const noexcept
    {
        return false;
    }

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    bool operator==(const class_label_type& other) const
    {
        return type_ == other.type_;
    }

private:
    nd::type type_;
};

} // namespace deeplake_core
