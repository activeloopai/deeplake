#pragma once

#include "basic_index_type.hpp"
#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class dict_type
{
public:
    dict_type() = default;

    dict_type(json_index_type index_type)
        : index_type_(index_type)
    {
    }

    static dict_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const;

    base::htype htype() const noexcept
    {
        return base::htype::json;
    }

    inline bool is_link() const noexcept
    {
        return false;
    }

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    nd::array convert_batch_to_compact(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    json_index_type index_type() const
    {
        return index_type_;
    }

    bool operator==(const dict_type&) const = default;

private:
    json_index_type index_type_ = deeplake_index_type::type::none;
};

} // namespace deeplake_core
