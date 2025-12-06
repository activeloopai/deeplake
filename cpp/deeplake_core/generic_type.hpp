#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <deeplake_core/index_type.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class generic_type
{
public:
    generic_type()
        : type_(nd::type::unknown())
    {
    }

    explicit generic_type(nd::type type, std::optional<numeric_index_type> index_type = std::nullopt)
        : type_(std::move(type))
        , format_(format_definition::default_format_for_type(type_))
        , index_type_(index_type)
    {
    }

    generic_type(nd::type type, format_definition format, std::optional<numeric_index_type> index_type = std::nullopt)
        : type_(std::move(type))
        , format_(std::move(format))
        , index_type_(index_type)
    {
    }

    static generic_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    inline bool is_link() const noexcept
    {
        return false;
    }

    base::htype htype() const noexcept
    {
        return base::htype::generic;
    }

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    bool operator==(const generic_type& other) const
    {
        return type_ == other.type_ && format_ == other.format_ && index_type_ == other.index_type_;
    }

    std::optional<numeric_index_type> index_type() const
    {
        return index_type_;
    }

private:
    nd::type type_;
    format_definition format_;
    std::optional<numeric_index_type> index_type_ = std::nullopt;
};

} // namespace deeplake_core
