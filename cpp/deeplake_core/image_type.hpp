#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class image_type
{
public:
    image_type(const nd::type& type, codecs::compression sample_compression)
        : type_(nd::type::array(type.get_scalar_type(), 3))
        , sample_compression_(sample_compression)
    {
    }

    static image_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::image;
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

    bool operator==(const image_type& other) const = default;

private:
    nd::type type_;
    codecs::compression sample_compression_;
};

} // namespace deeplake_core
