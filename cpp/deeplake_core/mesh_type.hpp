#pragma once

#include "format_definition.hpp"

#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class mesh_type
{
public:
    explicit mesh_type()
    {
    }

    static mesh_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    bool is_sequence() const
    {
        return false;
    }

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::mesh;
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

    bool operator==(const mesh_type& other) const = default;

private:
    nd::type type_ = nd::type::scalar(nd::scalar_type(nd::dtype::object));
};

} // namespace deeplake_core
