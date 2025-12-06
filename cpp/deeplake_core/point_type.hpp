#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class point_type
{
public:
    point_type() = default;

    point_type(nd::dtype dtype, uint32_t dimensions)
        : dtype_(dtype)
        , dimensions_(dimensions)
    {
    }

    static point_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return nd::type::array(dtype_, dimensions_);
    }

    base::htype htype() const noexcept
    {
        return base::htype::point;
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

    bool operator==(const point_type& other) const
    {
        return true;
    }

private:
    nd::dtype dtype_ = nd::dtype::int32;
    uint32_t dimensions_ = 2; // Default to 2D points
};

} // namespace deeplake_core
