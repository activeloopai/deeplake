#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

enum class coord_type
{
    CCWH,
    LTWH,
    LTRB,
    unknown,
};

enum class pixel_type
{
    pixel,
    fractional,
};

std::string coord_type_to_string(coord_type coord);
coord_type coord_type_from_string(const std::string& str);
std::string pixel_type_to_string(pixel_type p_type);
pixel_type pixel_type_from_string(const std::string& str);

struct bbox_format
{
public:
    bbox_format(std::optional<coord_type> c_system, std::optional<pixel_type> p_type)
        : coord_type_(c_system)
        , pixel_type_(p_type)
    {
    }

    icm::json to_json() const noexcept;
    static bbox_format from_json(const icm::const_json& j);
    static bbox_format create(const std::optional<std::string>& format, const std::optional<std::string>& type);

    bool operator==(const bbox_format& other) const noexcept = default;

public:
    std::optional<coord_type> coord_type_;
    std::optional<pixel_type> pixel_type_;
};

class bbox_type
{
public:
    bbox_type(nd::type type, std::optional<bbox_format> format)
        : type_(nd::type::array(type.get_scalar_type(), 3))
        , bbox_format_(format)
    {
    }

    static bbox_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::bbox;
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

    bool operator==(const bbox_type& other) const
    {
        return type_ == other.type_ && bbox_format_ == other.bbox_format_;
    }

    const std::optional<bbox_format>& get_bbox_format() const noexcept
    {
        return bbox_format_;
    }

private:
    nd::type type_;
    std::optional<bbox_format> bbox_format_;
};

} // namespace deeplake_core
