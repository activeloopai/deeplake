#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <memory>

namespace deeplake_core {

class type;
class convert_context;

class link_type
{
public:
    explicit link_type(std::shared_ptr<type> type);

    static link_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const;

    base::htype htype() const noexcept;

    inline bool is_link() const noexcept
    {
        return true;
    }

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context* ctx) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context* ctx) const;

    std::string to_string() const;

    bool operator==(const link_type& other) const noexcept = default;

    bool is_image() const;

    bool is_video() const;

    bool is_segment_mask() const;

    bool is_audio() const;

    bool is_mesh() const;

    auto get_type() const
    {
        return type_;
    }

private:
    std::shared_ptr<type> type_;
};

} // namespace deeplake_core
