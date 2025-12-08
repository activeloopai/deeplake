#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class video_type
{
public:
    video_type(codecs::compression sample_compression)
        : type_(nd::type::array(nd::scalar_type(nd::dtype::byte), 4))
        , sample_compression_(sample_compression)
    {
    }

    static video_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    inline bool is_link() const {
        return false;
    }

    base::htype htype() const noexcept
    {
        return base::htype::video;
    }

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    bool operator==(const video_type& other) const
    {
        return sample_compression_ == other.sample_compression_;
    }

private:
    nd::type type_;
    codecs::compression sample_compression_;
};

} // namespace deeplake_core
