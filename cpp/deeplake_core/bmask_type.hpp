#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class bmask_type
{
public:
    bmask_type(codecs::compression sample_compression, codecs::compression chunk_compression)
        : type_(nd::type::array(nd::scalar_type(nd::dtype::boolean), static_cast<uint8_t>(3)))
        , sample_compression_(sample_compression)
        , chunk_compression_(chunk_compression)
    {
    }

    static bmask_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::binary_mask;
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

    bool operator==(const bmask_type& other) const
    {
        return type_ == other.type_ && sample_compression_ == other.sample_compression_ &&
               chunk_compression_ == other.chunk_compression_;
    }

    bool operator!=(const bmask_type& other) const
    {
        return !(*this == other);
    }

private:
    nd::type type_;
    codecs::compression sample_compression_;
    codecs::compression chunk_compression_;
};

} // namespace deeplake_core
