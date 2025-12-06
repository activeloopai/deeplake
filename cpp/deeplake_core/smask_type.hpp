#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class smask_type
{
public:
    smask_type(nd::type type, codecs::compression sample_compression, codecs::compression chunk_compression)
        : type_(nd::type::array(type.get_scalar_type(), 3))
        , sample_compression_(sample_compression)
        , chunk_compression_(chunk_compression)
    {
    }

    static smask_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::segment_mask;
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

    bool operator==(const smask_type& other) const
    {
        return type_ == other.type_ && sample_compression_ == other.sample_compression_ &&
               chunk_compression_ == other.chunk_compression_;
    }

private:
    nd::type type_;
    codecs::compression sample_compression_;
    codecs::compression chunk_compression_;
};

} // namespace deeplake_core
