#pragma once

#include "format_definition.hpp"
#include "exceptions.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class convert_context;

class audio_type
{
public:
    audio_type(const nd::type& type, codecs::compression sample_compression)
        : type_(nd::type::array(type.get_scalar_type(), 2))
        , compression_(sample_compression)
    {
        if (compression_ != codecs::compression::null && compression_ != codecs::compression::wav &&
            compression_ != codecs::compression::mp3) {
            throw invalid_audio_compression(codecs::compression_to_str(compression_));
        }
    }

    static audio_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const
    {
        return type_;
    }

    base::htype htype() const noexcept
    {
        return base::htype::audio;
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

    bool operator==(const audio_type& other) const = default;

private:
    nd::type type_;
    codecs::compression compression_;
};

} // namespace deeplake_core
