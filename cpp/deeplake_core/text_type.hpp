#pragma once

#include "basic_index_type.hpp"
#include "deeplake_index_type.hpp"
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

class text_type
{
public:
    explicit text_type(codecs::compression chunk_compression = codecs::compression::lz4,
                       text_index_type index_type = deeplake_index_type::type::none)
        : chunk_compression_(chunk_compression)
        , indexing_(index_type)

    {
    }

    static text_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    nd::type data_type() const;

    base::htype htype() const noexcept
    {
        return base::htype::text;
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

    text_index_type index_type() const
    {
        return indexing_;
    }

    bool operator==(const text_type&) const = default;

private:
    codecs::compression chunk_compression_;
    text_index_type indexing_;
};

} // namespace deeplake_core
