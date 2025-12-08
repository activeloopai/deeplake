#pragma once

#include "chunk_format_definition.hpp"
#include "chunk_strategy.hpp"
#include "datafile_format.hpp"

#include <codecs/compression.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace deeplake_core {

class format_definition
{
public:
    format_definition() = default;

    static format_definition chunk(codecs::compression sample_compression,
                                   codecs::compression chunk_compression,
                                   bool is_sequence,
                                   chunk_strategy strategy = chunk_strategy::default_strategy());

    static format_definition video(nd::type type, codecs::compression sample_compression);

    static format_definition default_format_for_type(nd::type type);

    static format_definition from_json(const icm::const_json& json);

public:
    icm::json to_json() const;

    std::string to_string() const
    {
        if (data_.index() == 2) {
            return std::get<2>(data_).to_string();
        }
        return std::get<1>(data_).to_string();
    }

    const chunk_strategy& get_strategy() const
    {
        return strategy_;
    }

    bool operator==(const format_definition& other) const = default;

    explicit operator chunk_format_definition() const
    {
        ASSERT(data_.index() == 1);
        return std::get<1>(data_);
    }

    explicit operator video_format_definition() const
    {
        ASSERT(data_.index() == 2);
        return std::get<2>(data_);
    }

public:
    datafile_format get_datafile_format_for_type(const nd::type& type) const
    {
        if (data_.index() == 2) {
            return std::get<2>(data_).get_datafile_format_for_type(type);
        }
        return std::get<1>(data_).get_datafile_format_for_type(type);
    }

private:
    format_definition(chunk_format_definition format, chunk_strategy strategy)
        : data_(std::move(format))
        , strategy_(std::move(strategy))
    {
    }

    format_definition(video_format_definition format)
        : data_(std::move(format))
        , strategy_(chunk_strategy::num_rows(1))
    {
    }

    using data_type = std::variant<std::monostate, chunk_format_definition, video_format_definition>;

    data_type data_;
    chunk_strategy strategy_;
};

} // namespace deeplake_core
