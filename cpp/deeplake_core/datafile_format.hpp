#pragma once

#include "chunk_format_definition.hpp"
#include "datafile_reader.hpp"
#include "datafile_writer.hpp"

#include <codecs/compression.hpp>
#include <icm/const_json.hpp>
#include <icm/json.hpp>
#include <nd/type.hpp>

namespace deeplake_core {

class datafile;

class datafile_format
{
public:
    datafile_format() = default;

    static datafile_format chunk_v1(nd::type type,
                                    codecs::compression chunk_compression = codecs::compression::null,
                                    codecs::compression sample_compression = codecs::compression::null,
                                    bool is_sequence = false);

    static datafile_format chunk(nd::type type,
                                 codecs::compression chunk_compression = codecs::compression::null,
                                 codecs::compression sample_compression = codecs::compression::null,
                                 bool is_sequence = false);

    static datafile_format video(nd::type type, codecs::compression compression);

    static datafile_format from_json(const icm::const_json& json);

public:
    bool operator==(const datafile_format& other) const = default;

    explicit operator bool() const noexcept
    {
        return data_.index() != 0;
    }

    [[nodiscard]] icm::json to_json() const;

public:
    std::shared_ptr<datafile_reader> create_reader(const deeplake_core::datafile& file,
                                                   const std::shared_ptr<storage::reader>& reader) const;

    std::shared_ptr<datafile_writer> create_writer(std::shared_ptr<storage::writer> writer,
                                                   std::string path_prefix = std::string()) const
    {
        ASSERT(data_.index() != 0);
        if (data_.index() == 1) {
            return std::get<1>(data_).create_writer(std::move(writer), std::move(path_prefix));
        } else if (data_.index() == 2) {
            return std::get<2>(data_).create_writer(std::move(writer), std::move(path_prefix));
        } else if (data_.index() == 3) {
            return std::get<3>(data_).create_writer(std::move(writer), std::move(path_prefix));
        }
        return std::get<4>(data_).create_writer(std::move(writer), std::move(path_prefix));
    }

    [[nodiscard]] const nd::type& type() const
    {
        ASSERT(data_.index() != 0);
        if (data_.index() == 1) {
            return std::get<1>(data_).type();
        } else if (data_.index() == 2) {
            return std::get<2>(data_).type();
        } else if (data_.index() == 3) {
            return std::get<3>(data_).type();
        }
        return std::get<4>(data_).type();
    }

    float compression_ratio() const noexcept
    {
        ASSERT(data_.index() != 0);
        if (data_.index() == 1) {
            return std::get<1>(data_).compression_ratio();
        } else if (data_.index() == 2) {
            return std::get<2>(data_).compression_ratio();
        } else if (data_.index() == 3) {
            return std::get<3>(data_).compression_ratio();
        }
        return std::get<4>(data_).compression_ratio();
    }

private:
    explicit datafile_format(chunk_datafile_format format)
        : data_(std::move(format))
    {
    }

    explicit datafile_format(chunk_datafile_format_v2 format)
        : data_(std::move(format))
    {
    }

    explicit datafile_format(video_datafile_format format)
        : data_(std::move(format))
    {
    }

    explicit datafile_format(chunk_datafile_sequence_format format)
        : data_(std::move(format))
    {
    }

    using data_type = std::variant<std::monostate,
                                   chunk_datafile_format,
                                   chunk_datafile_format_v2,
                                   video_datafile_format,
                                   chunk_datafile_sequence_format>;
    data_type data_;
};

} // namespace deeplake_core
