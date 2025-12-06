#pragma once

#include "datafile_reader.hpp"
#include "datafile_writer.hpp"
#include "impl/calculate_compression_ratio.hpp"

#include <codecs/compression.hpp>
#include <icm/const_json.hpp>
#include <icm/json.hpp>
#include <nd/type.hpp>
#include <storage/reader.hpp>
#include <storage/writer.hpp>

namespace deeplake_core {

class datafile;
class datafile_format;

class chunk_format_definition
{
public:
    chunk_format_definition(codecs::compression chunk_compression,
                            codecs::compression sample_compression,
                            bool is_sequence)
        : chunk_compression_(chunk_compression)
        , sample_compression_(sample_compression)
        , is_sequence_(is_sequence)
    {
    }

    static chunk_format_definition from_json(const icm::const_json& json);

    icm::json to_json() const;

    std::string to_string() const;

    bool operator==(const chunk_format_definition& other) const = default;

    codecs::compression chunk_compression() const noexcept
    {
        return chunk_compression_;
    }

    codecs::compression sample_compression() const noexcept
    {
        return sample_compression_;
    }

    bool is_sequence() const noexcept
    {
        return is_sequence_;
    }

    datafile_format get_datafile_format_for_type(const nd::type& type) const;

private:
    codecs::compression chunk_compression_;
    codecs::compression sample_compression_;
    bool is_sequence_ = false;
};

class chunk_datafile_format
{
public:
    chunk_datafile_format(nd::type type,
                          codecs::compression chunk_compression,
                          codecs::compression sample_compression,
                          bool is_sequence)
        : type_(std::move(type))
        , chunk_compression_(chunk_compression)
        , sample_compression_(sample_compression)
        , is_sequence_(is_sequence)
    {
    }

    static chunk_datafile_format from_json(const icm::const_json& json);

    icm::json to_json() const;

public:
    std::shared_ptr<datafile_reader> create_reader(const datafile& file,
                                                   const std::shared_ptr<storage::reader>& reader) const;

    std::shared_ptr<datafile_writer> create_writer(const std::shared_ptr<storage::writer>& writer,
                                                   std::string path_prefix) const;

    codecs::compression chunk_compression() const noexcept
    {
        return chunk_compression_;
    }

    codecs::compression sample_compression() const noexcept
    {
        return sample_compression_;
    }

    [[nodiscard]] const nd::type& type() const noexcept
    {
        return type_;
    }

    float compression_ratio() const noexcept
    {
        return impl::calculate_compression_ratio(type_.get_dtype(), sample_compression_, chunk_compression_);
    }

    bool operator==(const chunk_datafile_format& other) const = default;

private:
    nd::type type_;
    codecs::compression chunk_compression_;
    codecs::compression sample_compression_;
    bool is_sequence_ = false;
};

class chunk_datafile_sequence_format
{
public:
    chunk_datafile_sequence_format(nd::type type,
                                   codecs::compression chunk_compression,
                                   codecs::compression sample_compression,
                                   bool is_sequence)
        : type_(std::move(type))
        , chunk_compression_(chunk_compression)
        , sample_compression_(sample_compression)
        , is_sequence_(is_sequence)
    {
    }

    static chunk_datafile_sequence_format from_json(const icm::const_json& json);

    icm::json to_json() const;

    std::shared_ptr<datafile_reader> create_reader(const datafile& file,
                                                   const std::shared_ptr<storage::reader>& reader) const;

    std::shared_ptr<datafile_writer> create_writer(const std::shared_ptr<storage::writer>& writer,
                                                   std::string path_prefix) const;

    codecs::compression chunk_compression() const noexcept
    {
        return chunk_compression_;
    }

    codecs::compression sample_compression() const noexcept
    {
        return sample_compression_;
    }

    [[nodiscard]] const nd::type& type() const noexcept
    {
        return type_;
    }

    float compression_ratio() const noexcept
    {
        return impl::calculate_compression_ratio(type_.get_dtype(), sample_compression_, chunk_compression_);
    }

    bool operator==(const chunk_datafile_sequence_format& other) const = default;

private:
    nd::type type_;
    codecs::compression chunk_compression_;
    codecs::compression sample_compression_;
    bool is_sequence_ = false;
};

class chunk_datafile_format_v2
{
public:
    explicit chunk_datafile_format_v2(nd::type type)
        : type_(std::move(type))
    {
    }

    static chunk_datafile_format_v2 from_json(const icm::const_json& json);

    icm::json to_json() const;

    std::shared_ptr<datafile_reader> create_reader(const datafile& file,
                                                   const std::shared_ptr<storage::reader>& reader) const;

    std::shared_ptr<datafile_writer> create_writer(const std::shared_ptr<storage::writer>& writer,
                                                   std::string path_prefix) const;

    [[nodiscard]] const nd::type& type() const noexcept
    {
        return type_;
    }

    float compression_ratio() const noexcept
    {
        return 1.0f;
    }

    bool operator==(const chunk_datafile_format_v2& other) const = default;

private:
    nd::type type_;
};

class video_format_definition
{
public:
    video_format_definition(nd::type type, codecs::compression compression)
        : compression_(compression)
        , type_(std::move(type))
    {
    }

    static video_format_definition from_json(const icm::const_json& json)
    {
        return video_format_definition(nd::type::from_json(json.at("type")),
                                       codecs::compression_from_json(json.at("compression")));
    }

public:
    icm::json to_json() const
    {
        return icm::json{{"type", type_.to_json()}, {"compression", codecs::compression_to_json(compression_)}};
    }

    std::string to_string() const;

    bool operator==(const video_format_definition& other) const = default;

    codecs::compression compression() const noexcept
    {
        return compression_;
    }

public:
    datafile_format get_datafile_format_for_type(const nd::type& type) const;

private:
    codecs::compression compression_;
    nd::type type_;
};

class video_datafile_format
{
public:
    explicit video_datafile_format(nd::type type, codecs::compression compression)
        : type_(std::move(type))
    {
    }

    static video_datafile_format from_json(const icm::const_json& json)
    {
        auto t = nd::type::from_json(json.at("type"));
        auto c = codecs::compression_from_json(json.at("compression"));
        return video_datafile_format(t, c);
    }

    icm::json to_json() const
    {
        return icm::json{{"type", type_.to_json()}, {"compression", codecs::compression_to_json(compression_)}};
    }

    std::shared_ptr<datafile_reader> create_reader(const datafile& file,
                                                   const std::shared_ptr<storage::reader>& reader) const;

    std::shared_ptr<datafile_writer> create_writer(const std::shared_ptr<storage::writer>& writer,
                                                   std::string prefix) const;

    [[nodiscard]] const nd::type& type() const noexcept
    {
        return type_;
    }

    [[nodiscard]] codecs::compression compression() const noexcept
    {
        return compression_;
    }

    float compression_ratio() const noexcept
    {
        return 20.f; // this is a guess based on some articles and chatgpt :D
    }

    bool operator==(const video_datafile_format& other) const = default;

private:
    nd::type type_;
    codecs::compression compression_ = codecs::compression::null;
};

} // namespace deeplake_core
