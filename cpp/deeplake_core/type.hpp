#pragma once

#include "audio_type.hpp"
#include "bbox_type.hpp"
#include "bmask_type.hpp"
#include "class_label_type.hpp"
#include "dict_type.hpp"
#include "embedding_type.hpp"
#include "format_definition.hpp"
#include "generic_type.hpp"
#include "image_type.hpp"
#include "link_type.hpp"
#include "medical_type.hpp"
#include "mesh_type.hpp"
#include "point_type.hpp"
#include "polygon_type.hpp"
#include "sequence_type.hpp"
#include "smask_type.hpp"
#include "struct_type.hpp"
#include "text_type.hpp"
#include "video_type.hpp"

#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

namespace async {
template <typename T>
class promise;
} // namespace async

namespace icm {
class const_json;
class json;
} // namespace icm

namespace deeplake_core {

class convert_context;

enum class type_kind
{
    generic,
    text,
    dict,
    embedding,
    sequence,
    image,
    bbox,
    bmask,
    smask,
    link,
    polygon,
    class_label,
    point,
    video,
    medical,
    audio,
    mesh,
    struct_,
};

std::string_view type_kind_to_string(type_kind kind);

type_kind type_kind_from_string(const std::string& str);

class type
{
public:
    type() = default;

    static type generic(nd::type type);

    static type generic(nd::type type, std::optional<numeric_index_type> index);

    static type
    generic(nd::type type, format_definition format, std::optional<numeric_index_type> index = std::nullopt);

    static type image(nd::type type, codecs::compression sample_compression);

    static type text(codecs::compression chunk_compression);

    static type text(codecs::compression chunk_compression, text_index_type index_type);

    static type dict();

    static type dict(deeplake_core::json_index_type index_type);

    static type
    embedding(int32_t size, nd::dtype dtype, embedding_index_type index_type = embedding_index_type::type::clustered);

    static type sequence(type&& type);

    static type sequence(const type& type);

    static type bbox(nd::type type, std::optional<bbox_format> format);

    static type bmask(codecs::compression sample_compression, codecs::compression chunk_compression);

    static type smask(nd::type type, codecs::compression sample_compression, codecs::compression chunk_compression);

    static type polygon();

    static type class_label(nd::type type);

    static type link(const type& tp);

    static type point(nd::dtype dtype = nd::dtype::float32, uint32_t dimensions = 2);

    static type video(codecs::compression compression);

    static type medical(codecs::compression compression);

    static type audio(nd::type type, codecs::compression sample_compression);

    static type mesh();

    static type struct_(deeplake_core::schema final_schema);

    static type from_json(const icm::const_json& json);

    static type from_basic_type_str(std::string_view str);

public:
    icm::json to_json() const;

    std::string id() const;

    type_kind kind() const
    {
        return static_cast<type_kind>(data_.index());
    }

    auto visit(auto&& visitor) const
    {
        return std::visit(std::forward<decltype(visitor)>(visitor), data_);
    }

    const generic_type& as_generic() const
    {
        return std::get<generic_type>(data_);
    }

    const image_type& as_image() const
    {
        return std::get<image_type>(data_);
    }

    const text_type& as_text() const
    {
        return std::get<text_type>(data_);
    }

    const dict_type& as_dict() const
    {
        return std::get<dict_type>(data_);
    }

    const embedding_type& as_embedding() const
    {
        return std::get<embedding_type>(data_);
    }

    embedding_type& as_embedding()
    {
        return std::get<embedding_type>(data_);
    }

    const sequence_type& as_sequence() const
    {
        return std::get<sequence_type>(data_);
    }

    const bbox_type& as_bbox() const
    {
        return std::get<bbox_type>(data_);
    }

    const bmask_type& as_bmask() const
    {
        return std::get<bmask_type>(data_);
    }

    const smask_type& as_smask() const
    {
        return std::get<smask_type>(data_);
    }

    const polygon_type& as_polygon() const
    {
        return std::get<polygon_type>(data_);
    }

    const class_label_type& as_label() const
    {
        return std::get<class_label_type>(data_);
    }

    const link_type& as_link() const
    {
        return std::get<link_type>(data_);
    }

    const point_type& as_point() const
    {
        return std::get<point_type>(data_);
    }

    const video_type& as_video() const
    {
        return std::get<video_type>(data_);
    }

    const audio_type& as_audio() const
    {
        return std::get<audio_type>(data_);
    }

    const mesh_type& as_mesh() const
    {
        return std::get<mesh_type>(data_);
    }

    bool is_generic() const;

    bool is_text() const;

    bool is_dict() const;

    bool is_sequence() const;

    bool is_link() const;

    bool is_image() const;

    bool is_segment_mask() const;

    bool is_embedding() const;

    bool is_video() const;

    bool is_audio() const;

    bool is_mesh() const;

    bool is_struct() const;

    icm::shape shape() const;

    nd::type data_type() const;

    nd::type storage_type() const;

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    nd::array convert_batch_to_compact(nd::array array) const;

    base::htype htype() const noexcept;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context* ctx = nullptr) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context* ctx = nullptr) const;

    std::string to_string() const;

    bool operator==(const type& other) const
    {
        return data_ == other.data_;
    }

private:
    using data_t = std::variant<generic_type,
                                text_type,
                                dict_type,
                                embedding_type,
                                sequence_type,
                                image_type,
                                bbox_type,
                                bmask_type,
                                smask_type,
                                link_type,
                                polygon_type,
                                class_label_type,
                                point_type,
                                video_type,
                                medical_type,
                                audio_type,
                                mesh_type,
                                struct_type>;

    explicit type(data_t data)
        : data_(std::move(data))
    {
    }

private:
    data_t data_;
};

} // namespace deeplake_core
