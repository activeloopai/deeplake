#pragma once

#include <deeplake_core/embedding_type.hpp>
#include <deeplake_core/format_definition.hpp>
#include <deeplake_core/schema.hpp>
#include <deeplake_core/basic_index_type.hpp>
#include <deeplake_core/type.hpp>
#include <nd/schema.hpp>
#include <nd/type.hpp>

#include <icm/string_map.hpp>

#include <optional>
#include <variant>

namespace deeplake_api {

using type_kind = deeplake_core::type_kind;
using type = deeplake_core::type;
using nd_or_string = std::variant<nd::type, std::string>;
using nd_or_string_or_type = std::variant<nd::type, std::string, type>;
using nd_or_type = std::variant<nd::type, type>;

namespace types {

nd::type int8();
nd::type int16();
nd::type int32();
nd::type int64();
nd::type uint8();
nd::type uint16();
nd::type uint32();
nd::type uint64();
nd::type float16();
nd::type float32();
nd::type float64();

type int8(std::optional<deeplake_core::numeric_index_type> index);
type int16(std::optional<deeplake_core::numeric_index_type> index);
type int32(std::optional<deeplake_core::numeric_index_type> index);
type int64(std::optional<deeplake_core::numeric_index_type> index);
type uint8(std::optional<deeplake_core::numeric_index_type> index);
type uint16(std::optional<deeplake_core::numeric_index_type> index);
type uint32(std::optional<deeplake_core::numeric_index_type> index);
type uint64(std::optional<deeplake_core::numeric_index_type> index);
type float16(std::optional<deeplake_core::numeric_index_type> index);
type float32(std::optional<deeplake_core::numeric_index_type> index);
type float64(std::optional<deeplake_core::numeric_index_type> index);
nd::type boolean();
nd::type bytes();
nd::type array(const nd_or_string& dtype, uint8_t dimensions);
nd::type array(const nd_or_string& dtype, const std::vector<int>& shape);
nd::type array(const nd_or_string& dtype);
type struct_(const icm::string_map<nd_or_string_or_type>& schema);

type generic(const nd_or_string& type);
type dict();
type dict(deeplake_core::json_index_type index_type);
type dict(const std::string& index_type);

type embedding(std::optional<int> dimensions = std::nullopt,
               const nd_or_string& dtype = "float32",
               std::optional<deeplake_core::embedding_index_type> index_type = std::nullopt);
type sequence(const nd_or_string_or_type type);
type image(const nd_or_string& type = "uint8", std::string sample_compression = "png");
type text(const std::optional<std::string>& chunk_compression = std::nullopt);
type text(const std::optional<std::string>& chunk_compression, deeplake_core::text_index_type index_type);
type text(const std::optional<std::string>& chunk_compression, const std::string& index_type);
type bbox(const nd_or_string& type = "float32");
type bbox(const nd_or_string& type,
          const std::optional<std::string>& format,
          const std::optional<std::string>& bbox_type);

type bmask(const std::optional<std::string>& sample_compression = std::nullopt,
           const std::optional<std::string>& chunk_compression = std::nullopt);
type smask(const nd_or_string& type = "uint8",
           const std::optional<std::string>& sample_compression = std::nullopt,
           const std::optional<std::string>& chunk_compression = std::nullopt);

type polygon();
type point(uint32_t dimensions);
type class_label(const nd_or_string& type);
type medical(const std::string& compression);
type audio(const nd_or_string& dtype = "uint8", std::string sample_compression = "mp3");
type mesh();
type link(const nd_or_type& tp);
type video(const std::string& compression = "mp4");

type to_type(const nd_or_string_or_type& tp);

} // namespace types

} // namespace deeplake_api
