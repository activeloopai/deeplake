#include "create_tensor_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> create_tensor_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8(), true),
                                   arrow::field("name", arrow::utf8(), true),
                                   arrow::field("dtype", arrow::utf8(), true),
                                   arrow::field("htype", arrow::utf8(), true),
                                   arrow::field("length", arrow::uint64(), true),
                                   arrow::field("is_link", arrow::boolean(), true),
                                   arrow::field("is_sequence", arrow::boolean(), true),
                                   arrow::field("hidden", arrow::boolean(), true),
                                   arrow::field("chunkCompression", arrow::utf8(), true),
                                   arrow::field("sampleCompression", arrow::utf8(), true),
//                                   arrow::field("links", arrow::map(arrow::utf8(), arrow::map(arrow::utf8(), arrow::utf8())), true),
                                   arrow::field("maxChunkSize", arrow::uint64(), true),
                                   arrow::field("minShape", arrow::list(arrow::uint64()), true),
                                   arrow::field("maxShape", arrow::list(arrow::uint64()), true),
                                   arrow::field("tilingThreshold", arrow::uint64(), true),
                                   arrow::field("typestr", arrow::utf8(), true),
                                   arrow::field("verify", arrow::boolean(), true),
                                   arrow::field("version", arrow::utf8(), true),
                           }));

    create_tensor_action::create_tensor_action(std::string id,
                                               std::string name,
                                               std::optional<std::string> dtype,
                                               std::string htype,
                                               const long &length,
                                               const bool &is_link,
                                               const bool &is_sequence,
                                               const bool &hidden,
                                               const std::optional<std::string> &chunk_compression,
                                               const std::optional<std::string> &sample_compression,
                                               const std::map<std::string, std::map<std::string, std::variant<std::string, bool>>> &links,
                                               const std::optional<long> &max_chunk_size,
                                               const std::vector<unsigned long> &min_shape,
                                               const std::vector<unsigned long> &max_shape,
                                               const std::optional<long> &tiling_threshold,
                                               const std::optional<std::string> &typestr,
                                               const bool &verify,
                                               std::string version)
            : id(std::move(id)), name(std::move(name)),
              dtype(std::move(dtype)), htype(std::move(htype)),
              length(length), is_link(is_link), is_sequence(is_sequence), hidden(hidden),
              chunk_compression(chunk_compression), sample_compression(sample_compression),
              links(links),
              max_chunk_size(max_chunk_size),
              min_shape(min_shape),
              max_shape(max_shape),
              tiling_threshold(tiling_threshold), typestr(typestr),
              verify(verify), version(std::move(version)) {}

    create_tensor_action::create_tensor_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = from_struct<std::string>("id", value).value();
        name = from_struct<std::string>("name", value).value();
        dtype = from_struct<std::string>("dtype", value);
        htype = from_struct<std::string>("htype", value).value();
        length = from_struct<long>("length", value).value();
        is_link = from_struct<bool>("is_link", value).value();
        is_sequence = from_struct<bool>("is_sequence", value).value();
        hidden = from_struct<bool>("hidden", value).value();
        chunk_compression = from_struct<std::string>("chunkCompression", value);
        sample_compression = from_struct<std::string>("sampleCompression", value);
        max_chunk_size = from_struct<long>("maxChunkSize", value);
        min_shape = from_arraystruct<unsigned long>("minShape", value);
        max_shape = from_arraystruct<unsigned long>("maxShape", value);
        tiling_threshold = from_struct<long>("tilingThreshold", value);
        typestr = from_struct<std::string>("typestr", value);
        verify = from_struct<bool>("verify", value).value();
        version = from_struct<std::string>("version", value).value();
    }

    std::string create_tensor_action::action_name() {
        return "tensor";
    }

    std::shared_ptr<arrow::StructType> create_tensor_action::action_type() {
        return arrow_type;
    }

    nlohmann::json create_tensor_action::to_json() {
        nlohmann::json json;

        json["id"] = id;
        json["name"] = name;
        json["dtype"] = to_json_value<std::string>(dtype);
        json["htype"] = htype;
        json["length"] = length;
        json["is_link"] = is_link;
        json["is_sequence"] = is_sequence;
        json["hidden"] = hidden;
        json["chunkCompression"] = to_json_value<std::string>(chunk_compression);
        json["sampleCompression"] = to_json_value<std::string>(sample_compression);
//        json["links"] = links;
        json["maxChunkSize"] = to_json_value<long>(max_chunk_size);
        json["minShape"] = min_shape;
        json["maxShape"] = max_shape;
        json["tilingThreshold"] = to_json_value<long>(tiling_threshold);
        json["typestr"] = to_json_value<std::string>(typestr);
        json["verify"] = verify;
        json["version"] = version;

        return json;
    }
}
