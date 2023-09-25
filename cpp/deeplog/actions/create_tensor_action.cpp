#include "create_tensor_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> create_tensor_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8()),
                                   arrow::field("name", arrow::utf8()),
                                   arrow::field("dtype", arrow::utf8()),
                                   arrow::field("htype", arrow::utf8()),
                                   arrow::field("length", arrow::uint64()),
                                   arrow::field("link", arrow::boolean()),
                                   arrow::field("sequence", arrow::boolean()),
                                   arrow::field("hidden", arrow::boolean()),
                                   arrow::field("chunkCompression", arrow::utf8()),
                                   arrow::field("sampleCompression", arrow::utf8()),
                                   arrow::field("links", arrow::map(arrow::utf8(), arrow::map(arrow::utf8(), arrow::utf8()))),
                                   arrow::field("maxChunkSize", arrow::uint64()),
                                   arrow::field("minShape", arrow::list(arrow::uint64())),
                                   arrow::field("maxShape", arrow::list(arrow::uint64())),
                                   arrow::field("tilingThreshold", arrow::uint64()),
                                   arrow::field("typestr", arrow::utf8()),
                                   arrow::field("verify", arrow::boolean()),
                                   arrow::field("version", arrow::utf8()),
                           }));

    create_tensor_action::create_tensor_action(std::string id,
                                               std::string name,
                                               std::string dtype,
                                               std::string htype,
                                               const long &length,
                                               const bool &is_link,
                                               const bool &is_sequence,
                                               const bool &hidden,
                                               const std::optional<std::string> &chunk_compression,
                                               const std::optional<std::string> &sample_compression,
                                               const std::map<std::string, std::map<std::string, std::string>> &links,
                                               const std::optional<long> &max_chunk_size,
                                               const std::vector<long> &min_shape,
                                               const std::vector<long> &max_shape,
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
        id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        name = reinterpret_pointer_cast<arrow::StringScalar>(value->field("name").ValueOrDie())->view();
        dtype = reinterpret_pointer_cast<arrow::StringScalar>(value->field("dtype").ValueOrDie())->view();
        htype = reinterpret_pointer_cast<arrow::StringScalar>(value->field("htype").ValueOrDie())->view();
        length = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("length").ValueOrDie())->value;
        is_link = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("link").ValueOrDie())->value;
        is_sequence = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("sequence").ValueOrDie())->value;
        hidden = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("hidden").ValueOrDie())->value;
        chunk_compression = reinterpret_pointer_cast<arrow::StringScalar>(value->field("chunkCompression").ValueOrDie())->view();
        sample_compression = reinterpret_pointer_cast<arrow::StringScalar>(value->field("sampleCompression").ValueOrDie())->view();
        max_chunk_size = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("maxChunkSize").ValueOrDie())->value;
//        min_shape = reinterpret_pointer_cast<arrow::ListScalar>(value->field("minShape").ValueOrDie())->value;
//        max_shape = reinterpret_pointer_cast<arrow::ListScalar>(value->field("maxShape").ValueOrDie())->value;
        tiling_threshold = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("tilingThreshold").ValueOrDie())->value;
        typestr = reinterpret_pointer_cast<arrow::StringScalar>(value->field("typestr").ValueOrDie())->view();
        verify = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("verify").ValueOrDie())->value;
        version = reinterpret_pointer_cast<arrow::StringScalar>(value->field("version").ValueOrDie())->view();
    }

    std::string create_tensor_action::action_name() {
        return "tensor";
    }

    nlohmann::json create_tensor_action::to_json() {
        nlohmann::json json;

        json["id"] = id;
        json["name"] = name;
        if (dtype.has_value()) {
            json["dtype"] = dtype.value();
        } else {
            json["dtype"] = nlohmann::json::value_t::null;
        }
        json["htype"] = htype;
        json["length"] = length;
        json["link"] = is_link;
        json["sequence"] = is_sequence;
        json["hidden"] = hidden;
        if (chunk_compression.has_value()) {
            json["chunkCompression"] = chunk_compression.value();
        } else {
            json["chunkCompression"] = nlohmann::json::value_t::null;
        }
        if (sample_compression.has_value()) {
            json["sampleCompression"] = sample_compression.value();
        } else {
            json["sampleCompression"] = nlohmann::json::value_t::null;
        }
        json["links"] = links;
        if (max_chunk_size.has_value()) {
            json["maxChunkSize"] = max_chunk_size.value();
        } else {
            json["maxChunkSize"] = nlohmann::json::value_t::null;
        }
        json["minShape"] = min_shape;
        json["maxShape"] = max_shape;
        if (tiling_threshold.has_value()) {
            json["tilingThreshold"] = tiling_threshold.value();
        } else {
            json["tilingThreshold"] = nlohmann::json::value_t::null;
        }
        if (typestr.has_value()) {
            json["typestr"] = typestr.value();
        } else {
            json["typestr"] = nlohmann::json::value_t::null;
        }
        json["verify"] = verify;
        json["version"] = version;

        return json;
    }
}
