#include "create_tensor_action.hpp"

namespace deeplog {
    create_tensor_action::create_tensor_action(std::string id, std::string name,
                                               std::optional<std::string> dtype, std::string htype,
                                               long length, bool is_link, bool is_sequence, bool hidden,
                                               std::optional<std::string> chunk_compression, std::optional<std::string> sample_compression,
                                               std::map<std::string, std::map<std::string, std::string>> links,
                                               std::optional<long> max_chunk_size,
                                               std::vector<long> min_shape,
                                               std::vector<long> max_shape,
                                               std::optional<long> tiling_threshold,
                                               std::optional<std::string> typestr,
                                               bool verify, std::string version)
            : id(id), name(name),
              dtype(dtype), htype(htype),
              length(length), is_link(is_link), is_sequence(is_sequence), hidden(hidden),
              chunk_compression(chunk_compression), sample_compression(sample_compression),
              links(links),
              max_chunk_size(max_chunk_size),
              min_shape(min_shape),
              max_shape(max_shape),
              tiling_threshold(tiling_threshold), typestr(typestr),
              verify(verify), version(version) {}

    create_tensor_action::create_tensor_action(const nlohmann::json &j) {
        const auto &base = j.at("tensor");
        base.at("id").get_to(id);
        base.at("name").get_to(name);
        if (!base.at("dtype").is_null()) {
            dtype = base.at("dtype").get<std::string>();
        }
        base.at("htype").get_to(htype);
        base.at("link").get_to(is_link);
        base.at("sequence").get_to(is_sequence);
        base.at("hidden").get_to(hidden);
        if (!base.at("chunkCompression").is_null()) {
            chunk_compression = base.at("chunkCompression").get<std::string>();
        }
        if (!base.at("sampleCompression").is_null()) {
            sample_compression = base.at("sampleCompression").get<std::string>();
        }
        base.at("links").get_to(links);
        if (!base.at("maxChunkSize").is_null()) {
            max_chunk_size = base.at("maxChunkSize").get<long>();
        }
        base.at("minShape").get_to(min_shape);
        base.at("maxShape").get_to(max_shape);
        if (!base.at("maxChunkSize").is_null()) {
            tiling_threshold = base.at("tilingThreshold").get<long>();
        }
        if (!base.at("typestr").is_null()) {
            typestr = base.at("typestr").get<std::string>();
        }
        base.at("version").get_to(version);
    }

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

    void create_tensor_action::to_json(nlohmann::json &j) {
        j["tensor"]["id"] = id;
        j["tensor"]["name"] = name;
        if (dtype.has_value()) {
            j["tensor"]["dtype"] = dtype.value();
        } else {
            j["tensor"]["dtype"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["htype"] = htype;
        j["tensor"]["length"] = length;
        j["tensor"]["link"] = is_link;
        j["tensor"]["sequence"] = is_sequence;
        j["tensor"]["hidden"] = hidden;
        if (chunk_compression.has_value()) {
            j["tensor"]["chunkCompression"] = chunk_compression.value();
        } else {
            j["tensor"]["chunkCompression"] = nlohmann::json::value_t::null;
        }
        if (sample_compression.has_value()) {
            j["tensor"]["sampleCompression"] = sample_compression.value();
        } else {
            j["tensor"]["sampleCompression"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["links"] = links;
        if (max_chunk_size.has_value()) {
            j["tensor"]["maxChunkSize"] = max_chunk_size.value();
        } else {
            j["tensor"]["maxChunkSize"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["minShape"] = min_shape;
        j["tensor"]["maxShape"] = max_shape;
        if (tiling_threshold.has_value()) {
            j["tensor"]["tilingThreshold"] = tiling_threshold.value();
        } else {
            j["tensor"]["tilingThreshold"] = nlohmann::json::value_t::null;
        }
        if (typestr.has_value()) {
            j["tensor"]["typestr"] = typestr.value();
        } else {
            j["tensor"]["typestr"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["verify"] = verify;
        j["tensor"]["version"] = version;
    }

    std::shared_ptr<arrow::StructBuilder> create_tensor_action::arrow_array() {
        auto action_struct = arrow::struct_({
                                                    arrow::field("id", arrow::utf8()),
                                                    arrow::field("name", arrow::utf8()),
                                                    arrow::field("dtype", arrow::utf8()),
                                                    arrow::field("htype", arrow::utf8()),
                                                    arrow::field("length", arrow::uint64()),
                                                    arrow::field("is_link", arrow::boolean()),
                                                    arrow::field("is_sequence", arrow::boolean()),
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
                                            });
        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(action_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
//                std::make_shared<arrow::MapBuilder>(arrow::MapBuilder(arrow::utf8(), arrow::MapBuilarrow::map(arrow::utf8(), arrow::utf8()))),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::ListBuilder>(arrow::ListBuilder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()))),
                std::make_shared<arrow::ListBuilder>(arrow::ListBuilder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()))),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
        })));
    }

    arrow::Status create_tensor_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{name}));
        if (dtype.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::StringScalar{dtype.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::StringScalar{htype}));
        ARROW_RETURN_NOT_OK(builder->field_builder(4)->AppendScalar(arrow::Int64Scalar{length}));
        ARROW_RETURN_NOT_OK(builder->field_builder(5)->AppendScalar(arrow::BooleanScalar{is_link}));
        ARROW_RETURN_NOT_OK(builder->field_builder(6)->AppendScalar(arrow::BooleanScalar{is_sequence}));
        ARROW_RETURN_NOT_OK(builder->field_builder(7)->AppendScalar(arrow::BooleanScalar{hidden}));
        if (chunk_compression.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(8)->AppendScalar(arrow::StringScalar{chunk_compression.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(8)->AppendNull());
        }
        if (sample_compression.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(9)->AppendScalar(arrow::StringScalar{sample_compression.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(9)->AppendNull());
        }
//        ARROW_RETURN_NOT_OK(builder->field_builder(10)->AppendScalar(arrow::MapScalar{arrow::MapType::Make(arrow::utf8(), arrow::MapType::Make(arrow::utf8(), arrow::utf8())), std::make_shared<arrow::StringArray>(std::make_shared<arrow::StringType>(), 0, std::vector<std::string>{}), std::make_shared<arrow::StringArray>(std::make_shared<arrow::StringType>(), 0, std::vector<std::string>{})}));
        if (max_chunk_size.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(11)->AppendScalar(arrow::Int64Scalar{max_chunk_size.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(11)->AppendNull());
        }
//        ARROW_RETURN_NOT_OK(builder->field_builder(12)->AppendScalar(arrow::ListScalar{std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(std::make_shared<arrow::Int64Type>()), 0, std::make_shared<arrow::Int64Array>(std::make_shared<arrow::Int64Type>(), 0, std::vector<int64_t>{}))}));
//        ARROW_RETURN_NOT_OK(builder->field_builder(13)->AppendScalar(arrow::ListScalar{std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(std::make_shared<arrow::Int64Type>()), 0, std::make_shared<arrow::Int64Array>(std::make_shared<arrow::Int64Type>(), 0, std::vector<int64_t>{}))}));
        if (tiling_threshold.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(14)->AppendScalar(arrow::Int64Scalar{tiling_threshold.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(14)->AppendNull());
        }
        if (typestr.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(15)->AppendScalar(arrow::StringScalar{typestr.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(15)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(16)->AppendScalar(arrow::BooleanScalar{verify}));
        ARROW_RETURN_NOT_OK(builder->field_builder(17)->AppendScalar(arrow::StringScalar{version}));

        return arrow::Status::OK();
    }

}
