#include "create_tensor_action.hpp"

namespace deeplog {
    create_tensor_action::create_tensor_action(std::string id, std::string name,
                                               std::string dtype, std::string htype,
                                               long length, bool link, bool sequence, bool hidden,
                                               std::optional<std::string> chunk_compression, std::optional<std::string> sample_compression,
                                               std::map<std::string, std::map<std::string, std::string>> links,
                                               std::optional<long> max_chunk_size,
                                               std::vector<long> min_shape,
                                               std::vector<long> max_shape,
                                               std::optional<long> tiling_threshold,
                                               std::optional<std::string> typestr,
                                               bool verify, std::string version)
            : id_(id), name_(name),
              dtype_(dtype), htype_(htype),
              length_(length), link_(link), sequence_(sequence), hidden_(hidden),
              chunk_compression_(chunk_compression), sample_compression_(sample_compression),
              links_(links),
              max_chunk_size_(max_chunk_size),
              min_shape_(min_shape),
              max_shape_(max_shape),
              tiling_threshold_(tiling_threshold), typestr_(typestr),
              verify_(verify), version_(version) {}

    create_tensor_action::create_tensor_action(const nlohmann::json &j) {
        const auto &base = j.at("tensor");
        base.at("id").get_to(id_);
        base.at("name").get_to(name_);
        base.at("dtype").get_to(dtype_);
        base.at("htype").get_to(htype_);
        base.at("link").get_to(link_);
        base.at("sequence").get_to(sequence_);
        base.at("hidden").get_to(hidden_);
        if (!base.at("chunkCompression").is_null()) {
            chunk_compression_ = base.at("chunkCompression").get<std::string>();
        }
        if (!base.at("sampleCompression").is_null()) {
            sample_compression_ = base.at("sampleCompression").get<std::string>();
        }
        base.at("links").get_to(links_);
        if (!base.at("maxChunkSize").is_null()) {
            max_chunk_size_ = base.at("maxChunkSize").get<long>();
        }
        base.at("minShape").get_to(min_shape_);
        base.at("maxShape").get_to(max_shape_);
        if (!base.at("maxChunkSize").is_null()) {
            tiling_threshold_ = base.at("tilingThreshold").get<long>();
        }
        if (!base.at("typestr").is_null()) {
            typestr_ = base.at("typestr").get<std::string>();
        }
        base.at("version").get_to(version_);
    }

    create_tensor_action::create_tensor_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        name_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("name").ValueOrDie())->view();
        dtype_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("dtype").ValueOrDie())->view();
        htype_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("htype").ValueOrDie())->view();
        length_ = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("length").ValueOrDie())->value;
        link_ = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("link").ValueOrDie())->value;
        sequence_ = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("sequence").ValueOrDie())->value;
        hidden_ = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("hidden").ValueOrDie())->value;
        chunk_compression_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("chunkCompression").ValueOrDie())->view();
        sample_compression_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("sampleCompression").ValueOrDie())->view();
        max_chunk_size_ = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("maxChunkSize").ValueOrDie())->value;
//        min_shape_ = reinterpret_pointer_cast<arrow::ListScalar>(value->field("minShape").ValueOrDie())->value;
//        max_shape_ = reinterpret_pointer_cast<arrow::ListScalar>(value->field("maxShape").ValueOrDie())->value;
        tiling_threshold_ = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("tilingThreshold").ValueOrDie())->value;
        typestr_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("typestr").ValueOrDie())->view();
        verify_ = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("verify").ValueOrDie())->value;
        version_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("version").ValueOrDie())->view();
    }

    std::string create_tensor_action::id() const {
        return id_;
    }

    std::string create_tensor_action::name() const {
        return name_;
    }

    std::string create_tensor_action::dtype() const {
        return dtype_;
    }

    std::string create_tensor_action::htype() const {
        return htype_;
    }

    long create_tensor_action::length() const {
        return length_;
    }

    bool create_tensor_action::link() const {
        return link_;
    }

    bool create_tensor_action::sequence() const {
        return sequence_;
    }

    bool create_tensor_action::hidden() const {
        return hidden_;
    }

    std::optional<std::string> create_tensor_action::chunk_compression() const {
        return chunk_compression_;
    }

    std::optional<std::string> create_tensor_action::sample_compression() const {
        return sample_compression_;
    }

    std::map<std::string, std::map<std::string, std::string>> create_tensor_action::links() const {
        return links_;
    }

    std::optional<long> create_tensor_action::max_chunk_size() const {
        return max_chunk_size_;
    }

    std::vector<long> create_tensor_action::min_shape() const {
        return min_shape_;
    }

    std::vector<long> create_tensor_action::max_shape() const {
        return max_shape_;
    }

    std::optional<long> create_tensor_action::tiling_threshold() const {
        return tiling_threshold_;
    }

    std::optional<std::string> create_tensor_action::typestr() const {
        return typestr_;
    }

    bool create_tensor_action::verify() const {
        return verify_;
    }

    std::string create_tensor_action::version() const {
        return version_;
    }

    void create_tensor_action::to_json(nlohmann::json &j) {
        j["tensor"]["id"] = id_;
        j["tensor"]["name"] = name_;
        j["tensor"]["dtype"] = dtype_;
        j["tensor"]["htype"] = htype_;
        j["tensor"]["length"] = length_;
        j["tensor"]["link"] = link_;
        j["tensor"]["sequence"] = sequence_;
        j["tensor"]["hidden"] = hidden_;
        if (chunk_compression_.has_value()) {
            j["tensor"]["chunkCompression"] = chunk_compression_.value();
        } else {
            j["tensor"]["chunkCompression"] = nlohmann::json::value_t::null;
        }
        if (sample_compression_.has_value()) {
            j["tensor"]["sampleCompression"] = sample_compression_.value();
        } else {
            j["tensor"]["sampleCompression"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["links"] = links_;
        if (max_chunk_size_.has_value()) {
            j["tensor"]["maxChunkSize"] = max_chunk_size_.value();
        } else {
            j["tensor"]["maxChunkSize"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["minShape"] = min_shape_;
        j["tensor"]["maxShape"] = max_shape_;
        if (tiling_threshold_.has_value()) {
            j["tensor"]["tilingThreshold"] = tiling_threshold_.value();
        } else {
            j["tensor"]["tilingThreshold"] = nlohmann::json::value_t::null;
        }
        if (typestr_.has_value()) {
            j["tensor"]["typestr"] = typestr_.value();
        } else {
            j["tensor"]["typestr"] = nlohmann::json::value_t::null;
        }
        j["tensor"]["verify"] = verify_;
        j["tensor"]["version"] = version_;
    }

    std::shared_ptr<arrow::StructBuilder> create_tensor_action::arrow_array() {
        auto action_struct = arrow::struct_({
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
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{name_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::StringScalar{dtype_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::StringScalar{htype_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(4)->AppendScalar(arrow::Int64Scalar{length_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(5)->AppendScalar(arrow::BooleanScalar{link_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(6)->AppendScalar(arrow::BooleanScalar{sequence_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(7)->AppendScalar(arrow::BooleanScalar{hidden_}));
        if (chunk_compression_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(8)->AppendScalar(arrow::StringScalar{chunk_compression_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(8)->AppendNull());
        }
        if (sample_compression_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(9)->AppendScalar(arrow::StringScalar{sample_compression_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(9)->AppendNull());
        }
//        ARROW_RETURN_NOT_OK(builder->field_builder(10)->AppendScalar(arrow::MapScalar{arrow::MapType::Make(arrow::utf8(), arrow::MapType::Make(arrow::utf8(), arrow::utf8())), std::make_shared<arrow::StringArray>(std::make_shared<arrow::StringType>(), 0, std::vector<std::string>{}), std::make_shared<arrow::StringArray>(std::make_shared<arrow::StringType>(), 0, std::vector<std::string>{})}));
        if (max_chunk_size_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(11)->AppendScalar(arrow::Int64Scalar{max_chunk_size_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(11)->AppendNull());
        }
//        ARROW_RETURN_NOT_OK(builder->field_builder(12)->AppendScalar(arrow::ListScalar{std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(std::make_shared<arrow::Int64Type>()), 0, std::make_shared<arrow::Int64Array>(std::make_shared<arrow::Int64Type>(), 0, std::vector<int64_t>{}))}));
//        ARROW_RETURN_NOT_OK(builder->field_builder(13)->AppendScalar(arrow::ListScalar{std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(std::make_shared<arrow::Int64Type>()), 0, std::make_shared<arrow::Int64Array>(std::make_shared<arrow::Int64Type>(), 0, std::vector<int64_t>{}))}));
        if (tiling_threshold_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(14)->AppendScalar(arrow::Int64Scalar{tiling_threshold_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(14)->AppendNull());
        }
        if (typestr_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(15)->AppendScalar(arrow::StringScalar{typestr_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(15)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(16)->AppendScalar(arrow::BooleanScalar{verify_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(17)->AppendScalar(arrow::StringScalar{version_}));

        return arrow::Status::OK();
    }

}
