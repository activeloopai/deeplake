#include "metadata_action.hpp"
#include "nlohmann/json.hpp"
#include <arrow/api.h>

using json = nlohmann::json;

namespace deeplake {


    deeplake::metadata_action::metadata_action(std::string id,
                                               std::optional<std::string> name,
                                               std::optional<std::string> description,
                                               long created_time) :
            id_(id), name_(name), description_(description), created_time_(created_time) {}

    metadata_action::metadata_action(const nlohmann::json &j) {
        const auto &base = j.at("metadata");
        base.at("id").get_to(id_);
        if (!base.at("name").is_null()) {
            name_ = base.at("name").get<std::string>();
        }
        if (!base.at("description").is_null()) {
            description_ = base.at("description").get<std::string>();
        }

        base.at("createdTime").get_to(created_time_);
    }

    metadata_action::metadata_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        name_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("name").ValueOrDie())->view();
        description_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("description").ValueOrDie())->view();
        created_time_ = reinterpret_pointer_cast<arrow::Int32Scalar>(value->field("description").ValueOrDie())->value;
    }



    std::string deeplake::metadata_action::id() const {
        return id_;
    }

    std::optional<std::string> deeplake::metadata_action::name() const {
        return name_;
    }

    std::optional<std::string> deeplake::metadata_action::description() const {
        return description_;
    }

    long deeplake::metadata_action::created_time() const {
        return created_time_;
    }

    void deeplake::metadata_action::to_json(nlohmann::json &j) {
        j["metadata"]["id"] = id_;
        if (name_.has_value()) {
            j["metadata"]["name"] = name_.value();
        } else {
            j["metadata"]["name"] = json::value_t::null;
        }

        if (description_.has_value()) {
            j["metadata"]["description"] = description_.value();
        } else {
            j["metadata"]["description"] = json::value_t::null;
        }
        j["metadata"]["createdTime"] = created_time_;

    }

    arrow::Status metadata_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id_}));
        if (name_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{name_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendNull());
        }
        if (description_.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::StringScalar{description_.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::Int64Scalar{created_time_}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }

    std::shared_ptr<arrow::StructBuilder> deeplake::metadata_action::arrow_array() {
        auto protocol_struct = arrow::struct_({
                                                      arrow::field("id", arrow::utf8()),
                                                      arrow::field("name", arrow::utf8()),
                                                      arrow::field("description", arrow::utf8()),
                                                      arrow::field("createdTime", arrow::int64()),
                                              });

        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(protocol_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder ()),
        })));

    }

    std::string generate_uuid() {
        std::random_device rd;
        auto seed_data = std::array<int, std::mt19937::state_size>{};
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        std::mt19937 generator(seq);

        return uuids::to_string(uuids::uuid_random_generator{generator}());
    }

    long current_timestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
    }
}