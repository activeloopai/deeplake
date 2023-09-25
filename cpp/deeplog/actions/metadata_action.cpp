#include "metadata_action.hpp"
#include <nlohmann/json.hpp>
#include <arrow/api.h>
#include <iostream>

using json = nlohmann::json;

namespace deeplog {

    std::shared_ptr<arrow::DataType> metadata_action::arrow_struct = arrow::struct_({
                                                           arrow::field("id", arrow::utf8()),
                                                           arrow::field("name", arrow::utf8()),
                                                           arrow::field("description", arrow::utf8()),
                                                           arrow::field("createdTime", arrow::int64()),
                                                   });

    deeplog::metadata_action::metadata_action(std::string id,
                                               std::optional<std::string> name,
                                               std::optional<std::string> description,
                                               long created_time) :
            id(id), name(name), description(description), created_time(created_time) {}

    metadata_action::metadata_action(const nlohmann::json &j) {
        const auto &base = j.at("metadata");
        base.at("id").get_to(id);
        if (!base.at("name").is_null()) {
            name = base.at("name").get<std::string>();
        }
        if (!base.at("description").is_null()) {
            description = base.at("description").get<std::string>();
        }

        base.at("createdTime").get_to(created_time);
    }

    metadata_action::metadata_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        auto scalar = value->field("name").ValueOrDie();
        if (scalar->is_valid) {
            name = reinterpret_pointer_cast<arrow::StringScalar>(scalar)->view();
        }
        scalar = value->field("description").ValueOrDie();
        if (scalar->is_valid) {
            description = reinterpret_pointer_cast<arrow::StringScalar>(scalar)->view();
        }
        created_time = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("createdTime").ValueOrDie())->value;
    }

    void deeplog::metadata_action::to_json(nlohmann::json &j) {
        j["metadata"]["id"] = id;
        if (name.has_value()) {
            j["metadata"]["name"] = name.value();
        } else {
            j["metadata"]["name"] = json::value_t::null;
        }

        if (description.has_value()) {
            j["metadata"]["description"] = description.value();
        } else {
            j["metadata"]["description"] = json::value_t::null;
        }
        j["metadata"]["createdTime"] = created_time;

    }

    arrow::Status metadata_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id}));
        if (name.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{name.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendNull());
        }
        if (description.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::StringScalar{description.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::Int64Scalar{created_time}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }

    std::shared_ptr<arrow::StructBuilder> deeplog::metadata_action::arrow_array() {
        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(arrow_struct, arrow::default_memory_pool(), {
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