#include "metadata_action.hpp"
#include <nlohmann/json.hpp>
#include <arrow/api.h>
#include <iostream>

using json = nlohmann::json;

namespace deeplog {

    std::shared_ptr<arrow::StructType> metadata_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8()),
                                   arrow::field("name", arrow::utf8()),
                                   arrow::field("description", arrow::utf8()),
                                   arrow::field("createdTime", arrow::int64()),
                           }));

    deeplog::metadata_action::metadata_action(std::string id, const std::optional<std::string> &name, const std::optional<std::string> &description,
                                              const long &created_time) :
            id(std::move(id)), name(std::move(name)), description(std::move(description)), created_time(created_time) {}

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

    std::string metadata_action::action_name() {
        return "metadata";
    }

    nlohmann::json deeplog::metadata_action::to_json() {
        nlohmann::json json;

        json["id"] = id;
        if (name.has_value()) {
            json["name"] = name.value();
        } else {
            json["name"] = json::value_t::null;
        }

        if (description.has_value()) {
            json["description"] = description.value();
        } else {
            json["description"] = json::value_t::null;
        }
        json["createdTime"] = created_time;

        return json;

    }

    std::string generate_uuid() {
        std::random_device rd;
        auto seed_data = std::array<int, std::mt19937::state_size>{};
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        std::mt19937 generator(seq);

        return uuids::to_string(uuids::uuid_random_generator{generator}());
    }

    bool metadata_action::replaces(std::shared_ptr<action> action) {
        return action->action_name() == action_name();
    }

    std::shared_ptr<action> metadata_action::replace(std::shared_ptr<action> action) {
        return shared_from_this();
    }

    long current_timestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
    }
}