#include "create_branch_action.hpp"

#include <utility>

namespace deeplog {
    std::shared_ptr<arrow::StructType> create_branch_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8(), true),
                                   arrow::field("name", arrow::utf8(), true),
                                   arrow::field("fromId", arrow::utf8(), true),
                                   arrow::field("fromVersion", arrow::int64(), true),
                           }));


    create_branch_action::create_branch_action(std::string id, std::string name, std::string from_id, const long &from_version) :
            id(std::move(id)), name(std::move(name)), from_id(std::move(from_id)), from_version(from_version) {}

    create_branch_action::create_branch_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = from_struct<std::string>("id", value).value();
        name = from_struct<std::string>("name", value).value();
        from_id = from_struct<std::string>("fromId", value).value();
        from_version = from_struct<long>("fromVersion", value).value();
    }

    std::string create_branch_action::action_name() {
        return "branch";
    }

    std::shared_ptr<arrow::StructType> create_branch_action::action_type() {
        return arrow_type;
    }

    nlohmann::json create_branch_action::to_json() {
        nlohmann::json json;
        json["id"] = id;
        json["name"] = name;
        json["fromId"] = from_id;
        json["fromVersion"] = from_version;

        return json;
    }
}