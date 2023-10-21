#include "create_commit_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> create_commit_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8(), true),
                                   arrow::field("branchId", arrow::utf8(), true),
                                   arrow::field("branchVersion", arrow::uint64(), true),
                                   arrow::field("message", arrow::utf8(), true),
                                   arrow::field("commitTime", arrow::uint64(), true),
                           }));

    create_commit_action::create_commit_action(std::string id, std::string branch_id, const unsigned long &branch_version, const std::optional<std::string> &message, const long &commit_time) :
            id(std::move(id)), branch_id(std::move(branch_id)), branch_version(branch_version), message(std::move(message)), commit_time(commit_time) {}

    create_commit_action::create_commit_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = from_struct<std::string>("id", value).value();
        branch_id = from_struct<std::string>("branchId", value).value();
        branch_version = from_struct<long>("branchVersion", value).value();
        message = from_struct<std::string>("message", value);
        commit_time = from_struct<long>("commitTime", value).value();
    }

    std::string create_commit_action::action_name() {
        return "commit";
    }

    std::shared_ptr<arrow::StructType> create_commit_action::action_type() {
        return arrow_type;
    }

    nlohmann::json create_commit_action::to_json() {
        nlohmann::json json;

        json["id"] = id;
        json["branchId"] = branch_id;
        json["branchVersion"] = branch_version;
        json["message"] = to_json_value<std::string>(message);
        json["commitTime"] = commit_time;

        return json;
    }
}