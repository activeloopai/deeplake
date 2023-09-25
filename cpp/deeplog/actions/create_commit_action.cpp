#include "create_commit_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> create_commit_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("id", arrow::utf8()),
                                   arrow::field("branchId", arrow::utf8()),
                                   arrow::field("branchVersion", arrow::uint64()),
                                   arrow::field("message", arrow::utf8()),
                                   arrow::field("commitTime", arrow::uint64()),
                           }));

    create_commit_action::create_commit_action(std::string id, std::string branch_id, const long &branch_version, const std::optional<std::string> &message, const long &commit_time) :
            id(std::move(id)), branch_id(std::move(branch_id)), branch_version(branch_version), message(std::move(message)), commit_time(commit_time) {}

    create_commit_action::create_commit_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        branch_id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("branchId").ValueOrDie())->view();
        branch_version = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("branchVersion").ValueOrDie())->value;
        message = reinterpret_pointer_cast<arrow::StringScalar>(value->field("message").ValueOrDie())->view();
        commit_time = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("commitTime").ValueOrDie())->value;
    }

    std::string create_commit_action::action_name() {
        return "commit";
    }

    nlohmann::json create_commit_action::to_json() {
        nlohmann::json json;

        json["id"] = id;
        json["branchId"] = branch_id;
        json["branchVersion"] = branch_version;
        if (message.has_value()) {
            json["message"] = message.value();
        }
        json["commitTime"] = commit_time;

        return json;
    }
}