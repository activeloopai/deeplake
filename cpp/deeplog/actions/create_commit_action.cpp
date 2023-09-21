#include "create_commit_action.hpp"

namespace deeplog {
    create_commit_action::create_commit_action(std::string id, std::string branch_id, long branch_version, std::optional<std::string> message, long commit_time) :
            id(std::move(id)), branch_id(std::move(branch_id)), branch_version(branch_version), message(std::move(message)), commit_time(commit_time) {}

    create_commit_action::create_commit_action(const nlohmann::json &j) {
        const auto &base = j.at("commit");
        base.at("id").get_to(id);
        base.at("branchId").get_to(branch_id);
        base.at("branchVersion").get_to(branch_version);
        if (!base.at("message").is_null()) {
            message = base.at("message").get<std::string>();
        }
        base.at("commitTime").get_to(commit_time);
    }

    create_commit_action::create_commit_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        branch_id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("branchId").ValueOrDie())->view();
        branch_version = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("branchVersion").ValueOrDie())->value;
        message = reinterpret_pointer_cast<arrow::StringScalar>(value->field("message").ValueOrDie())->view();
        commit_time = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("commitTime").ValueOrDie())->value;
    }

    void create_commit_action::to_json(nlohmann::json &j) {
        j["commit"]["id"] = id;
        j["commit"]["branchId"] = branch_id;
        j["commit"]["branchVersion"] = branch_version;
        if (message.has_value()) {
            j["commit"]["message"] = message.value();
        }
        j["commit"]["commitTime"] = commit_time;
    }

    std::shared_ptr<arrow::StructBuilder> deeplog::create_commit_action::arrow_array() {
        auto action_struct = arrow::struct_({
                                                      arrow::field("id", arrow::utf8()),
                                                      arrow::field("branchId", arrow::utf8()),
                                                      arrow::field("branchVersion", arrow::uint64()),
                                                      arrow::field("message", arrow::utf8()),
                                                      arrow::field("commitTime", arrow::uint64()),
                                              });

        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(action_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
        })));

    }

    arrow::Status create_commit_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{branch_id}));
        ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::Int64Scalar{branch_version}));
        if (message.has_value()) {
            ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::StringScalar{message.value()}));
        } else {
            ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendNull());
        }
        ARROW_RETURN_NOT_OK(builder->field_builder(4)->AppendScalar(arrow::Int64Scalar{commit_time}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }

}