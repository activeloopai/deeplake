#pragma once

#include "action.hpp"

namespace deeplog {
    class create_commit_action : public action {

    public:
        std::string id;
        std::string branch_id;
        unsigned long branch_version;
        std::optional<std::string> message;
        long commit_time;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        create_commit_action(std::string id, std::string branch_id, const long &branch_version, const std::optional<std::string> &message, const long &commit_time);

        explicit create_commit_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;

        std::shared_ptr<arrow::StructType> action_type() override;
    };
}
