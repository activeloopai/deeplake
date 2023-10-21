#pragma once

#include "action.hpp"

namespace deeplog {
    class create_branch_action : public action {

    public:
        std::string id;
        std::string name;
        std::optional<std::string> from_id;
        std::optional<unsigned long> from_version;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        create_branch_action(std::string id, std::string name, std::optional<std::string> from_id, const std::optional<unsigned long> &from_version);

        explicit create_branch_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;

        std::shared_ptr<arrow::StructType> action_type() override;
    };
}
