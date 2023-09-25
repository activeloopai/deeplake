#pragma once

#include "action.hpp"

namespace deeplog {
    class create_branch_action : public action {

    public:
        static std::shared_ptr<arrow::DataType> arrow_struct;

        create_branch_action(std::string id, std::string name, std::string from_id, long from_version);

        create_branch_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        create_branch_action(const nlohmann::json &j);

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;


    public:
        std::string id;
        std::string name;
        std::string from_id;
        long from_version;

    };
}
