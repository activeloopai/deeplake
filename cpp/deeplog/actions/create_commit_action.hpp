#pragma once

#include "action.hpp"

namespace deeplog {
    class create_commit_action : public action {

    public:
        static std::shared_ptr<arrow::StructBuilder> arrow_array();

        create_commit_action(std::string id, std::string branch_id, long branch_version, std::optional<std::string> message, long commit_time);

        create_commit_action(const nlohmann::json &j);

        create_commit_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);


        std::string id() const;

        std::string branch_id() const;

        long branch_version() const;

        std::optional<std::string> message() const;

        long commit_time() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

    private:
        std::string id_;
        std::string branch_id_;
        long branch_version_;
        std::optional<std::string> message_;
        long commit_time_;
    };
}
