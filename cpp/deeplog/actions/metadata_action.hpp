#pragma once

#include "action.hpp"
#include "replace_action.hpp"
#include <random>
#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class metadata_action : public action, public replace_action, public std::enable_shared_from_this<metadata_action> {
    public:
        std::string id;
        std::optional<std::string> name;
        std::optional<std::string> description;
        long created_time;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        metadata_action(std::string id, const std::optional<std::string> &name, const std::optional<std::string> &description,
                        const long &created_time);

        explicit metadata_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;

        std::shared_ptr<arrow::StructType> action_type() override;

        bool replaces(std::shared_ptr<action> action) override;

        std::shared_ptr<action> replace(std::shared_ptr<action> action) override;

    };
}
