#pragma once

#include "action.hpp"
#include <uuid.h>
#include <random>
#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class metadata_action : public action {
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
    };

    std::string generate_uuid();

    long current_timestamp();
}
