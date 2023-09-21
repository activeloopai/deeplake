#pragma once

#include "action.hpp"
#include <uuid.h>
#include <random>
#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class metadata_action : public action {
    public:
        metadata_action(std::string id, std::optional<std::string> name, std::optional<std::string> description,
                        long created_time);

        metadata_action(const nlohmann::json &j);

        metadata_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();

    public:
        std::string id;
        std::optional<std::string> name;
        std::optional<std::string> description;
        long created_time;
    };

    std::string generate_uuid();

    long current_timestamp();
}
