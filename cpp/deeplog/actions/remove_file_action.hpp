#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "action.hpp"

namespace deeplog {
    class remove_file_action : public action {
    public:
        std::string path;
        long deletion_time;
        bool data_change;
        unsigned long size;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        remove_file_action(std::string path, const long &size, const long &deletion_timestamp, const bool &data_change);

        explicit remove_file_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;
    };
}
