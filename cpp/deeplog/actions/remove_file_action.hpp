#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "action.hpp"

namespace deeplog {
    class remove_file_action : public action {
    public:
        static std::shared_ptr<arrow::DataType> arrow_struct;

        remove_file_action(std::string path, long size, long deletion_timestamp, bool data_change);

        remove_file_action(const nlohmann::json &j);

        virtual void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;


    public:
        std::string path;
        long deletion_time;
        bool data_change;
        long size;
    };
}
