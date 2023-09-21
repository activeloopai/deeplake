#pragma once

#include "action.hpp"

namespace deeplog {
    class add_file_action : public action {

    public:
        add_file_action(std::string path, long size, long modification_time, bool data_change);

        add_file_action(const nlohmann::json &j);

        add_file_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();


    public:
        std::string path;
        long size;
        long modification_time;
        bool data_change;
    };
}
