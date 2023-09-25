#pragma once

#include "action.hpp"
#include <arrow/api.h>

namespace deeplog {

    class protocol_action : public action {
    public:
        static std::shared_ptr<arrow::StructBuilder> arrow_array();

        static std::shared_ptr<arrow::DataType> arrow_struct;

        protocol_action(int min_reader_version, int min_writer_version);

        protocol_action(const nlohmann::json &j);

        protocol_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

    public:
        int min_reader_version;
        int min_writer_version;
    };

}
