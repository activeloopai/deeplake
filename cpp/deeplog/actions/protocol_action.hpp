#pragma once

#include "action.hpp"
#include <arrow/api.h>

namespace deeplog {

    class protocol_action : public action {
    public:
        int min_reader_version;
        int min_writer_version;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        protocol_action(const int &min_reader_version, const int &min_writer_version);

        explicit protocol_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;
    };

}
