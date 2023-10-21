#pragma once

#include <arrow/api.h>
#include "deeplog_serializable.hpp"

namespace deeplog {

    class action : public deeplog_serializable {

    public:
        virtual nlohmann::json to_json() = 0;

        virtual std::string action_name() = 0;

        virtual std::shared_ptr<arrow::StructType> action_type() = 0;
    };
}
