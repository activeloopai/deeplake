#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class action {

    public:

        virtual void to_json(nlohmann::json &json) = 0;
        virtual arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) = 0;

    };


}
