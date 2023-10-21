#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>
#include "deeplog_serializable.hpp"

namespace deeplog {

    class tensor_link : public deeplog_serializable {
    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        std::optional<std::string> extend;
        std::optional<bool> flatten_sequence;
        std::optional<std::string> update;

        tensor_link(const std::optional<std::string> &extend, const std::optional<bool> &flatten_sequence, const std::optional<std::string> &update);

        explicit tensor_link(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() const;
    };
}
