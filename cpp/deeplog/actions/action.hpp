#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class action {

    public:
        virtual nlohmann::json to_json() = 0;

        virtual std::string action_name() = 0;

        template<typename T>
        std::optional<T> from_struct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        template<typename T>
        std::vector<T> from_arraystruct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        template <typename T>
        nlohmann::json to_json_value(const std::optional<T> &value);
    };
}
