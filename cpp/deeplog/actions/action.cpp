#include "action.hpp"

namespace deeplog {

    template<typename T>
    std::optional<T> action::from_struct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar) {
        auto scalar = struct_scalar->field(field_name).ValueOrDie();
        if (!scalar->is_valid) {
            return std::nullopt;
        }


        if constexpr (std::is_same<T, std::string>::value) {
            return std::reinterpret_pointer_cast<arrow::StringScalar>(scalar)->value->ToString();
        } else if constexpr (std::is_same<T, long>::value) {
            return std::reinterpret_pointer_cast<arrow::Int64Scalar>(scalar)->value;
        } else if constexpr (std::is_same<T, unsigned long>::value) {
            return std::reinterpret_pointer_cast<arrow::UInt64Scalar>(scalar)->value;
        } else if constexpr (std::is_same<T, int>::value) {
            return std::reinterpret_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        } else if constexpr (std::is_same<T, bool>::value) {
            return std::reinterpret_pointer_cast<arrow::BooleanScalar>(scalar)->value;
        } else {
            throw std::runtime_error("Unsupported struct type: " + std::string(typeid(T).name()));
        }
    }

    template<typename T>
    std::vector<T> action::from_arraystruct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar) {
        auto list_scalar = reinterpret_pointer_cast<arrow::ListScalar>(struct_scalar->field(field_name).ValueOrDie());
        if (!list_scalar->is_valid) {
            return {};
        }

        auto array = reinterpret_pointer_cast<arrow::UInt64Array>(list_scalar->value);

        std::vector<T> return_vector = {};
        return_vector.reserve(array->length());
        for (auto i = 0; i < array->length(); ++i) {
            return_vector.push_back(array->Value(i));
        }

        return return_vector;
    }


    template<typename T>
    nlohmann::json action::to_json_value(const std::optional<T> &value) {
        if (!value.has_value()) {
            return nlohmann::json::value_t::null;
        }

        return value.value();
    }

    template std::optional<std::string> action::from_struct<std::string>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<long> action::from_struct<long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<unsigned long> action::from_struct<unsigned long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<bool> action::from_struct<bool>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<int> action::from_struct<int>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::vector<unsigned long> action::from_arraystruct<unsigned long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template nlohmann::json action::to_json_value(const std::optional<std::string> &value);

    template nlohmann::json action::to_json_value(const std::optional<long> &value);

};
