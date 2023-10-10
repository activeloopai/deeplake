#include "deeplog_serializable.hpp"
#include "tensor_link.hpp"
#include <iostream>

namespace deeplog {

    template<typename T>
    nlohmann::json deeplog_serializable::to_json_value(const std::optional<T> &value) const {
        if (!value.has_value()) {
            return nlohmann::json::value_t::null;
        }

        if constexpr (std::is_same<T, std::map<std::string, tensor_link>>::value) {
            auto return_map = nlohmann::json::object();
            for (auto &item : value.value()) {
                return_map[item.first] = item.second.to_json();
            }
            return return_map;
        } else {
            return value.value();
        }
    }

    template nlohmann::json deeplog_serializable::to_json_value(const std::optional<bool> &value) const;

    template nlohmann::json deeplog_serializable::to_json_value(const std::optional<std::string> &value) const;

    template nlohmann::json deeplog_serializable::to_json_value(const std::optional<long> &value) const;

    template nlohmann::json deeplog_serializable::to_json_value(const std::optional<unsigned long> &value) const;

    //need to figure out how to not need to specify every map template option. Same below
    template nlohmann::json deeplog_serializable::to_json_value(const std::optional<std::map<std::string, tensor_link>> &value) const;

    template<typename T>
    std::optional<T> deeplog_serializable::from_struct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar) {
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
        } else if constexpr (std::is_same<T, std::map<std::string, tensor_link>>::value) {
            auto map_data = std::reinterpret_pointer_cast<arrow::MapScalar>(scalar)->value;
            T return_map = {};

            for (auto i = 0; i < map_data->length(); ++i) {
                auto raw_scalar = map_data->GetScalar(i).ValueOrDie();
                auto map_struct = std::dynamic_pointer_cast<arrow::StructScalar>(raw_scalar);
                std::string key = std::dynamic_pointer_cast<arrow::StringScalar>(map_struct->field("key").ValueOrDie())->view().data();
                tensor_link value = tensor_link(std::dynamic_pointer_cast<arrow::StructScalar>(map_struct->field("value").ValueOrDie()));
                return_map.insert({key, value});
            }

            return return_map;
        } else {
            throw std::runtime_error("Unsupported struct type: " + std::string(typeid(T).name()));
        }
    }

    template<typename T>
    std::vector<T> deeplog_serializable::from_arraystruct(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar) {
        auto list_scalar = std::reinterpret_pointer_cast<arrow::ListScalar>(struct_scalar->field(field_name).ValueOrDie());
        if (!list_scalar->is_valid) {
            return {};
        }

        auto array = std::reinterpret_pointer_cast<arrow::UInt64Array>(list_scalar->value);

        std::vector<T> return_vector = {};
        return_vector.reserve(array->length());
        for (auto i = 0; i < array->length(); ++i) {
            return_vector.push_back(array->Value(i));
        }

        return return_vector;
    }

    template std::optional<std::string> deeplog_serializable::from_struct<std::string>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<long> deeplog_serializable::from_struct<long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<unsigned long> deeplog_serializable::from_struct<unsigned long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<bool> deeplog_serializable::from_struct<bool>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<int> deeplog_serializable::from_struct<int>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::optional<std::map<std::string, tensor_link>> deeplog_serializable::from_struct<std::map<std::string, tensor_link>>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);

    template std::vector<unsigned long> deeplog_serializable::from_arraystruct<unsigned long>(const std::string &field_name, const std::shared_ptr<arrow::StructScalar> &struct_scalar);
};