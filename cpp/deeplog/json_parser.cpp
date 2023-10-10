#include "json_parser.hpp"
#include "spdlog/spdlog.h"
#include <iostream>

#include <arrow/json/api.h>

namespace deeplog {
    arrow::Status json_parser::parse(const std::shared_ptr<arrow::io::InputStream> &buffer_reader, const std::shared_ptr<arrow::RecordBatchBuilder> &batch_builder) {
        ARROW_ASSIGN_OR_RAISE(auto json_reader, arrow::json::StreamingReader::Make(buffer_reader,
                                                                                   arrow::json::ReadOptions::Defaults(),
                                                                                   arrow::json::ParseOptions::Defaults()));

        for (arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_json: *json_reader) {
            if (!maybe_json.ok()) {
                throw std::runtime_error("Error reading JSON: " + maybe_json.status().message());
            }
            std::shared_ptr<arrow::RecordBatch> json = *maybe_json;

            auto json_columns = json->schema()->field_names();

            for (int builder_column_id = 0; builder_column_id < batch_builder->num_fields(); ++builder_column_id) {
                const auto builder_column_name = batch_builder->schema()->field(builder_column_id)->name();
                if (builder_column_name == "version") {
                    //doesn't come from json
                    ARROW_RETURN_NOT_OK(batch_builder->GetField(builder_column_id)->AppendNulls(json->num_rows()));
                    continue;
                }

                const auto column_builder = batch_builder->GetFieldAs<arrow::StructBuilder>(builder_column_id);
                if (column_builder == nullptr) {
                    throw std::runtime_error("Unexpected builder for field " + std::to_string(builder_column_id));
                }

                const auto json_column_it = std::find(json_columns.begin(), json_columns.end(), builder_column_name);

                if (json_column_it == json_columns.end()) {
                    spdlog::debug("No {} columns in JSON", builder_column_name);

                    ARROW_RETURN_NOT_OK(column_builder->AppendNulls(json->num_rows()));
                    continue;
                }
                const auto json_column_id = static_cast<int>(std::distance(json_columns.begin(), json_column_it));

                for (int row = 0; row < json->num_rows(); ++row) {
                    ARROW_ASSIGN_OR_RAISE(auto json_column_value_scalar, json->column(json_column_id)->GetScalar(row));
                    if (!json_column_value_scalar->is_valid) {
                        ARROW_RETURN_NOT_OK(column_builder->AppendNull());
                        continue;
                    }

                    auto json_column_value = std::dynamic_pointer_cast<arrow::StructScalar>(json_column_value_scalar);

                    if (json_column_value == nullptr) {
                        throw std::runtime_error("Unexpected json data type in " + std::to_string(json_column_id) + ": " + json_column_value_scalar->ToString());
                    }

                    auto column_type = column_builder->type();
                    for (int builder_field_id = 0; builder_field_id < column_type->num_fields(); ++builder_field_id) {
                        auto field_name = column_type->field(builder_field_id)->name();
                        auto field_builder = column_builder->field_builder(builder_field_id);

                        std::vector<std::string> json_struct_fields = {};
                        for (const auto &field: json_column_value->type->fields()) {
                            json_struct_fields.push_back(field->name());
                        }
                        const auto json_struct_field_it = std::find(json_struct_fields.begin(), json_struct_fields.end(), field_name);

                        if (json_struct_field_it == json_struct_fields.end()) {
                            spdlog::debug("No {} records in JSON", field_name);

                            ARROW_RETURN_NOT_OK(field_builder->AppendNull());
                            continue;
                        }
                        const auto json_struct_field_id = static_cast<int>(std::distance(json_struct_fields.begin(), json_struct_field_it));

                        auto json_field_value = json_column_value->value.at(json_struct_field_id);
                        if (!json_field_value->is_valid) {
                            ARROW_RETURN_NOT_OK(field_builder->AppendNull());
                            continue;
                        }

                        auto wanted_type = field_builder->type();
                        ARROW_ASSIGN_OR_RAISE(auto converted_value, convert(json_field_value, wanted_type));

                        ARROW_RETURN_NOT_OK(field_builder->AppendScalar(*converted_value));
                    }
                    ARROW_RETURN_NOT_OK(column_builder->Append()); //finish scalar
                }
            }
        }

        return arrow::Status::OK();
    }

    arrow::Result<std::shared_ptr<arrow::Scalar>> json_parser::convert(const std::shared_ptr<arrow::Scalar> &json_field_value, const std::shared_ptr<arrow::DataType> &wanted_type) {

        if (wanted_type->Equals(json_field_value->type)) {
            return json_field_value;
        }
        if (json_field_value->type->Equals(arrow::int64())) {
            ARROW_ASSIGN_OR_RAISE(auto new_value, arrow::MakeScalar(wanted_type, std::dynamic_pointer_cast<arrow::Int64Scalar>(json_field_value)->value));
            return new_value;
        } else if (json_field_value->type->Equals(arrow::utf8())) {
            ARROW_ASSIGN_OR_RAISE(auto new_value, arrow::MakeScalar(wanted_type, std::dynamic_pointer_cast<arrow::StringScalar>(json_field_value)->value));
            return new_value;
        } else if (json_field_value->type->id() == arrow::Type::LIST) {
            auto list_type = wanted_type->field(0)->type();
            ARROW_ASSIGN_OR_RAISE(auto array_builder, arrow::MakeBuilder(list_type));

            auto json_list = std::dynamic_pointer_cast<arrow::ListScalar>(json_field_value)->value;
            for (auto i = 0; i < json_list->length(); ++i) {
                ARROW_ASSIGN_OR_RAISE(auto current_value, json_list->GetScalar(i));
                ARROW_ASSIGN_OR_RAISE(auto converted, convert(current_value, list_type));
                ARROW_RETURN_NOT_OK(array_builder->AppendScalar(*converted));
            }
            ARROW_ASSIGN_OR_RAISE(auto array, array_builder->Finish());
            ARROW_ASSIGN_OR_RAISE(auto new_value, arrow::MakeScalar(wanted_type, array));
            return new_value;
        } else if (json_field_value->type->id() == arrow::Type::STRUCT) {
            auto json_struct = std::dynamic_pointer_cast<arrow::StructScalar>(json_field_value);

            if (wanted_type->id() == arrow::Type::MAP) {
                auto wanted_type_struct = std::dynamic_pointer_cast<arrow::StructType>(wanted_type->field(0)->type());
                const auto &wanted_key_type = arrow::utf8();
                auto wanted_value_type = wanted_type_struct->field(1)->type();

                ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::ArrayBuilder> map_struct_builder, arrow::MakeBuilder(arrow::struct_({field("key", arrow::utf8(), false), field("value", wanted_value_type)})));

                for (auto json_field: json_struct->type->fields()) {
                    ARROW_ASSIGN_OR_RAISE(auto converted_key, convert(arrow::MakeScalar(json_field->name()), wanted_key_type));
                    ARROW_RETURN_NOT_OK(map_struct_builder->child(0)->AppendScalar(*converted_key));

                    ARROW_ASSIGN_OR_RAISE(auto original_json_field_value, json_struct->field(json_field->name()));
                    ARROW_ASSIGN_OR_RAISE(auto converted_value, convert(original_json_field_value, wanted_value_type));
                    ARROW_RETURN_NOT_OK(map_struct_builder->child(1)->AppendScalar(*converted_value));

                    ARROW_RETURN_NOT_OK(std::dynamic_pointer_cast<arrow::StructBuilder>(map_struct_builder)->Append());
                }


                ARROW_ASSIGN_OR_RAISE(auto map_array, map_struct_builder->Finish());

                return std::make_shared<arrow::MapScalar>(arrow::MapScalar(map_array));
            } else if (wanted_type->id() == arrow::Type::STRUCT) {
                std::vector<std::shared_ptr<arrow::Scalar>> struct_scalars = {};
                std::vector<std::string> field_names = {};

                for (auto i=0; i<wanted_type->num_fields(); ++i) {
                    auto struct_field  = wanted_type->field(i);
                    field_names.push_back(struct_field->name());

                    ARROW_ASSIGN_OR_RAISE(auto json_struct_field_value, json_struct->field(struct_field->name()));
                    ARROW_ASSIGN_OR_RAISE(auto converted_json_field_value, convert(json_struct_field_value, struct_field->type()));

                    struct_scalars.push_back(converted_json_field_value);
                }

                return arrow::StructScalar::Make(struct_scalars, field_names);
            } else {
                throw std::runtime_error("Unexpected struct mapping: " + wanted_type->name());
            }
        } else {
            throw std::runtime_error("Unexpected json type: " + json_field_value->type->name());
        }
    }
}