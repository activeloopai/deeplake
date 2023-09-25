#include "add_file_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::DataType> add_file_action::arrow_struct = arrow::struct_({
                                                                           arrow::field("path", arrow::utf8()),
                                                                           arrow::field("size", arrow::uint64()),
                                                                           arrow::field("modificationTime", arrow::uint64()),
                                                                           arrow::field("dataChange", arrow::boolean()),
                                                                   });

    add_file_action::add_file_action(std::string path, long size, long modification_time, bool data_change) :
            path(path), size(size), modification_time(modification_time), data_change(data_change) {}

    add_file_action::add_file_action(const nlohmann::json &j) {
        const auto &base = j.at("add");
        base.at("path").get_to(path);
        base.at("size").get_to(size);
        base.at("modificationTime").get_to(modification_time);
        base.at("dataChange").get_to(data_change);
    }

    add_file_action::add_file_action(const std::shared_ptr<arrow::StructScalar> &value) {
        path = reinterpret_pointer_cast<arrow::StringScalar>(value->field("path").ValueOrDie())->view();
        size = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("size").ValueOrDie())->value;
        modification_time = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("modificationTime").ValueOrDie())->value;
        data_change = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("dataChange").ValueOrDie())->value;
    }

    void add_file_action::to_json(nlohmann::json &j) {
        j["add"]["path"] = path;
        j["add"]["size"] = size;
        j["add"]["modificationTime"] = modification_time;
        j["add"]["dataChange"] = data_change;
    }

    std::shared_ptr<arrow::StructBuilder> deeplog::add_file_action::arrow_array() {
        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(arrow_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
        })));

    }

    arrow::Status add_file_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{path}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::Int64Scalar{size}));
        ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::Int64Scalar{modification_time}));
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::BooleanScalar{data_change}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }

}