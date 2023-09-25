#include "add_file_action.hpp"

#include <utility>

namespace deeplog {

    std::shared_ptr<arrow::StructType> add_file_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("path", arrow::utf8()),
                                   arrow::field("size", arrow::uint64()),
                                   arrow::field("modificationTime", arrow::uint64()),
                                   arrow::field("dataChange", arrow::boolean()),
                           }));

    add_file_action::add_file_action(std::string path, const long &size, const long &modification_time, const bool &data_change) :
            path(std::move(path)), size(size), modification_time(modification_time), data_change(data_change) {}

    add_file_action::add_file_action(const std::shared_ptr<arrow::StructScalar> &value) {
        path = reinterpret_pointer_cast<arrow::StringScalar>(value->field("path").ValueOrDie())->view();
        size = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("size").ValueOrDie())->value;
        modification_time = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("modificationTime").ValueOrDie())->value;
        data_change = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("dataChange").ValueOrDie())->value;
    }

    std::string add_file_action::action_name() {
        return "add";
    }

    nlohmann::json add_file_action::to_json() {
        nlohmann::json json;
        json["path"] = path;
        json["size"] = size;
        json["modificationTime"] = modification_time;
        json["dataChange"] = data_change;

        return json;
    }
}