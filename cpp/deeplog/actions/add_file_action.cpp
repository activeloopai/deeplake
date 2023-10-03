#include "add_file_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> add_file_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("path", arrow::utf8(), true),
                                   arrow::field("size", arrow::uint64(), true),
                                   arrow::field("modificationTime", arrow::uint64(), true),
                                   arrow::field("dataChange", arrow::boolean(), true),
                           }));

    add_file_action::add_file_action(std::string path, const long &size, const long &modification_time, const bool &data_change) :
            path(std::move(path)), size(size), modification_time(modification_time), data_change(data_change) {}

    add_file_action::add_file_action(const std::shared_ptr<arrow::StructScalar> &value) {
        path = from_struct<std::string>("path", value).value();
        size = from_struct<unsigned long>("size", value).value();
        modification_time = from_struct<long>("modificationTime", value).value();
        data_change = from_struct<bool>("dataChange", value).value();
    }

    std::string add_file_action::action_name() {
        return "add";
    }

    std::shared_ptr<arrow::StructType> add_file_action::action_type() {
        return arrow_type;
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