#include "remove_file_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> remove_file_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("path", arrow::utf8()),
                                   arrow::field("deletionTime", arrow::int64()),
                                   arrow::field("dataChange", arrow::boolean()),
                                   arrow::field("size", arrow::uint64()),
                           }));

    remove_file_action::remove_file_action(std::string path, const long &size, const long &deletion_timestamp, const bool &data_change) :
            path(std::move(path)), size(size), deletion_time(deletion_timestamp), data_change(data_change) {};

    std::string remove_file_action::action_name() {
        return "remove";
    }

    nlohmann::json remove_file_action::to_json() {
        nlohmann::json json;

        json["path"] = path;
        json["deletionTime"] = deletion_time;
        json["dataChange"] = data_change;
        json["size"] = size;

        return json;
    }
}
