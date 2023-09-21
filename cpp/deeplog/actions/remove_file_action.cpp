#include "remove_file_action.hpp"

namespace deeplog {
    remove_file_action::remove_file_action(const nlohmann::json &j) {
        const auto &base = j.at("remove");
        base.at("path").get_to(path);
        base.at("deletionTime").get_to(deletion_time);
        base.at("dataChange").get_to(data_change);
        base.at("size").get_to(size);
    }

    remove_file_action::remove_file_action(std::string path, long size, long deletion_timestamp, bool data_change) :
            path(path), size(size), deletion_time(deletion_timestamp), data_change(data_change) {};

    void remove_file_action::to_json(nlohmann::json &j) {
        j["remove"]["path"] = path;
        j["remove"]["deletionTime"] = deletion_time;
        j["remove"]["dataChange"] = data_change;
        j["remove"]["size"] = size;
    }

    arrow::Status remove_file_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        return arrow::Status::OK();
    }

}
