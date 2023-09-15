#include "remove_file_action.hpp"

namespace deeplake {
    remove_file_action::remove_file_action(const nlohmann::json &j) {
        const auto &base = j.at("remove");
        base.at("path").get_to(path_);
        base.at("deletionTime").get_to(deletion_time_);
        base.at("dataChange").get_to(data_change_);
        base.at("size").get_to(size_);
    }

    remove_file_action::remove_file_action(std::string path, long size, long deletion_timestamp, bool data_change) :
            path_(path), size_(size), deletion_time_(deletion_timestamp), data_change_(data_change) {};

    std::string remove_file_action::path() { return path_; }

    long remove_file_action::size() { return size_; }

    long remove_file_action::deletion_timestamp() { return deletion_time_; }

    void remove_file_action::to_json(nlohmann::json &j) {
        j["remove"]["path"] = path_;
        j["remove"]["deletionTime"] = deletion_time_;
        j["remove"]["dataChange"] = data_change_;
        j["remove"]["size"] = size_;
    }

    arrow::Status remove_file_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        return arrow::Status::OK();
    }

}
