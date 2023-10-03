#include "local_storage.hpp"

#include <utility>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace storage {

    local_storage::local_storage(std::string path) : path_(std::filesystem::absolute(path)) {
        std::filesystem::create_directories(path_);
    }

    std::filesystem::path local_storage::full_path(const std::string &path) const {
        auto sub_path = path;
        if (sub_path.find('/') == 0) {
            sub_path = path.substr(1);
        }
        return path_ / std::filesystem::path(sub_path);
    }

    file_ref local_storage::file(const std::string &path) const {
        auto file_path = full_path(path);
        if (std::filesystem::exists(file_path)) {
            if (std::filesystem::is_regular_file(file_path)) {
                return file_ref(path, std::filesystem::file_size(file_path));
            } else {
                return file_ref(path, 0);
            }
        } else {
            return file_ref(path, -1);
        }
    }

    std::vector<file_ref> local_storage::list_files(const std::string &base_dir) const {
        auto base_dir_path = full_path(base_dir);
        std::vector<file_ref> files;
        for (const auto &entry: std::filesystem::directory_iterator(base_dir_path)) {
            if (entry.is_regular_file()) {
                files.push_back(file_ref("/" + std::filesystem::relative(entry.path(), path_).string(), entry.file_size()));
            }
        }
        return files;
    }

    std::vector<uint8_t> local_storage::get_bytes(const std::string &path) const {
        auto final_path = full_path(path);

        auto file = std::ifstream(final_path);

        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + final_path.string());
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        if (fileSize <= 0) {
            return {};
        }

        std::vector<uint8_t> return_data(static_cast<size_t>(fileSize));
        file.read(reinterpret_cast<char*>(return_data.data()), fileSize);


        if (!file) {
            throw std::runtime_error("Error reading file: " + final_path.string());
        }
        file.close();

        return return_data;
    }

    void local_storage::set_bytes(const std::string &path, const std::string &data) const {
        auto final_path = full_path(path);
        std::filesystem::create_directories(final_path.parent_path());

        std::ofstream stream {final_path};
        stream << data;
        stream.close();
    }

}