#pragma once

#include "storage.hpp"
#include <string>
#include <iostream>
#include <filesystem>

namespace storage {
    class local_storage : public ::storage::storage {

    public:
        local_storage(std::string  path);

        file_ref file(const std::string &path) const override;

        std::vector<file_ref> list_files(const std::string &base_dir) const override;

        std::vector<uint8_t>  get_bytes(const std::string &path) const override;

        void set_bytes(const std::string &path, const std::string &data) const override;

    private:
        std::filesystem::path path_;

        std::filesystem::path full_path(const std::string &path) const;
    };
}
