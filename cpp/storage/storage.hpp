#pragma once

#include <string>
#include <vector>
#include "file_ref.hpp"

namespace storage {
    class storage {
    public:
        virtual file_ref file(const std::string &path) const = 0;

        virtual std::vector<file_ref> list_files(const std::string &base_dir) const = 0;

        virtual std::vector<uint8_t>  get_bytes(const std::string &path) const = 0;

        virtual void set_bytes(const std::string &path, const std::string &data) const = 0;
    };
}
