#pragma once

#include <string>

namespace storage {
    struct file_ref {
        std::string path;
        long size;
        bool exists();

        file_ref(std::string path, long size) : path(std::move(path)), size(size) {}

        bool operator < (const file_ref &other) const {
            return path < other.path;
        }
    };
}
