#pragma once

#include <chrono>
#include <string>
#include <utility>

namespace storage {

struct resource_meta
{
    std::string path;
    uintmax_t size = 0;
    std::chrono::system_clock::time_point last_modified;
    std::string etag;

    bool operator<(const resource_meta& rhs) const
    {
        return path < rhs.path;
    }

    bool operator==(const resource_meta& rhs) const = default;

    explicit operator bool() const noexcept
    {
        return !path.empty();
    }
};

} // namespace storage
