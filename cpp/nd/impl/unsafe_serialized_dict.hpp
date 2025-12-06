#pragma once

#include "../array.hpp"

#include <icm/const_json.hpp>

#include <string>
#include <vector>

namespace nd::impl {

struct unsafe_serialized_dict
{
    explicit unsafe_serialized_dict(base::memory_buffer data)
        : data_(std::move(data))
    {
    }

    std::string serialize() const noexcept
    {
        return std::string(data_.string_view());
    }

    std::vector<std::string> keys() const
    {
        return icm::const_json::keys(data_.string_view());
    }

    icm::const_json data() const
    {
        return icm::const_json::parse(data_.string_view());
    }

private:
    base::memory_buffer data_;
};

}
