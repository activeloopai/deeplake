#pragma once

#include "../array.hpp"

#include <icm/const_json.hpp>

namespace nd::impl {

struct const_json_dict
{
    explicit const_json_dict(icm::const_json&& data)
        : data_(std::move(data))
    {
    }

    icm::const_json data() const noexcept
    {
        return data_;
    }

    dict eval() const
    {
        return dict(data_);
    }

private:
    icm::const_json data_;
};

}
