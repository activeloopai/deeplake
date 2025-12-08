#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../type.hpp"

#include <base/memory_buffer.hpp>
#include <icm/shape.hpp>

#include <icm/const_json.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace nd::impl {

class json_array
{
public:
    struct holder_t
    {
        explicit holder_t(const icm::const_json& d)
            : data(d)
        {
            ASSERT(data.is_array());
        }

        icm::const_json data;
    };

public:
    explicit json_array(const icm::const_json& value)
        : value_(std::make_shared<holder_t>(value))
    {
    }

public:
    enum dtype dtype() const
    {
        return dtype::object;
    }

    nd::array get(int64_t index) const
    {
        return adapt(value_->data.at(index));
    }

    icm::shape shape() const
    {
        return icm::shape(value_->data.size());
    }

    uint8_t dimensions() const
    {
        return nd::type::unknown_dimensions;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<holder_t> value_;
};

}
