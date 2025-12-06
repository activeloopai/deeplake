#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"

#include "dynamic_array.hpp"
#include "std_span_array.hpp"

#include <base/span_cast.hpp>
#include <icm/shape.hpp>

#include <cstdint>
#include <span>
#include <vector>
#include <iostream>

namespace nd::impl {

class string_vector_array
{
public:
    using value_type = std::vector<std::string>;

    struct holder_t
    {
        explicit holder_t(value_type&& d)
            : data(std::move(d))
            , shape(data.size())
        {}

        value_type data;
        icm::shape shape;
    };

    explicit string_vector_array(value_type&& value)
        : value_(std::make_shared<holder_t>(std::move(value)))
    {}

    enum dtype dtype() const
    {
        return dtype::string;
    }

    uint8_t dimensions() const
    {
        return 1;
    }

    nd::array get(int64_t index) const
    {
        return nd::array(string_array(std::string(value_->data[index])));
    }

    const icm::shape& shape() const
    {
        return value_->shape;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    const std::shared_ptr<holder_t> value_;
};

}
