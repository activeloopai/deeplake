#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"

#include <base/span_cast.hpp>

#include <cstdint>
#include <span>

namespace nd::impl {

class scalar_array
{
public:
    explicit scalar_array(dict value) : value_(std::move(value))
    {
    }

public:
    enum dtype dtype() const
    {
        return dtype::object;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(std::span<const dict>(&value_, 1));
    }

    icm::shape shape() const
    {
        return icm::shape();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

    static constexpr bool is_scalar_tag = true;

private:
    dict value_;
};

} // namespace nd::impl
