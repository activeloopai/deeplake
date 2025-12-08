#pragma once

#include "../exceptions.hpp"

namespace nd::impl {

class none_array
{
public:
    explicit none_array(uint8_t dims, enum dtype dt)
        : dims_(dims)
        , dtype_(dt)
    {}

public:
    enum dtype dtype() const
    {
        return dtype_;
    }

    icm::shape shape() const
    {
        return icm::shape(std::vector<int64_t>(dims_, 0));
    }

    std::span<const uint8_t> data() const
    {
        return {};
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

    static constexpr bool is_none = true;

private:
    uint8_t dims_;
    enum dtype dtype_;
};

}
