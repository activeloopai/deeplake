#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"

#include <base/span_cast.hpp>
#include <icm/small_vector.hpp>

#include <cstdint>
#include <span>

namespace nd {

namespace impl {

class std_span_array_nd
{
public:
    std_span_array_nd(std::shared_ptr<void> owner, std::span<const uint8_t> data, icm::shape shape, dtype dt)
        : owner_(std::move(owner))
        , data_(data)
        , shape_(std::move(shape))
        , dtype_(dt)
    {
    }

    inline enum dtype dtype() const
    {
        return dtype_;
    }

    inline std::span<const uint8_t> data() const
    {
        return data_;
    }

    inline const std::shared_ptr<void>& owner() const
    {
        return owner_;
    }

    inline const icm::shape& shape() const
    {
        return shape_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const std::shared_ptr<void> owner_;
    const std::span<const uint8_t> data_;
    const icm::shape shape_;
    const enum dtype dtype_;
};

} // namespace impl

} // namespace nd
