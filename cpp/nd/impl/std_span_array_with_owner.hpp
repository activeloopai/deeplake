#pragma once

#include "../array.hpp"
#include "../dtype.hpp"

#include <base/span_cast.hpp>
#include <icm/shape.hpp>
#include <icm/small_vector.hpp>

#include <cstdint>
#include <span>

namespace nd {

namespace impl {

template <typename O, typename T>
class std_span_array_span_shape_owner_array
{
public:
    std_span_array_span_shape_owner_array(std_span_array_span_shape_owner_array&&) noexcept = default;
    std_span_array_span_shape_owner_array& operator=(std_span_array_span_shape_owner_array&&) noexcept = default;
    std_span_array_span_shape_owner_array(const std_span_array_span_shape_owner_array&) noexcept = default;
    std_span_array_span_shape_owner_array(O owner, std::span<T> value, icm::shape shape)
        : owner_(owner), value_(value), shape_(shape)
    {
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(value_);
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    const O& owner() const
    {
        return owner_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const O owner_;
    const std::span<T> value_;
    const icm::shape shape_;
};

} // namespace impl

} // namespace nd
