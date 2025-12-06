#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace nd {

namespace impl {

template <uint32_t size>
icm::shape shape(size);

template <typename T, std::size_t size>
class std_array_array
{
public:
    explicit std_array_array(std::array<T, size> value)
        : value_(std::make_shared<std::array<T, size>>(value))
    {}

    std_array_array(const std_array_array&) = default;
    std_array_array& operator=(const std_array_array&) = default;
    std_array_array(std_array_array&&) noexcept = default;
    std_array_array& operator=(std_array_array&&) noexcept = default;
    ~std_array_array() = default;

public:
    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(std::span<const T>(value_->data(), size));
    }

    const auto& owner() const
    {
        return value_;
    }

    icm::shape shape() const
    {
        return ::nd::impl::shape<size>;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    std::shared_ptr<std::array<T, size>> value_;
};

}

}
