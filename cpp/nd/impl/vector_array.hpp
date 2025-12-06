#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"

#include <base/memory_buffer.hpp>
#include <icm/shape.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace nd {

namespace impl {

template <typename T>
class vector_array
{
public:
    using value_type = typename T::value_type;

    struct holder_t
    {
        explicit holder_t(T&& d)
            : data(std::move(d))
            , shape(data.size())
        {
        }

        T data;
        icm::shape shape;
    };

public:
    explicit vector_array(T&& value)
        : value_(std::make_shared<holder_t>(std::move(value)))
    {
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(
            std::span<const typename T::value_type>(value_->data.data(), value_->data.size()));
    }

    const auto& owner() const
    {
        return value_;
    }

    const icm::shape& shape() const
    {
        return value_->shape;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const std::shared_ptr<holder_t> value_;
};

template <typename T>
class vector_array_with_shape
{
public:
    using value_type = typename T::value_type;

    struct holder_t
    {
        holder_t(T&& data, icm::shape&& sh)
            : data(std::move(data))
            , shape(std::move(sh))
        {
        }

        T data;
        icm::shape shape;
    };

    vector_array_with_shape(T&& value, icm::shape&& sh)
        : value_(std::make_shared<holder_t>(std::move(value), std::move(sh)))
    {
    }

public:
    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(
            std::span<const typename T::value_type>(value_->data.data(), value_->data.size()));
    }

    const auto& owner() const
    {
        return value_;
    }

    const icm::shape& shape() const
    {
        return value_->shape;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const std::shared_ptr<holder_t> value_;
};

class buffer_array
{
public:
    buffer_array(base::memory_buffer data, enum dtype dtype)
        : data_(std::move(data))
        , dtype_(dtype)
    {
    }

public:
    enum dtype dtype() const
    {
        return dtype_;
    }

    std::span<const uint8_t> data() const
    {
        return data_.span();
    }

    const auto& owner() const
    {
        return data_.owner();
    }

    icm::shape shape() const
    {
        return icm::shape(data_.size() / dtype_bytes(dtype_));
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    base::memory_buffer data_;
    enum dtype dtype_;
};

class string_array
{
public:
    struct holder_t
    {
        explicit holder_t(std::string&& d)
            : data(std::move(d))
        {
        }

        std::string data;
    };

public:
    explicit string_array(std::string&& value)
        : value_(std::make_shared<holder_t>(std::move(value)))
    {
    }

public:
    enum dtype dtype() const
    {
        return dtype::string;
    }

    std::span<const uint8_t> data() const
    {
        return base::span_cast<const uint8_t>(base::span_cast(std::string_view(value_->data)));
    }

    const auto& owner() const
    {
        return value_;
    }

    icm::shape shape() const
    {
        return icm::shape();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    std::shared_ptr<holder_t> value_;
};

} // namespace impl

} // namespace nd
