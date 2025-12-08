#pragma once

#include "../array.hpp"
#include "scalar_array.hpp"
#include "../exceptions.hpp"

#include <cstdint>
#include <span>

namespace nd::impl {

template <typename T>
class empty_array
{
public:
    using value_type = T;

public:
    explicit empty_array(const icm::shape& shape)
        : shape_(shape)
    {
    }

    empty_array(const empty_array&) = default;
    empty_array& operator=(const empty_array&) = delete;
    empty_array(empty_array&&) noexcept = default;
    empty_array& operator=(empty_array&&) = delete;
    ~empty_array() = default;

public:
    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    value_type value(int64_t) const
    {
        return value_type();
    }

    array get(int64_t) const
    {
        if (shape_.size() == 1) {
            if constexpr (base::arithmetic<value_type>) {
                return array(value_type());
            } else if constexpr (std::is_same_v<value_type, nd::dict>) {
                return adapt(nd::dict());
            } else if constexpr (std::is_same_v<value_type, std::string_view>) {
                return adapt(std::string());
            } else {
                ASSERT(false);
            }
        }
        return array(empty_array<value_type>(icm::shape(shape_.begin() + 1, shape_.end())));
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    icm::shape shape_;
};

class empty_dynamic_array
{
public:
    explicit empty_dynamic_array(enum dtype dt, uint32_t size)
        : size_(size)
        , dtype_(dt)
    {}

    empty_dynamic_array(const empty_dynamic_array&) = default;
    empty_dynamic_array& operator=(const empty_dynamic_array&) = delete;
    empty_dynamic_array(empty_dynamic_array&&) noexcept = default;
    empty_dynamic_array& operator=(empty_dynamic_array&&) = delete;
    ~empty_dynamic_array() = default;

public:
    enum dtype dtype() const
    {
        return dtype_;
    }

    icm::shape shape() const
    {
        return icm::shape(size_);
    }

    array get(int64_t) const
    {
        return nd::none(dtype_, 0);
    }

    uint8_t dimensions() const
    {
        return 1;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    int32_t size_;
    enum dtype dtype_;
};

template <typename S>
class empty_dynamic_multidimensional_array
{
public:
    explicit empty_dynamic_multidimensional_array(enum dtype dt, S shapes)
        : shapes_(std::make_shared<S>(std::move(shapes)))
        , size_(shapes_->size())
        , dtype_(dt)
    {}

    empty_dynamic_multidimensional_array(const empty_dynamic_multidimensional_array&) = default;
    empty_dynamic_multidimensional_array& operator=(const empty_dynamic_multidimensional_array&) = delete;
    empty_dynamic_multidimensional_array(empty_dynamic_multidimensional_array&&) noexcept = default;
    empty_dynamic_multidimensional_array& operator=(empty_dynamic_multidimensional_array&&) = delete;
    ~empty_dynamic_multidimensional_array() = default;

public:
    enum dtype dtype() const
    {
        return dtype_;
    }

    icm::shape shape() const
    {
        return icm::shape(size_);
    }

    array get(int64_t index) const
    {
        return nd::empty(dtype_, (*shapes_)[index]);
    }

    uint8_t dimensions() const
    {
        return 1 + (*shapes_)[0].size();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<S> shapes_;
    int32_t size_;
    enum dtype dtype_;
};

}
