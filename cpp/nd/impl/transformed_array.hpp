#pragma once

#include "../adapt.hpp"
#include "../array.hpp"
#include "chained_iterator.hpp"

#include <icm/small_vector.hpp>

namespace nd::impl {

template <typename F>
requires(std::is_same_v<std::invoke_result_t<F, nd::array>, nd::array>)
class dynamic_transformed_array
{
public:
    dynamic_transformed_array(nd::array arr, F f)
        : array_(std::move(arr))
        , f_(std::move(f))
        , shape_(array_.size())
    {
        try {
            if (!array_.empty()) {
                auto a = f_(array_[0]);
                dtype_ = a.dtype();
                dimensions_ += a.dimensions();
            }
        } catch (const nd::exception&) {
            dtype_ = dtype::unknown;
        }
    }

    enum dtype dtype() const
    {
        return dtype_;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    using iterator = chained_iterator<F, nd::iterator>;

    iterator begin() const
    {
        return iterator(f_, array_.begin());
    }

    iterator end() const
    {
        return iterator(f_, array_.end());
    }

    array get(int64_t index) const
    {
        return f_(array_[index]);
    }

    uint8_t dimensions() const
    {
        return dimensions_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    nd::array array_;
    F f_;
    icm::shape shape_;
    uint8_t dimensions_ = 1;
    enum dtype dtype_ = dtype::unknown;
};

template <typename F>
requires(!std::is_same_v<std::invoke_result_t<F, nd::array>, nd::array>)
class transformed_array
{
public:
    using value_type = std::invoke_result_t<F, nd::array>;

    transformed_array(nd::array arr, F f)
        : array_(std::move(arr))
        , f_(std::move(f))
        , shape_(array_.size())
    {}

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    static auto converter(F op)
    {
        return [op = std::move(op)](nd::array v) {
            return adapt(op(std::move(v)));
        };
    }

    using iterator = chained_iterator<decltype(converter(std::declval<F>())), nd::iterator>;

    iterator begin() const
    {
        return iterator(converter(f_), array_.begin());
    }

    iterator end() const
    {
        return iterator(converter(f_), array_.end());
    }

    value_type value(int64_t index) const
    {
        return f_(array_[index]);
    }

    array get(int64_t index) const
    {
        return adapt(value(index));
    }

    uint8_t dimensions() const
    {
        return 1;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    nd::array array_;
    F f_;
    icm::shape shape_;
};

template <typename F>
requires(std::is_same_v<std::invoke_result_t<F, array, array>, array>)
class dynamic_transformed_array_2
{
public:
    dynamic_transformed_array_2(array a1, array a2, F f)
        : a1_(std::move(a1))
        , a2_(std::move(a2))
        , f_(std::move(f))
        , shape_(a1_.size())
    {
        ASSERT(a1_.size() == a2_.size());
        try {
            if (!a1_.empty()) {
                auto a = f_(a1_[0], a2_[0]);
                dtype_ = a.dtype();
                dimensions_ += a.dimensions();
            }
        } catch (const nd::exception&) {
            dtype_ = dtype::unknown;
        }
    }

    enum dtype dtype() const
    {
        return dtype_;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    using iterator = chained_iterator<F, nd::iterator, nd::iterator>;

    iterator begin() const
    {
        return iterator(f_, a1_.begin(), a2_.begin());
    }

    iterator end() const
    {
        return iterator(f_, a1_.end(), a2_.end());
    }

    array get(int64_t index) const
    {
        return f_(a1_[index], a2_[index]);
    }

    uint8_t dimensions() const
    {
        return dimensions_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    array a1_;
    array a2_;
    F f_;
    icm::shape shape_;
    uint8_t dimensions_ = 1;
    enum dtype dtype_ = dtype::unknown;
};

template <typename F>
requires(!std::is_same_v<std::invoke_result_t<F, array, array>, array>)
class transformed_array_2
{
public:
    using value_type = std::invoke_result_t<F, array, array>;

    transformed_array_2(array a1, array a2, F f)
        : a1_(std::move(a1))
        , a2_(std::move(a2))
        , f_(std::move(f))
        , shape_(a1_.size())
    {
        ASSERT(a1_.size() == a2_.size());
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    static auto converter(F op)
    {
        return [op = std::move(op)](nd::array a, nd::array b) {
            return adapt(op(std::move(a), std::move(b)));
        };
    }

    using iterator = chained_iterator<decltype(converter(std::declval<F>())), nd::iterator, nd::iterator>;

    iterator begin() const
    {
        return iterator(converter(f_), a1_.begin(), a2_.begin());
    }

    iterator end() const
    {
        return iterator(converter(f_), a1_.end(), a2_.end());
    }

    value_type value(int64_t index) const
    {
        return f_(a1_[index], a2_[index]);
    }

    array get(int64_t index) const
    {
        return adapt(value(index));
    }

    uint8_t dimensions() const
    {
        return 1;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    array a1_;
    array a2_;
    F f_;
    icm::shape shape_;
};

template <typename F>
array create_transformed_array(array arr, F f)
{
    if constexpr (std::is_same_v<std::invoke_result_t<F, array>, array>) {
        return array(dynamic_transformed_array<F>(std::move(arr), std::move(f)));
    } else {
        return array(transformed_array<F>(std::move(arr), std::move(f)));
    }
}

template <typename F>
array create_transformed_array(array a1, array a2, F f)
{
    if constexpr (std::is_same_v<std::invoke_result_t<F, array, array>, array>) {
        return array(dynamic_transformed_array_2<F>(std::move(a1), std::move(a2), std::move(f)));
    } else {
        return array(transformed_array_2<F>(std::move(a1), std::move(a2), std::move(f)));
    }
}

}
