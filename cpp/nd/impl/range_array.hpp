#pragma once

#include <icm/shape.hpp>
#include <icm/small_vector.hpp>

#include <cstdint>

namespace nd::impl {

template <typename T>
class range_array
{
public:
    range_array(T arr, int64_t start, int64_t stop)
        : arr_(std::move(arr))
        , start_(start)
        , stop_(stop)
    {
    }

    auto dtype() const noexcept
    {
        return arr_.dtype();
    }

    icm::shape shape() const noexcept
    {
        auto sh = arr_.shape();
        icm::small_vector<int64_t> r(sh.begin(), sh.end());
        r[0] = stop_ - start_;
        return icm::shape(std::move(r));
    }

    auto begin() const
    {
        auto it = arr_.begin();
        it += start_;
        return it;
    }

    auto end() const
    {
        auto it = arr_.begin();
        it += stop_;
        return it;
    }

    auto get(int64_t index) const
    {
        return arr_[start_ + index];
    }

    constexpr bool is_dynamic() const noexcept
    {
        return arr_.is_dynamic();
    }

    uint8_t dimensions() const noexcept
    {
        return arr_.dimensions();
    }

    auto& holder_array() const noexcept
    {
        return arr_;
    }

    int64_t start() const noexcept
    {
        return start_;
    }

    int64_t stop() const noexcept
    {
        return stop_;
    }

private:
    T arr_;
    int64_t start_;
    int64_t stop_;
};

} // namespace nd::impl
