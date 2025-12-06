#pragma once

#include <base/assert.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>
#include <nd/none.hpp>

namespace heimdall_common::impl {

template <bool is_dynamic_>
struct padded_array
{
    padded_array(nd::array source, int64_t pad_size)
        : source_(std::move(source))
        , shape_(source_.shape()[0] + pad_size)
    {
    }

    padded_array(const padded_array&) = default;
    padded_array& operator=(const padded_array&) = delete;
    padded_array(padded_array&& s) noexcept = default;
    padded_array& operator=(padded_array&&) noexcept = default;
    ~padded_array() = default;

    inline nd::dtype dtype() const
    {
        return source_.dtype();
    }

    inline icm::shape shape() const
    {
        return shape_;
    }

    inline nd::array get(int64_t index) const
    {
        if (index < source_.shape()[0]) {
            return source_[index];
        }
        return nd::none(source_.dtype(), dimensions() - 1);
    }

    inline uint8_t dimensions() const
    {
        return static_cast<uint8_t>(source_.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return is_dynamic_;
    }

private:
    nd::array source_;
    icm::shape shape_;
};
} // namespace heimdall_common::impl
