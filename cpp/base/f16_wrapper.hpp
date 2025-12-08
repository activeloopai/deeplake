#pragma once

#include <Eigen/Core>

namespace base {

struct half_selector{};
struct bfloat16_selector{};

namespace detail {

namespace math = Eigen::numext;

template<typename T>
concept FP16_T = std::same_as<T, half_selector> || std::same_as<T, bfloat16_selector>;

template <typename T>
struct impl;

template <FP16_T T>
struct impl<T>
{
    // As of 2025, https://sourceforge.net/projects/half/ (half_float::half) has better performance than Eigen::half
    using value_type = std::conditional_t<std::is_same_v<T, half_selector>, Eigen::half, Eigen::bfloat16>;

    constexpr impl() noexcept = default;

    template<typename P>
    explicit constexpr impl(P val) noexcept
        : value_{val}
    {
    }

    value_type value_;
};

} // namespace detail

} // namespace base
