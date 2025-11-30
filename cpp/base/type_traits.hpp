#pragma once

#include <concepts>
#include <type_traits>

namespace base {

template <typename T>
class f16;

// floating point

template <typename T>
struct is_floating_point : std::is_floating_point<T> {};

template <typename T>
struct is_floating_point<f16<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

// integral

template <typename T>
struct is_integral_impl : std::is_integral<T> {};

template <typename T>
using is_integral = is_integral_impl<T>;

template <typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

// concepts

template <typename T>
concept floating_point = is_floating_point_v<T>;

template <typename T>
concept integral = is_integral_v<T>;

// concept only for arithmetic to avoid defining bulky __or_ template
template <typename T>
concept arithmetic = integral<T> || floating_point<T>;

} // namespace base
