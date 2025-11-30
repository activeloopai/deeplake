#pragma once

#include <cstdint>
#include <type_traits>

namespace icm::mpl {

template <typename T, typename... Ts>
struct is_one_of
{
    static constexpr bool value = (std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<Ts>> || ...);
};

template <typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template <typename T>
inline constexpr bool is_numeric_integral_v =
    is_one_of_v<T, uint8_t, int8_t, int16_t, uint16_t, int32_t /*same as 'int'*/, uint32_t, int64_t, uint64_t,
                long, long long, unsigned long long, unsigned long>; // for emscripten  std::is_same_v<uint64_t, unsigned long long> == false

} // namespace icm::mpl
