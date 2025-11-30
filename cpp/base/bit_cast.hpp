#pragma once

/**
 * @file bit_cast.hpp
 * @brief Definition and implementation of the bit_cast function.
 */

#include "type_traits.hpp"

#if __has_include(<bit>)
#include <bit>
#endif

#include <cstring>

namespace base {

/**
 * @brief bit_cast is C++20 `std::bit_cast` function, which is not available in some old compilers.
 *
 * @tparam To Input type
 * @tparam From Output type
 * @param src Source
 * @return To Result object.
 */
template <class To, class From>
requires(sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>)
To bit_cast(const From& src) noexcept
{
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<To>(src);
#else
    if constexpr (!arithmetic<To>) {
        static_assert(std::is_trivially_constructible_v<To>,
                    "This implementation additionally requires "
                    "destination type to be trivially constructible");
    }
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
#endif
}

} // namespace base
