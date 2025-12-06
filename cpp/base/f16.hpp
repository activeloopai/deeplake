#pragma once

#include "f16_wrapper.hpp"

#include "bit_cast.hpp"
#include "type_traits.hpp"

#include <ostream>

namespace base {

template <typename T>
class f16
{
public:
    using storage_type = detail::impl<T>::value_type;

    constexpr f16() noexcept = default;

    template <arithmetic P>
    constexpr f16(P value) noexcept
        : storage_{value}
    {
    }

    constexpr f16(const storage_type& value) noexcept
        : storage_{value}
    {
    }

    template <arithmetic P>
    constexpr f16& operator=(P value) noexcept
    {
        storage_.value_ = static_cast<storage_type>(value);
        return *this;
    }

    template <arithmetic P>
    friend constexpr f16 operator+(const f16& a, const P& b) noexcept
    {
        f16 result{a};
        return result += b;
    }

    template <arithmetic P>
    friend constexpr f16 operator-(const f16& a, const P& b) noexcept
    {
        f16 result{a};
        return result -= b;
    }

    template <arithmetic P>
    friend constexpr f16 operator*(const f16& a, const P& b) noexcept
    {
        f16 result{a};
        return result *= b;
    }

    template <arithmetic P>
    friend constexpr f16 operator/(const f16& a, const P& b) noexcept
    {
        f16 result{a};
        return result /= b;
    }

    // unary minus operator
    friend constexpr f16 operator-(const f16& a) noexcept
    {
        f16 result{a};
        result.storage_.value_ = -result.storage_.value_;
        return result;
    }

    template <arithmetic P>
    constexpr f16& operator+=(const P& value) noexcept
    {
        storage_.value_ += f16(value).storage_.value_;
        return *this;
    }

    template <arithmetic P>
    constexpr f16& operator-=(const P& value) noexcept
    {
        storage_.value_ -= f16(value).storage_.value_;
        return *this;
    }

    template <arithmetic P>
    constexpr f16& operator*=(const P& value) noexcept
    {
        storage_.value_ *= f16(value).storage_.value_;
        return *this;
    }

    template <arithmetic P>
    constexpr f16& operator/=(const P& value) noexcept
    {
        storage_.value_ /= f16(value).storage_.value_;
        return *this;
    }

    template <arithmetic P>
    constexpr auto operator<=>(const P& b) const noexcept
    {
        return storage_.value_ <=> f16(b).storage_.value_;
    }

    template <arithmetic P>
    constexpr bool operator==(const P& b) const noexcept
    {
        return storage_.value_ == f16(b).storage_.value_;
    }

    // type conversion operators
    template <typename TO_TYPE>
    constexpr explicit operator TO_TYPE() const noexcept
    {
        return static_cast<TO_TYPE>(storage_.value_);
    }

    friend constexpr std::ostream& operator<<(std::ostream& stream, const f16& num)
    {
        stream << num.storage_.value_;
        return stream;
    }

    template <arithmetic P>
    static constexpr bool isnan(const P& x) noexcept
    {
        return detail::math::isnan(static_cast<storage_type>(x));
    }

    template <arithmetic P>
    static constexpr f16 abs(const P& x) noexcept
    {
        return f16(detail::math::abs(static_cast<storage_type>(x)));
    }

    template <arithmetic P>
    static constexpr f16 fabs(const P& x) noexcept
    {
        return abs(x);
    }

    template <arithmetic P>
    static constexpr f16 sqrt(const P& x) noexcept
    {
        return f16(detail::math::sqrt(static_cast<storage_type>(x)));
    }

    static constexpr f16 epsilon() noexcept
    {
        return f16(std::numeric_limits<storage_type>::epsilon());
    }

    static constexpr size_t hash(const f16& x) noexcept
    {
        return std::hash<storage_type>()(x.storage_.value_);
    }

    [[nodiscard]] constexpr uint16_t raw() const noexcept
    {
        return base::bit_cast<uint16_t>(storage_.value_);
    }

private:
    detail::impl<T> storage_;
};

using half = f16<half_selector>;
using bfloat16 = f16<bfloat16_selector>;

static_assert(!std::is_convertible_v<half, float>, "half should not be implicitly convertible to float");
static_assert(std::is_constructible_v<float, half>, "half should be explicitly convertible to float");

static_assert(!std::is_convertible_v<bfloat16, float>, "bfloat16 should not be implicitly convertible to float");
static_assert(std::is_constructible_v<float, bfloat16>, "bfloat16 should be explicitly convertible to float");

} // namespace base

// Standard hash support
template<typename T>
struct std::hash<base::f16<T>>
{
    size_t operator()(const base::f16<T>& arg) const noexcept
    {
        return base::f16<T>::hash(arg);
    }
};

namespace std {

template <base::detail::FP16_T T>
struct numeric_limits<base::f16<T>> : numeric_limits<typename base::f16<T>::storage_type>
{
    static constexpr base::f16<T> quiet_NaN() noexcept
    {
        return numeric_limits<typename base::f16<T>::storage_type>::quiet_NaN();
    }

    static constexpr base::f16<T> signaling_NaN() noexcept
    {
        return numeric_limits<typename base::f16<T>::storage_type>::signaling_NaN();
    }
};

// Also need to specialize c, v and cv versions of the numeric_limits
// https://stackoverflow.com/a/16519653/

template<base::detail::FP16_T T>
struct numeric_limits<const base::f16<T>> : numeric_limits<base::f16<T>> {};
template<base::detail::FP16_T T>
struct numeric_limits<volatile base::f16<T>> : numeric_limits<base::f16<T>> {};
template<base::detail::FP16_T T>
struct numeric_limits<const volatile base::f16<T>> : numeric_limits<base::f16<T>> {};

} // namespace std
