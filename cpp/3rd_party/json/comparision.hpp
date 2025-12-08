#pragma once

#include <boost/json/value.hpp>
#include <boost/json/value_from.hpp>
#include <boost/json/kind.hpp>

#include <array>

// order: null < boolean < number < object < array < string < binary
static constexpr std::array<std::uint8_t, 9> json_kind_order_ = {{
    0, // null
    1, // boolean
    2, // number
    2, // unsigned
    2, // float
    5, // string
    4, // array
    3, // object
    }
};

namespace boost {

namespace json {

// Less than operators
template<typename T>
inline bool operator<(const boost::json::value& lhs, const T& rhs)
{
    return lhs < boost::json::value_from(rhs);
}

template<typename T>
inline bool operator<(const T& lhs, const boost::json::value& rhs)
{
    return boost::json::value_from(lhs) < rhs;
}

inline bool operator<(const boost::json::value& lhs, const boost::json::value& rhs)
{
    auto lkind_idx = static_cast<std::size_t>(lhs.kind());
    auto rkind_idx = static_cast<std::size_t>(rhs.kind());
    if (lkind_idx == rkind_idx || (lhs.is_number() && rhs.is_number())) {
        if (lhs.is_null()) {
            return false;
        } else if (lhs.is_bool()) {
            return lhs.as_bool() < rhs.is_bool();
        } else if (lhs.is_string()) {
            return lhs.as_string() < rhs.as_string();
        } else if (lhs.is_number()) {
            if (lhs.is_int64() && rhs.is_int64()) {
                return lhs.as_int64() < rhs.as_int64();
            } else if (lhs.is_double() && rhs.is_double()) {
                return lhs.as_double() < rhs.as_double();
            } else if (lhs.is_int64() && rhs.is_double()) {
                return static_cast<double>(lhs.as_int64()) < rhs.as_double();
            } else if (lhs.is_double() && rhs.is_int64()) {
                return lhs.as_double() < static_cast<double>(rhs.as_int64());
            } else if (lhs.is_int64() && rhs.is_uint64()) {
                return lhs.as_int64() < static_cast<int64_t>(rhs.as_uint64());
            } else if (lhs.is_uint64() && rhs.is_int64()) {
                return static_cast<int64_t>(lhs.as_uint64()) < rhs.as_int64();
            } else if (lhs.is_uint64() && rhs.is_double()) {
                return static_cast<double>(lhs.as_uint64()) < rhs.as_double();
            } else if (lhs.is_double() && rhs.is_uint64()) {
                return lhs.as_double() < static_cast<double>(rhs.as_uint64());
            } else if (lhs.is_uint64() && rhs.is_uint64()) {
                return lhs.as_uint64() < rhs.as_uint64();
            }
        }
    }
    return lkind_idx < json_kind_order_.size() && rkind_idx < json_kind_order_.size() && json_kind_order_[lkind_idx] < json_kind_order_[rkind_idx];
}

// Greater than operators
template<typename T>
inline bool operator>(const boost::json::value& lhs, const T& rhs)
{
    return rhs < lhs;
}

template<typename T>
inline bool operator>(const T& lhs, const boost::json::value& rhs)
{
    return rhs < lhs;
}

inline bool operator>(const boost::json::value& lhs, const boost::json::value& rhs)
{
    return rhs < lhs;
}

// Less than or equal operators
template<typename T>
inline bool operator<=(const boost::json::value& lhs, const T& rhs)
{
    return !(rhs < lhs);
}

template<typename T>
inline bool operator<=(const T& lhs, const boost::json::value& rhs)
{
    return !(rhs < lhs);
}

inline bool operator<=(const boost::json::value& lhs, const boost::json::value& rhs)
{
    return !(rhs < lhs);
}

// Greater than or equal operators
template<typename T>
inline bool operator>=(const boost::json::value& lhs, const T& rhs)
{
    return !(lhs < rhs);
}

template<typename T>
inline bool operator>=(const T& lhs, const boost::json::value& rhs)
{
    return !(lhs < rhs);
}

inline bool operator>=(const boost::json::value& lhs, const boost::json::value& rhs)
{
    return (lhs > rhs) || (lhs == rhs);
}

// Equality operators
template<typename T>
inline bool operator==(const boost::json::value& lhs, const T& rhs)
{
    return lhs == boost::json::value_from(rhs);
}

template<typename T>
inline bool operator==(const T& lhs, const boost::json::value& rhs)
{
    return rhs == lhs;
}

// Inequality operators
template<typename T>
inline bool operator!=(const T& lhs, const boost::json::value& rhs)
{
    return boost::json::value_from(lhs) != rhs;
}

template<typename T>
inline bool operator!=(const boost::json::value& lhs, const T& rhs)
{
    return rhs != lhs;
}

}
    
} // namespace boost
