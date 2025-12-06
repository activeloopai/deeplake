#pragma once

#include <base/assert.hpp>

#include <rapidjson/document.h>

#include <array>
#include <concepts>
#include <functional>

namespace icm::impl {

template <typename T>
concept is_vector = requires { typename T::value_type; } && std::same_as<T, std::vector<typename T::value_type>>;

template <typename T>
concept is_std_array = requires {
    typename T::value_type;
    std::tuple_size<T>::value;
} && std::is_same_v<T, std::array<typename T::value_type, std::tuple_size<T>::value>>;

template <typename C>
inline bool compare_bool_num(const rapidjson::Value& lhs, const rapidjson::Value& rhs, C cmp) noexcept
{
    ASSERT(lhs.IsBool() && rhs.IsNumber());
    if (rhs.IsInt()) {
        return cmp(static_cast<int32_t>(lhs.IsTrue()), rhs.GetInt());
    } else if (rhs.IsUint()) {
        return cmp(static_cast<uint32_t>(lhs.IsTrue()), rhs.GetUint());
    } else if (rhs.IsInt64()) {
        return cmp(static_cast<int64_t>(lhs.IsTrue()), rhs.GetInt64());
    } else if (rhs.IsUint64()) {
        return cmp(static_cast<uint64_t>(lhs.IsTrue()), rhs.GetUint64());
    }
    return cmp(static_cast<double>(lhs.IsTrue()), rhs.GetDouble());
}

inline bool bool_number_equal(const rapidjson::Value& lhs, const rapidjson::Value& rhs)
{
    if (rhs.IsBool() && lhs.IsNumber()) {
        return bool_number_equal(rhs, lhs);
    }
    ASSERT(lhs.IsBool() && rhs.IsNumber());
    return compare_bool_num(lhs, rhs, std::equal_to{});
}

inline bool bool_number_less_than(const rapidjson::Value& lhs, const rapidjson::Value& rhs)
{
    if (rhs.IsBool() && lhs.IsNumber()) {
        if (bool_number_equal(rhs, lhs)) {
            return false;
        }
        return !bool_number_less_than(rhs, lhs);
    }
    ASSERT(lhs.IsBool() && rhs.IsNumber());
    return compare_bool_num(lhs, rhs, std::less{});
}

inline bool less_than(const rapidjson::Value& lhs, const rapidjson::Value& rhs)
{
    if (lhs.GetType() != rhs.GetType()) {
        if ((lhs.IsBool() && rhs.IsNumber()) || (lhs.IsNumber() && rhs.IsBool())) {
            return bool_number_less_than(lhs, rhs);
        } else {
            return lhs.GetType() < rhs.GetType();
        }
    }

    switch (lhs.GetType()) {
        case rapidjson::kNullType: {
            return false;
        }
        case rapidjson::kFalseType:
        case rapidjson::kTrueType: {
            return lhs.GetBool() < rhs.GetBool();
        }
        case rapidjson::kNumberType: {
            if (lhs.IsInt() && rhs.IsInt()) {
                return lhs.GetInt() < rhs.GetInt();
            }
            if (lhs.IsUint() && rhs.IsUint()) {
                return lhs.GetUint() < rhs.GetUint();
            }
            if (lhs.IsInt64() && rhs.IsInt64()) {
                return lhs.GetInt64() < rhs.GetInt64();
            }
            if (lhs.IsUint64() && rhs.IsUint64()) {
                return lhs.GetUint64() < rhs.GetUint64();
            }
            return (lhs.GetDouble() < rhs.GetDouble());
        }
        case rapidjson::kStringType: {
            return (std::string(lhs.GetString()) < std::string(rhs.GetString()));
        }
        case rapidjson::kArrayType: {
            if (lhs.Size() != rhs.Size()) {
                return lhs.Size() < rhs.Size(); // Shorter array is "smaller"
            }
            for (auto i = 0; i < std::min(lhs.Size(), rhs.Size()); ++i) {
                if (less_than(lhs[i], rhs[i])) {
                    return true;
                }
                if (less_than(rhs[i], lhs[i])) {
                    return false;
                }
            }
            return false;
        }
        case rapidjson::kObjectType: {
            // Compare objects lexicographically by key-value pairs
            if (lhs.MemberCount() != rhs.MemberCount()) {
                return (lhs.MemberCount() < rhs.MemberCount());
            }
            auto lit = lhs.MemberBegin();
            auto rit = rhs.MemberBegin();
            while (lit != lhs.MemberEnd() && rit != rhs.MemberEnd()) {
                // Compare keys first
                if (less_than(lit->name, rit->name)) {
                    return true;
                }
                if (less_than(rit->name, lit->name)) {
                    return false;
                }
                // If keys are equal, compare values
                if (less_than(lit->value, rit->value)) {
                    return true;
                }
                if (less_than(rit->value, lit->value)) {
                    return false;
                }
                ++lit;
                ++rit;
            }
            return false; // If all keys and values are equal, then lhs is not < rhs
        }
    }
    return false; // Should never reach here
}

struct key_collector
{
    std::vector<std::string> keys;
    uint32_t nest_level = 0; // Track nesting level

    bool Key(const char* str, size_t length, bool copy)
    {
        // Only collect keys at top level
        if (nest_level == 1) {
            keys.emplace_back(str, length);
        }
        return true;
    }

    bool StartObject()
    {
        ++nest_level;
        return true;
    }

    bool EndObject()
    {
        --nest_level;
        return true;
    }

    bool StartObject(rapidjson::SizeType)
    {
        ++nest_level;
        return true;
    }

    bool EndObject(rapidjson::SizeType)
    {
        --nest_level;
        return true;
    }

    bool RawNumber(const char* str, rapidjson::SizeType length, bool copy) const noexcept
    {
        return (nest_level != 0);
    }

    bool Null() const noexcept
    {
        return (nest_level != 0);
    }

    bool Bool(bool) const noexcept
    {
        return (nest_level != 0);
    }

    bool Int(int32_t) const noexcept
    {
        return (nest_level != 0);
    }

    bool Uint(uint32_t) const noexcept
    {
        return (nest_level != 0);
    }

    bool Int64(int64_t) const noexcept
    {
        return (nest_level != 0);
    }

    bool Uint64(uint64_t) const noexcept
    {
        return (nest_level != 0);
    }

    bool Double(double) const noexcept
    {
        return (nest_level != 0);
    }

    bool String(const char*, size_t, bool) const noexcept
    {
        return (nest_level != 0);
    }

    bool StartArray() const noexcept
    {
        return (nest_level != 0);
    }

    bool EndArray() const noexcept
    {
        return (nest_level != 0);
    }

    bool StartArray(rapidjson::SizeType) const noexcept
    {
        return (nest_level != 0);
    }

    bool EndArray(rapidjson::SizeType) const noexcept
    {
        return (nest_level != 0);
    }
};

} // namespace icm::impl

inline bool operator<(const rapidjson::Value& lhs, const rapidjson::Value& rhs)
{
    return icm::impl::less_than(lhs, rhs);
}
