#pragma once

#include <string>
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#include <stdlib.h>
#endif

namespace base {

namespace detail {

inline char* get_env_impl(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    _dupenv_s(&value, &len, name);
    return value;
#else
    return std::getenv(name);
#endif
}

inline void free_env_impl(char* ptr) {
#ifdef _WIN32
    free(ptr);
#else
    (void)ptr;
#endif
    }
}

template <typename T>
requires std::is_same_v<T, std::string> || std::is_same_v<T, bool> || std::is_integral_v<T>
inline T getenv(const std::string& name, T default_value = T{})
{
    char* value = detail::get_env_impl(name.c_str());
    if (!value) {
        return default_value;
    }

    if constexpr (std::is_same_v<T, std::string>) {
        std::string result(value);
        detail::free_env_impl(value);
        return result;
    } else if constexpr (std::is_same_v<T, bool>) {
        std::string str(value);
        detail::free_env_impl(value);
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        return str == "1" || str == "true" || str == "yes";
    } else {
        static_assert(std::is_integral_v<T>, "Type T must be std::string, bool, or an integral type");
        std::string str(value);
        detail::free_env_impl(value);
        try {
            if constexpr (std::is_signed_v<T>) {
                return static_cast<T>(std::stoll(str));
            } else {
                return static_cast<T>(std::stoull(str));
            }
        } catch (...) {
            return default_value;
        }
    }
}

}
