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
requires std::is_same_v<T, std::string> || std::is_same_v<T, bool>
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
    } else {
        static_assert(std::is_same_v<T, bool>, "Type T must be either std::string or bool");
        std::string str(value);
        detail::free_env_impl(value);
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        return str == "1" || str == "true" || str == "yes";
    }
}

}
