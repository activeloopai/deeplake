#pragma once

#include "type_traits.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <concepts>

#include <boost/container/vector.hpp>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#include <windows.h>
#endif

namespace base {

/**
 * @brief Initializes the random engine.
 */
void initialize_random_engine();

/**
 * @brief Deinitializes the random engine.
 */
void deinitialize_random_engine();

std::mt19937& random_engine();

/**
 * @brief Generates a random array of a given type and size.
 * @tparam T Type of the array.
 * @tparam size Size of the array.
 * @return A random array of the given type and size.
 */
template <typename T, int size>
inline std::array<T, size> random_array()
{
    std::array<T, size> ret;
    auto& mersenne_engine = random_engine();
    if constexpr (base::is_floating_point_v<T>) {
        using actual_type = std::conditional_t<std::floating_point<T>, T, float>;
        std::uniform_real_distribution<actual_type> value_dist{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
    } else if constexpr (std::is_same_v<bool, T>) {
        std::bernoulli_distribution value_dist(0.5);
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
    } else if constexpr (sizeof(T) == 1) {
        // Handle 8-bit types separately
        using dist_type = std::conditional_t<std::is_signed_v<T>, int, unsigned int>;
        std::uniform_int_distribution<unsigned int> value_dist{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return static_cast<T>(value_dist(mersenne_engine));
        };
        std::generate(ret.begin(), ret.end(), value_gen);
    } else {
        std::uniform_int_distribution<T> value_dist{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
    }
    return ret;
}

/**
 * @brief Generates a random vector of a given type, size, and range.
 * @tparam T Type of the vector.
 * @param size Size of the vector.
 * @param range Range of values to generate.
 * @return A random vector of the given type and size.
 */
template <typename T>
inline auto random_vector(uint64_t size, std::pair<T, T> range)
{
    auto& mersenne_engine = random_engine();
    if constexpr (base::is_floating_point_v<T>) {
        std::vector<T> ret;
        ret.resize(size);
        using actual_type = std::conditional_t<std::floating_point<T>, T, float>;
        std::uniform_real_distribution<actual_type> value_dist{static_cast<actual_type>(range.first), static_cast<actual_type>(range.second)};
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
        return ret;
    } else if constexpr (std::is_same_v<bool, T>) {
        boost::container::vector<T> ret;
        ret.resize(size);
        std::bernoulli_distribution value_dist(0.5);
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
        return ret;
    } else if constexpr (sizeof(T) == 1) {
        std::vector<T> ret;
        ret.resize(size);
        // Handle 8-bit types separately
        using dist_type = std::conditional_t<std::is_signed_v<T>, int, unsigned int>;
        std::uniform_int_distribution<dist_type> value_dist{range.first, range.second};
        auto value_gen = [&value_dist, &mersenne_engine]() {
            return static_cast<T>(value_dist(mersenne_engine));
        };
        std::generate(ret.begin(), ret.end(), value_gen);
        return ret;
    } else {
        std::vector<T> ret;
        ret.resize(size);
        std::uniform_int_distribution<T> value_dist{range.first, range.second};

        auto value_gen = [&value_dist, &mersenne_engine]() {
            return value_dist(mersenne_engine);
        };
        std::generate(ret.begin(), ret.end(), value_gen);
        return ret;
    }
}

/**
 * @brief Generates a random vector of a given type and size.
 * @tparam T Type of the vector.
 * @param size Size of the vector.
 * @return A random vector of the given type and size.
 */
template <typename T>
inline auto random_vector(int size)
{
    return random_vector<T>(size, std::make_pair<T, T>((std::numeric_limits<T>::min)(), (std::numeric_limits<T>::max)()));
}

/**
 * @brief Generates a random number of a given type.
 * @tparam T Type of the number.
 * @return A random number of the given type.
 */
template <typename T>
inline T random_number()
{
    return random_array<T, 1>()[0];
}

/**
 * @brief Generates a random number of a given type within a given range.
 * @tparam T Type of the number.
 * @param range Range of values to generate from.
 * @return A random number of the given type and range.
 */
template <typename T>
inline T random_number(std::pair<T, T> range)
{
    return random_vector<T>(1, range)[0];
}

/**
 * @brief Generates a random string of a given size from a given set of characters.
 * @param size Size of the string to generate.
 * @param chrs Set of characters to use for the string.
 * @return A random string of the given size.
 */
inline std::string random_string(int size,
                                 const std::string_view chrs = "0123456789"
                                                               "abcdefghijklmnopqrstuvwxyz"
                                                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
{
    auto& rg = random_engine();
    std::uniform_int_distribution<std::string::size_type> pick(0, chrs.size()-1);
    std::string s;
    s.reserve(size);
    while(size--) {
        s += chrs[pick(rg)];
    }
    return s;
}

/**
 * @brief Uses the random_string function to generate a random uuid-formatted string. Not a true uuid, but random enough
 * to act as one.
 * @return A random uuid-formatted string.
 */
inline std::string random_uuid()
{
    static const std::string_view uuid_chars = "0123456789abcdef";
    return random_string(32, uuid_chars);
}

}
