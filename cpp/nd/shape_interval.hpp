#pragma once

/**
 * @file shape_interval.hpp
 * @brief Definition and implementation of `shape_interval_t` template class.
 */

#include <icm/small_vector.hpp>

#include <cstdint>
#include <optional>

namespace nd {

template <typename T>
class shape_interval_t {
public:
    inline explicit shape_interval_t(icm::small_vector<T> shape)
        : min_(shape)
    {}

    inline shape_interval_t(icm::small_vector<T> min, icm::small_vector<T> max)
        : min_(std::move(min))
        , max_(std::move(max))
    {}

    shape_interval_t(const shape_interval_t&) = default;
    shape_interval_t& operator=(const shape_interval_t&) = default;
    shape_interval_t(shape_interval_t&&) noexcept = default;
    shape_interval_t& operator=(shape_interval_t&&) noexcept = default;
    ~shape_interval_t() = default;

public:
    constexpr bool is_dynamic() const noexcept
    {
        return !static_cast<bool>(max_);
    }

    const auto& min() const noexcept
    {
        return min_;
    }

    const auto& max() const noexcept
    {
        if (max_) {
            return *max_;
        }
        return min_;
    }

private:
    icm::small_vector<T> min_;
    std::optional<icm::small_vector<T>> max_;
};

}
