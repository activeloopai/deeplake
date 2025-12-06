#pragma once

#include "index_mapping.hpp"
#include "small_vector.hpp"
#include "parse_negative_index.hpp"

#include <base/assert.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <type_traits>

namespace icm {

template <typename T>
class slice_t
{
    static_assert(std::is_arithmetic_v<T>);

public:
    using type = T;

    constexpr slice_t(std::optional<T> start, std::optional<T> stop, std::optional<T> step) noexcept
        : start_(start), stop_(stop), step_(step ? step.value() : static_cast<T>(1))
    {
    }

    constexpr slice_t(const slice_t&) noexcept = default;
    constexpr slice_t& operator=(const slice_t&) noexcept = default;
    constexpr slice_t(slice_t&&) noexcept = default;
    constexpr slice_t& operator=(slice_t&&) noexcept = default;
    ~slice_t() = default;

    constexpr static slice_t all() noexcept
    {
        return slice_t({}, {}, {});
    }

    constexpr static slice_t range(T start, T stop) noexcept
    {
        return slice_t(start, stop, {});
    }

    constexpr static slice_t range(T length) noexcept
    {
        return slice_t({}, length, {});
    }

    constexpr static slice_t index(T index) noexcept
    {
        return slice_t(index, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }

public:
    bool is_known() const noexcept
    {
        return static_cast<bool>(stop_);
    }

    bool is_all() const noexcept
    {
        return start() == 0 && step_ == 1 && !is_known();
    }

    bool is_index() const noexcept
    {
        return is_known() && stop_.value() == std::numeric_limits<T>::min() && step_ == std::numeric_limits<T>::max();
    }

    bool is_start_known() const noexcept
    {
        return start_.has_value();
    }

    T index() const noexcept
    {
        ASSERT(is_index());
        return start();
    }

    index_mapping_t<T> compute() const noexcept
    {
        ASSERT(is_known());
        if (is_index()) {
            return index_mapping_t<T>::single_index(start());
        }
        return index_mapping_t<T>::slice(std::array<T, 3>{start(), stop_.value(), step_});
    }

    index_mapping_t<T> compute(T length) const noexcept
    {
        if (is_index()) {
            return index_mapping_t<T>::single_index(start());
        }
        if (stop_) {
            length = std::min(length, stop_.value() - start());
        }
        return index_mapping_t<T>::slice(std::array<T, 3>{start(), start() + length, step_});
    }

    T start() const noexcept
    {
        if (start_.has_value()) {
            return start_.value();
        }
        return 0;
    }

    T stop() const noexcept
    {
        ASSERT(is_known());
        return *stop_;
    }

    T step() const noexcept
    {
        return step_;
    }

private:
    std::optional<T> start_;
    std::optional<T> stop_;
    T step_;
};

template <typename T>
using slice_vector_t = small_vector<slice_t<T>>;

using slice = slice_t<int64_t>;
using slice_vector = slice_vector_t<int64_t>;

} // namespace icm
