#pragma once

#include "shape.hpp"
#include "slice.hpp"

#include "exceptions.hpp"

#include <base/format.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace icm {
template <typename I>
shape slice_shape(const shape& shape, I begin, I end)
{
    auto d = std::distance(begin, end);
    if (shape.size() < d) {
        throw out_of_range(fmt::format(
            "Can't subscript more than dimensions. Shape has {} dimensions, but {} were requested.", shape.size(), d));
    }

    icm::small_vector<icm::shape::value_type> ret;
    ret.reserve(shape.size());
    for (auto it = begin; it != end; ++it) {
        if (!it->is_single_index()) {
            ret.push_back(static_cast<icm::shape::value_type>(it->size()));
        }
    }
    std::copy(shape.begin() + d, shape.end(), std::back_inserter(ret));
    return icm::shape(ret);
}

template <bool enforce_less_equal>
inline slice_vector slice_shape_to_max_side(const shape& shape, int max_side)
{
    slice_vector ret;
    if (max_side <= 0) {
        std::transform(shape.begin(), shape.end(), std::back_inserter(ret), [](auto s) {
            return slice::all();
        });
        return ret;
    }

    auto scale = static_cast<float>(*std::max_element(shape.begin(), shape.end())) / max_side;
    if (scale <= 1) {
        std::transform(shape.begin(), shape.end(), std::back_inserter(ret), [](auto s) {
            return slice::all();
        });
        return ret;
    }

    if constexpr (enforce_less_equal) {
        scale = std::ceil(scale);
    }

    for (const auto& s : shape) {
        ret.emplace_back(slice(0, {}, scale));
    }

    return ret;
}

} // namespace icm
