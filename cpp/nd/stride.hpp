#pragma once

/**
 * @file stride.hpp
 * @brief Definitions of `stride` functions.
 */

#include "array.hpp"

#include <icm/index_mapping.hpp>
#include <icm/indexable.hpp>
#include <icm/slice.hpp>
#include <icm/small_vector.hpp>

namespace nd {

array stride(array arr, icm::indexable_vector strides);

template <typename I>
array stride(array arr, const icm::indexable_t<I>& stride);

array stride(array arr, const icm::index_mapping_vector& strides);

template <typename I>
array stride(array arr, icm::index_mapping_t<I> stride);

array stride(array arr, const icm::slice_vector& slices);

template <typename I>
array stride(array arr, icm::slice_t<I> slice);


/**
 * @brief Returns the flipped array across the given axis.
 * @param axis The axis to flip.
 * @return array
 * @note Arithmetic data types are supported.
 */
array flip(array a, int64_t axis);

/**
 * @brief Returns the flipped array across the given axes.
 * @param a The array to flip.
 * @param axis The axes to flip. If the length of the axes is less than the array dimensions, the remaining axes are
 * not flipped. If the length of the axes is greater than the array dimensions, the extra axes are ignored.
 * @return array
 * @note Arithmetic data types are supported.
 */
array flip(array a, const std::vector<bool>& axes);


}
