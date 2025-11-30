#pragma once

/**
 * @file functions.hpp
 * @brief Definitions of all functions on `nd::array`.
 */

#include <cstdint>
#include <vector>

#include <icm/bit_vector.hpp>

namespace nd {

class array;

/**
 * @brief Fills output vector with indices which have non-zero element in the given array.
 *
 * @param a Input array
 * @param output Output vector to be filled with non-zero indices
 */
void nonzero(const array& a, std::vector<int64_t>& output);

/**
 * @brief Fills output bit vector view with non-zero positions from the given array.
 *
 * @param a Input array
 * @param output Output bit vector view to be filled with non-zero positions
 */
void nonzero(const array& a, icm::bit_vector_view output);

/**
 * @brief Returns array with true value if all of the input array elements contains truthy value.
 *
 * @param a
 * @return array
 */
bool all(const array& a);

/**
 * @brief Returns boolean array showing if all array elements along a given axis evaluates to True.
 *
 * @param a
 * @return array
 */
array all(const nd::array& a, int64_t axis);

/**
 * @brief Returns array with true value if any of the input array elements contains truthy value.
 *
 * @param a
 * @return array
 */
bool any(const array& a);

/**
 * @brief Returns boolean array showing if any array element along a given axis evaluates to True.
 *
 * @param a
 * @return array
 */
array any(const nd::array& a, int64_t axis);

/**
 * @brief Checks whether array contains the given subarray.
 * @param container - The container.
 * @param element - The element.
 * @return True if container contains the element.
 */
bool contains(const array& container, const array& element);

/**
 * @brief Returns single element array containing the max of all elements in the input array.
 *
 * @param a
 * @return array
 */
array amax(array a);

/**
 * @brief Returns single element array containing the min of all elements in the input array.
 *
 * @param a
 * @return array
 */
array amin(array a);

/**
 * @brief Returns single element array containing the sum of all elements in the input array.
 *
 * @param a
 * @return array
 */
array sum(array a);

/**
 * @brief Returns single element array containing the product of all elements in the input array.
 *
 * @param a
 * @return array
 */
array prod(array a);

/**
 * @brief Returns array which contains element wise absolute value of the input array.
 *
 * @param a
 * @return array
 */
array abs(array a);

/**
 * @brief Returns array which contains element wise square root of the input array.
 *
 * @param a
 * @return array
 */
array sqrt(array a);

/**
 * @brief Return single dimension array containing the data of input array.
 *
 * @param a
 * @return array
 */
array flat(array a);

/**
 * @brief Returns single element array containing dot product of the given arrays.
 *
 * @param f First array.
 * @param s Second array.
 * @return array
 */
array dot(array f, array s);

/**
 * @brief Returns the transposed array of the given array.
 */
array transpose(array a);

/**
 * @brief Returns the average of the given array.
 *
 * @param a
 * @return array
 */
array avg(array a);

} // namespace nd
