#pragma once

namespace format {

/**
 * @brief Template class to define serialization logic of the specific type.
 * `serializable` defines the following operations for T
 * - Default construction of `T`. If `T` is default constructible then this can be skipped. Otherwise,
 *   there should be function `T serializable<T>::construct()`
 * - Reader for `T` - `read_result<T> serializable<T>::read(const base::memory_buffer& bytes, int64_t offset)`
 *   This function reads the T from the given buffer at the given offset, and returns the result.
 * - Writer for `T`. There are two possible ways to define writer logic.
 *   1. (Preferred) If the write size for the given object is known for type `T`. Then serializable
 *      defines the following two functions:
 *      ```
 *      static int64_t serializable<T>::output_size(const C&);
 *      static void serializable<T>::write(const T& o, buffer_t& bytes, int64_t offset);
 *   2. If the write size can't be calculated easily. Then the common implementation is the following.
 *      ```
 *      static buffer_t serializable<T>::write(const C& o)
 *      ```
 *   For the first case, serialization logic allocates buffer with enough size and calls in-place write.
 *   For the second case, serialization calls write and then attaches the given result to the output buffer.
 * @tparam T The type for which the serialization is defined.
 */
template <typename T>
struct serializable;

}