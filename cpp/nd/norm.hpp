#pragma once

/**
 * @file functions.hpp
 * @brief Definitions of all functions on `nd::array`.
 */

#include <cstdint>
#include <vector>

#include <icm/slice.hpp>

namespace nd {

class array;

/**
 * @brief Returns l1 norm of the array.
 */
array l1_norm(array a);

/**
 * @brief Returns l2 norm of the array.
 */
array l2_norm(array a);

/**
 * @brief Returns infinite norm of the array.
 */
array linf_norm(array a);

/**
 * @brief Returns infinite norm of the array.
 */
array cosine_similarity(array f, array s);

/**
 * @brief Returns hamming distance of two arrays.
 */
array hamming_distance(array f, array s);

array quantize(const array data);

array mean_pool_rows(array input, size_t dim, size_t rows);

array maxsim_pooled(array document, array query);

array maxsim_bq(array document, array query);

/**
 * @brief Returns maxsim operator result of two arrays.
 * @param document Document array.
 * @param query Query array.
 * @link https://zilliz.com/learn/explore-colbert-token-level-embedding-and-ranking-model-for-similarity-search
 */
array maxsim(array document, array query);

} // namespace nd
