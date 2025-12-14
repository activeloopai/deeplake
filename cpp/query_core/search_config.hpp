#pragma once

/**
 * @file search_config.hpp
 * @brief Configuration structures for vector search and index building operations
 */

#include <cstdint>

namespace query_core {

/**
 * @brief Configuration for vector search operations.
 *
 * These parameters control the accuracy/performance trade-off during
 * vector similarity searches using clustered indices.
 */
struct search_config
{
    /**
     * @brief Rerank multiplier for quantized/MMR searches.
     *
     * Controls how many additional candidates to fetch for reranking.
     * Higher values improve accuracy at the cost of performance.
     * When using quantized indices or MMR, the actual number of candidates
     * retrieved is: k * accuracy_factor
     *
     * Default: 10
     * Typical range: 5-20
     */
    int32_t accuracy_factor = 10;

    /**
     * @brief Cluster search accuracy factor.
     *
     * Scales the number of clusters searched during vector search.
     * Higher values search more clusters for better accuracy at the cost of performance.
     * The formula is: num_clusters_to_search = base_formula * cluster_search_accuracy_factor
     *
     * Default: 1.0
     * Typical range: 0.5-2.0
     */
    double cluster_search_accuracy_factor = 1.0;

    /**
     * @brief Returns the default search configuration.
     * Balanced accuracy and performance.
     */
    static search_config default_config()
    {
        return search_config{};
    }

    /**
     * @brief Returns a high-accuracy search configuration.
     * Prioritizes accuracy over performance.
     */
    static search_config high_accuracy()
    {
        return search_config{.accuracy_factor = 20, .cluster_search_accuracy_factor = 8.0};
    }

    /**
     * @brief Returns a balanced search configuration.
     * Same as default_config().
     */
    static search_config balanced()
    {
        return search_config{.accuracy_factor = 5, .cluster_search_accuracy_factor = 2.0};
    }

    /**
     * @brief Returns a fast search configuration.
     * Prioritizes performance over accuracy.
     */
    static search_config fast()
    {
        return search_config{.accuracy_factor = 1, .cluster_search_accuracy_factor = 0.5};
    }

    /**
     * @brief Equality comparison operator.
     */
    bool operator==(const search_config& other) const
    {
        return accuracy_factor == other.accuracy_factor &&
               cluster_search_accuracy_factor == other.cluster_search_accuracy_factor;
    }

    /**
     * @brief Inequality comparison operator.
     */
    bool operator!=(const search_config& other) const
    {
        return !(*this == other);
    }
};

/**
 * @brief Configuration for index building operations.
 *
 * These parameters affect how indices are built, particularly for clustered indices.
 */
struct index_build_config
{
    /**
     * @brief K-means iteration multiplier for clustered index building.
     *
     * Controls the number of iterations used during k-means clustering when building
     * clustered indices. Higher values result in more iterations and potentially better
     * clustering quality at the cost of longer build times.
     *
     * Default: 4
     * Typical range: 2-10
     */
    int32_t build_multiplier = 4;

    /**
     * @brief Returns the default index build configuration.
     */
    static index_build_config default_config()
    {
        return index_build_config{};
    }

    /**
     * @brief Returns a high-quality index build configuration.
     * Results in better clustering but takes longer to build.
     */
    static index_build_config high_quality()
    {
        return index_build_config{.build_multiplier = 8};
    }

    /**
     * @brief Returns a fast index build configuration.
     * Builds quickly but may result in lower clustering quality.
     */
    static index_build_config fast()
    {
        return index_build_config{.build_multiplier = 2};
    }

    /**
     * @brief Equality comparison operator.
     */
    bool operator==(const index_build_config& other) const
    {
        return build_multiplier == other.build_multiplier;
    }

    /**
     * @brief Inequality comparison operator.
     */
    bool operator!=(const index_build_config& other) const
    {
        return !(*this == other);
    }
};

} // namespace query_core
