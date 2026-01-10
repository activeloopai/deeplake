#pragma once

/**
 * @file hybrid_query_merge.hpp
 * @brief 
 */

#include <icm/index_mapping.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/none.hpp>
#include <query_core/query_result.hpp>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace pg {

/**
 * @brief Apply softmax normalization to query result scores
 * @param result The query result to normalize
 * @param max_weight Maximum weight to clamp scores (default 700.0)
 * @return Normalized scores as nd::array
 */
inline nd::array softmax(const query_core::query_result& result, double max_weight = 700.0)
{
    if (result.is_empty() || result.scores.is_none()) {
        return nd::none(nd::dtype::float32, 0);
    }

    auto shape = result.scores.shape();
    if (shape.empty()) {
        return nd::none(nd::dtype::float32, 0);
    }

    const size_t num_scores = static_cast<size_t>(shape[0]);
    std::vector<float> exp_scores;
    exp_scores.reserve(num_scores);

    // Compute exponentials with clamping
    for (size_t i = 0; i < num_scores; ++i) {
        float score = result.scores[static_cast<size_t>(i)].value<float>(0);
        float clamped_score = std::min(score, static_cast<float>(max_weight));
        exp_scores.push_back(std::exp(clamped_score));
    }

    // Compute sum of exponentials
    float sum_exp = std::accumulate(exp_scores.begin(), exp_scores.end(), 0.0f);

    // Normalize by sum
    std::vector<float> normalized_scores;
    normalized_scores.reserve(num_scores);
    for (float exp_score : exp_scores) {
        normalized_scores.push_back(exp_score / sum_exp);
    }

    return nd::adapt(normalized_scores);
}

/**
 * @brief Merge two query results using weighted combination with softmax normalization
 * @param embedding_result Result from embedding search (cosine similarity or maxsim)
 * @param text_result Result from text search (BM25)
 * @param embedding_weight Weight for embedding scores (default 0.5)
 * @param text_weight Weight for text scores (default 0.5)
 * @param top_k Maximum number of results to return (default 10)
 * @return Merged query result
 */
inline query_core::query_result merge_query_results(
    const query_core::query_result& embedding_result,
    const query_core::query_result& text_result,
    double embedding_weight = 0.5,
    double text_weight = 0.5,
    size_t top_k = 10)
{
    // Apply softmax normalization to both results
    auto emb_normalized = softmax(embedding_result);
    auto text_normalized = softmax(text_result);

    // Create a map to store combined scores by index
    std::unordered_map<int64_t, float> combined_scores;

    // Process embedding results
    if (!embedding_result.is_empty() && !emb_normalized.is_none()) {
        auto emb_shape = emb_normalized.shape();
        if (!emb_shape.empty()) {
            const size_t num_emb = static_cast<size_t>(emb_shape[0]);
            for (size_t i = 0; i < num_emb; ++i) {
                int64_t idx = embedding_result.indices[static_cast<size_t>(i)];
                float score = emb_normalized[static_cast<size_t>(i)].value<float>(0) * static_cast<float>(embedding_weight);
                combined_scores[idx] = score;
            }
        }
    }

    // Process text results
    if (!text_result.is_empty() && !text_normalized.is_none()) {
        auto text_shape = text_normalized.shape();
        if (!text_shape.empty()) {
            const size_t num_text = static_cast<size_t>(text_shape[0]);
            for (size_t i = 0; i < num_text; ++i) {
                int64_t idx = text_result.indices[static_cast<size_t>(i)];
                float score = text_normalized[static_cast<size_t>(i)].value<float>(0) * static_cast<float>(text_weight);
                combined_scores[idx] += score; // Add to existing score if index already exists
            }
        }
    }

    // Convert map to vectors for final result
    std::vector<float> final_scores;
    std::vector<int64_t> final_indices;

    for (const auto& [idx, score] : combined_scores) {
        final_scores.push_back(score);
        final_indices.push_back(idx);
    }

    // Sort by score in descending order
    std::vector<size_t> sort_indices(final_scores.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);
    std::sort(sort_indices.begin(), sort_indices.end(), 
              [&](size_t a, size_t b) { return final_scores[a] > final_scores[b]; });

    // Take top_k results
    size_t result_size = std::min(top_k, final_scores.size());
    std::vector<float> top_scores;
    std::vector<int64_t> top_indices;
    top_scores.reserve(result_size);
    top_indices.reserve(result_size);

    for (size_t i = 0; i < result_size; ++i) {
        size_t idx = sort_indices[i];
        top_scores.push_back(final_scores[idx]);
        top_indices.push_back(final_indices[idx]);
    }

    // Create final result
    query_core::query_result merged_result;
    if (!top_scores.empty()) {
        merged_result.scores = nd::adapt(top_scores);
        merged_result.indices = icm::index_mapping_t<int64_t>::list(std::move(top_indices));
    }

    return merged_result;
}

} // namespace pg
