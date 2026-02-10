#pragma once

/**
 * @file string_array_holder.hpp
 * @brief Helper class to hold string array data.
 */

#include "array.hpp"

#include <algorithm>
#include <string_view>

namespace nd {

class string_stream_array_holder
{
public:
    string_stream_array_holder() = default;
    explicit string_stream_array_holder(const nd::array& arr);
    std::string_view data(int64_t index) const;

    inline bool is_valid() const noexcept
    {
        return is_valid_;
    }

    // OPTIMIZATION: Check if all data comes from a single contiguous chunk.
    // When true, batch_data_fast() can be used for O(1) sequential access without binary search.
    inline bool is_single_chunk() const noexcept
    {
        return offsets_.size() == 1;
    }

    // OPTIMIZATION: Fast batch data access for single-chunk case.
    // Avoids binary search overhead in get_range_data() by directly computing string bounds.
    // REQUIRES: is_single_chunk() == true, index must be valid
    inline std::string_view batch_data_fast(int64_t index) const noexcept
    {
        // Single chunk: offsets_[0] = 0, buffer_cache_[0], offsets_cache_[0] are the only entries
        const auto local_idx = range_offsets_[0] + index;
        const auto* offsets = offsets_cache_[0];
        const auto start_offset = offsets[0];
        const auto str_start = offsets[local_idx] - start_offset;
        const auto str_end = offsets[local_idx + 1] - start_offset;
        const auto* buffer = buffer_cache_[0];
        return std::string_view(reinterpret_cast<const char*>(buffer + str_start), str_end - str_start);
    }

    // OPTIMIZATION: Bulk access for contiguous string data extraction.
    // Returns the raw buffer pointer and string offsets for a batch, enabling vectorized operations.
    // This allows callers to set up all string_t entries in a single pass without per-string calls.
    // REQUIRES: is_single_chunk() == true
    struct contiguous_string_data
    {
        const uint8_t* buffer;      // Pointer to raw string data
        const uint32_t* offsets;    // String offset array (offsets[i] to offsets[i+1] is string i)
        uint32_t base_offset;       // Offset to subtract from all offset values
        int64_t start_index;        // Starting index within the offset array
    };

    inline contiguous_string_data get_contiguous_strings(int64_t batch_start) const noexcept
    {
        // For single chunk, return direct access to buffer and offset arrays
        const auto local_idx = range_offsets_[0] + batch_start;
        const auto* offsets = offsets_cache_[0];
        const auto base_offset = offsets[0];
        return {buffer_cache_[0], offsets, base_offset, local_idx};
    }

    // Calculate total bytes for a batch of strings (useful for pre-allocation)
    inline uint64_t get_batch_total_bytes(int64_t batch_start, int64_t count) const noexcept
    {
        if (!is_single_chunk()) {
            return 0; // Only optimized for single-chunk case
        }
        const auto local_idx = range_offsets_[0] + batch_start;
        const auto* offsets = offsets_cache_[0];
        const auto start = offsets[local_idx];
        const auto end = offsets[local_idx + count];
        return end - start;
    }

    // MULTI-CHUNK SUPPORT: Methods for iterating over data that spans multiple chunks.
    // These enable efficient batch processing without per-row binary search overhead.

    // Find which chunk contains the given global index.
    // Returns: (chunk_index, row_within_chunk)
    inline std::pair<size_t, int64_t> find_chunk(int64_t global_index) const noexcept
    {
        // offsets_ contains cumulative row counts: [chunk0_end, chunk0_end+chunk1_size, ...]
        // So chunk i spans from (i==0 ? 0 : offsets_[i-1]) to offsets_[i]
        //
        // Binary search to find first cumulative offset > global_index
        auto it = std::upper_bound(offsets_.begin(), offsets_.end(), global_index);
        size_t chunk_idx = static_cast<size_t>(std::distance(offsets_.begin(), it));
        
        // Calculate row within chunk
        int64_t chunk_start = (chunk_idx == 0) ? 0 : offsets_[chunk_idx - 1];
        int64_t row_in_chunk = global_index - chunk_start;
        
        return {chunk_idx, row_in_chunk};
    }

    // Get the global row range [start, end) for a specific chunk.
    inline std::pair<int64_t, int64_t> get_chunk_bounds(size_t chunk_idx) const noexcept
    {
        // offsets_ is cumulative: chunk i spans from (i==0 ? 0 : offsets_[i-1]) to offsets_[i]
        int64_t chunk_start = (chunk_idx == 0) ? 0 : offsets_[chunk_idx - 1];
        int64_t chunk_end = offsets_[chunk_idx];
        return {chunk_start, chunk_end};
    }

    // Get contiguous string data for a specific chunk, starting at row_in_chunk.
    // Similar to get_contiguous_strings() but for multi-chunk case.
    inline contiguous_string_data get_chunk_strings(size_t chunk_idx, int64_t row_in_chunk) const noexcept
    {
        const auto local_idx = range_offsets_[chunk_idx] + row_in_chunk;
        const auto* offsets = offsets_cache_[chunk_idx];
        const auto base_offset = offsets[0];
        return {buffer_cache_[chunk_idx], offsets, base_offset, local_idx};
    }

    // Get the number of chunks
    inline size_t num_chunks() const noexcept
    {
        return offsets_.empty() ? 0 : offsets_.size();
    }

private:
    // Storage - kept as separate members for cache efficiency
    nd::array vstack_holder_;
    std::vector<int64_t> offsets_;
    std::vector<int64_t> range_offsets_;

    // Zero-copy buffer cache: raw pointers to chunk buffer data and string offsets.
    // SAFETY: These pointers remain valid as long as vstack_holder_ keeps the source array alive.
    // This eliminates shared_ptr atomic reference counting in get_range_data() hot path.
    std::vector<const uint8_t*> buffer_cache_;
    std::vector<const uint32_t*> offsets_cache_;

    const void* dynamic_std_holder_ = nullptr;
    const void* dynamic_icm_holder_ = nullptr;
    bool is_valid_ = true;

    void initialize(const nd::array& arr);
    void initialize_single_range(const auto& range_adapter, const nd::array& source_arr);
    void initialize_complex(const nd::array& arr);
    bool try_initialize_range_arrays(const auto& vstacked);
    void clear_range_data();
    std::string_view get_range_data(int64_t index) const;
    std::string_view get_dynamic_data(int64_t index) const;
    std::string_view get_vstack_data(int64_t index) const;
};

} // namespace nd
