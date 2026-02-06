#pragma once

/**
 * @file string_array_holder.hpp
 * @brief Helper class to hold string array data.
 */

#include "array.hpp"

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

private:
    // Storage - kept as separate members for cache efficiency
    nd::array vstack_holder_;
    std::vector<int64_t> offsets_;
    std::vector<int64_t> range_offsets_;
    std::vector<const void*> holders_;

    // Zero-copy buffer cache: raw pointers to buffer data and offsets.
    // SAFETY: These pointers remain valid as long as holders_ contains the chunk arrays.
    // This eliminates shared_ptr atomic reference counting in get_range_data() hot path.
    std::vector<const uint8_t*> buffer_cache_;
    std::vector<const uint32_t*> offsets_cache_;

    const void* dynamic_std_holder_ = nullptr;
    const void* dynamic_icm_holder_ = nullptr;
    bool is_valid_ = true;

    void initialize(const nd::array& arr);
    void initialize_single_range(const auto& range_adapter, const nd::array& source_arr);
    void initialize_complex(const nd::array& arr);
    bool try_initialize_range_arrays(const auto& vstacked, const nd::array& fallback);
    void clear_range_data();
    std::string_view get_range_data(int64_t index) const;
    std::string_view get_dynamic_data(int64_t index) const;
    std::string_view get_vstack_data(int64_t index) const;
};

} // namespace nd
