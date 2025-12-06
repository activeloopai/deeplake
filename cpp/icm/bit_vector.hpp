#pragma once

#include "exceptions.hpp"

#include <base/assert.hpp>
#include <base/system_report.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

// SIMD intrinsics
#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse64)
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace icm {

// Forward declaration
class bit_vector_view;

class bit_vector
{
private:
    using block_type = uint64_t;
    static constexpr int64_t bits_per_block = 64;
    static constexpr int64_t simd_128_blocks = 2; // 128-bit / 64-bit blocks
    static constexpr int64_t simd_256_blocks = 4; // 256-bit / 64-bit blocks

public:
    using value_type = int64_t;

    // Construction
    explicit bit_vector(int64_t size, bool initial_value = false)
        : size_(size)
    {
        int64_t num_blocks = (size + bits_per_block - 1) / bits_per_block;
        blocks_.resize(num_blocks, initial_value ? ~block_type(0) : block_type(0));

        // Clear unused bits in last block if initial_value is true
        if (initial_value && size % bits_per_block != 0) {
            int64_t used_bits = size % bits_per_block;
            block_type mask = (block_type(1) << used_bits) - 1;
            blocks_.back() = blocks_.back() & mask;
        }
    }

    // Copy constructor with move semantics
    bit_vector(const bit_vector& other) = default;
    bit_vector(bit_vector&& other) noexcept
        : blocks_(std::move(other.blocks_))
        , size_(other.size_)
    {
        other.size_ = 0;
    }
    bit_vector& operator=(const bit_vector& other) = default;
    bit_vector& operator=(bit_vector&& other) noexcept
    {
        if (this != &other) {
            blocks_ = std::move(other.blocks_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    // Basic operations
    int64_t size() const
    {
        return size_;
    }

    int64_t num_blocks() const
    {
        return blocks_.size();
    }

    // Bit access - optimized for bulk operations
    bool get(int64_t index) const
    {
        ASSERT(index < size_);
        int64_t block_idx = index / bits_per_block;
        int64_t bit_idx = index % bits_per_block;
        return (blocks_[block_idx] >> bit_idx) & 1;
    }

    void set(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t block_idx = index / bits_per_block;
        int64_t bit_idx = index % bits_per_block;
        blocks_[block_idx] |= block_type(1) << bit_idx;
    }

    void clear(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t block_idx = index / bits_per_block;
        int64_t bit_idx = index % bits_per_block;
        blocks_[block_idx] = blocks_[block_idx] & (~(block_type(1) << bit_idx));
    }

    void flip(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t block_idx = index / bits_per_block;
        int64_t bit_idx = index % bits_per_block;
        blocks_[block_idx] ^= block_type(1) << bit_idx;
    }

    // Bulk operations - highly optimized
    void set_all()
    {
        std::fill(blocks_.begin(), blocks_.end(), ~block_type(0));
    }

    void clear_all()
    {
        std::fill(blocks_.begin(), blocks_.end(), block_type(0));
    }

    void flip_all()
    {
        // Flip all complete blocks
        for (auto& block : blocks_) {
            block = ~block;
        }

        // Clear unused bits in last block if size is not a multiple of bits_per_block
        if (size_ % bits_per_block != 0) {
            int64_t used_bits = size_ % bits_per_block;
            block_type mask = (block_type(1) << used_bits) - 1;
            blocks_.back() = blocks_.back() & mask;
        }
    }

    void set_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t start_block = start / bits_per_block;
        int64_t end_block = (end - 1) / bits_per_block;

        if (start_block == end_block) {
            // Range within single block
            int64_t start_bit = start % bits_per_block;
            int64_t end_bit = end % bits_per_block;
            if (end_bit == 0) {
                end_bit = bits_per_block;
            }
            int64_t num_bits = end_bit - start_bit;
            if (num_bits == bits_per_block) {
                blocks_[start_block] = ~block_type(0);
            } else {
                block_type mask = ((block_type(1) << num_bits) - 1) << start_bit;
                blocks_[start_block] |= mask;
            }
        } else {
            // Range spans multiple blocks
            // Set partial first block
            int64_t start_bit = start % bits_per_block;
            if (start_bit != 0) {
                block_type mask = ~((block_type(1) << start_bit) - 1);
                blocks_[start_block] |= mask;
                start_block++;
            }

            // Set complete middle blocks
            for (int64_t i = start_block; i < end_block; i++) {
                blocks_[i] = ~block_type(0);
            }

            // Set partial last block
            int64_t end_bit = end % bits_per_block;
            if (end_bit != 0) {
                block_type mask = (block_type(1) << end_bit) - 1;
                blocks_[end_block] |= mask;
            } else if (end_block < blocks_.size()) {
                blocks_[end_block] = ~block_type(0);
            }
        }
    }

    void flip_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t start_block = start / bits_per_block;
        int64_t end_block = (end - 1) / bits_per_block;

        if (start_block == end_block) {
            // Range within single block
            int64_t start_bit = start % bits_per_block;
            int64_t end_bit = end % bits_per_block;
            if (end_bit == 0) {
                end_bit = bits_per_block;
            }
            int64_t num_bits = end_bit - start_bit;
            if (num_bits == bits_per_block) {
                blocks_[start_block] = ~blocks_[start_block];
            } else {
                block_type mask = ((block_type(1) << num_bits) - 1) << start_bit;
                blocks_[start_block] ^= mask;
            }
        } else {
            // Range spans multiple blocks
            // Flip partial first block
            int64_t start_bit = start % bits_per_block;
            if (start_bit != 0) {
                block_type mask = ~((block_type(1) << start_bit) - 1);
                blocks_[start_block] ^= mask;
                start_block++;
            }

            // Flip complete middle blocks
            for (int64_t i = start_block; i < end_block; i++) {
                blocks_[i] = ~blocks_[i];
            }

            // Flip partial last block
            int64_t end_bit = end % bits_per_block;
            if (end_bit != 0) {
                block_type mask = (block_type(1) << end_bit) - 1;
                blocks_[end_block] ^= mask;
            } else if (end_block < blocks_.size()) {
                blocks_[end_block] = ~blocks_[end_block];
            }
        }
    }

    // Cross-platform SIMD-optimized bitwise operations
    bit_vector& operator&=(const bit_vector& other)
    {
        ASSERT(size_ == other.size_);

        const auto& features = base::system_report::get_cpu_features();

        if (features.has_simd_256 && blocks_.size() >= simd_256_blocks) {
#ifdef __x86_64__
            return bitwise_and_avx2(other);
#endif
        } else if (features.has_simd_128 && blocks_.size() >= simd_128_blocks) {
#ifdef __x86_64__
            return bitwise_and_sse2(other);
#elif defined(__aarch64__) || defined(__arm__)
            return bitwise_and_neon(other);
#endif
        }

        return bitwise_and_scalar(other);
    }

    bit_vector& operator|=(const bit_vector& other)
    {
        ASSERT(size_ == other.size_);

        const auto& features = base::system_report::get_cpu_features();

        if (features.has_simd_256 && blocks_.size() >= simd_256_blocks) {
#ifdef __x86_64__
            return bitwise_or_avx2(other);
#endif
        } else if (features.has_simd_128 && blocks_.size() >= simd_128_blocks) {
#ifdef __x86_64__
            return bitwise_or_sse2(other);
#elif defined(__aarch64__) || defined(__arm__)
            return bitwise_or_neon(other);
#endif
        }

        return bitwise_or_scalar(other);
    }

    bit_vector& operator^=(const bit_vector& other)
    {
        ASSERT(size_ == other.size_);

        const auto& features = base::system_report::get_cpu_features();

        if (features.has_simd_256 && blocks_.size() >= simd_256_blocks) {
#ifdef __x86_64__
            return bitwise_xor_avx2(other);
#endif
        } else if (features.has_simd_128 && blocks_.size() >= simd_128_blocks) {
#ifdef __x86_64__
            return bitwise_xor_sse2(other);
#elif defined(__aarch64__) || defined(__arm__)
            return bitwise_xor_neon(other);
#endif
        }

        return bitwise_xor_scalar(other);
    }

    // Optimized population count with cross-platform support
    int64_t count_set_bits() const
    {
        int64_t count = 0;
        const auto& features = base::system_report::get_cpu_features();

        if (features.has_popcnt) {
#ifdef __x86_64__
            count = count_set_bits_x86_popcnt();
#elif defined(__aarch64__)
            count = count_set_bits_arm_popcnt();
#else
            count = count_set_bits_builtin();
#endif
        } else {
            count = count_set_bits_software();
        }

        return count;
    }

    // Find first/last set bit - optimized with cross-platform support
    int64_t find_first_set() const
    {
        const auto& features = base::system_report::get_cpu_features();

        for (int64_t i = 0; i < blocks_.size(); i++) {
            if (blocks_[i] != 0) {
                int bit_pos;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanForward64(&index, blocks_[i])) {
                        bit_pos = static_cast<int>(index);
                    } else {
                        bit_pos = ctz_software(blocks_[i]);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        bit_pos = _tzcnt_u64(blocks_[i]);
                    } else {
                        bit_pos = ctz_software(blocks_[i]);
                    }
#else
                    bit_pos = ctz_software(blocks_[i]);
#endif
#else
                    bit_pos = __builtin_ctzll(blocks_[i]);
#endif
                } else {
                    bit_pos = ctz_software(blocks_[i]);
                }

                int64_t global_pos = i * bits_per_block + bit_pos;
                return (global_pos < size_) ? global_pos : -1;
            }
        }
        return size_;
    }

    int64_t find_last_set() const
    {
        const auto& features = base::system_report::get_cpu_features();

        for (int64_t i = blocks_.size() - 1; i >= 0; i--) {
            if (blocks_[i] != 0) {
                int bit_pos;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanReverse64(&index, blocks_[i])) {
                        bit_pos = static_cast<int>(index);
                    } else {
                        bit_pos = clz_software(blocks_[i]);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        bit_pos = bits_per_block - 1 - _lzcnt_u64(blocks_[i]);
                    } else {
                        bit_pos = clz_software(blocks_[i]);
                    }
#else
                    bit_pos = clz_software(blocks_[i]);
#endif
#else
                    bit_pos = bits_per_block - 1 - __builtin_clzll(blocks_[i]);
#endif
                } else {
                    bit_pos = clz_software(blocks_[i]);
                }

                int64_t global_pos = i * bits_per_block + bit_pos;
                return (global_pos < static_cast<int64_t>(size_)) ? global_pos : -1;
            }
        }
        return size_;
    }

    // Find next set bit starting from position (exclusive)
    int64_t find_next_set(int64_t start_pos) const
    {
        if (start_pos >= size_) {
            return size_;
        }

        int64_t search_pos = start_pos + 1;
        if (search_pos >= size_) {
            return size_;
        }

        int64_t block_idx = search_pos / bits_per_block;
        int64_t bit_idx = search_pos % bits_per_block;

        const auto& features = base::system_report::get_cpu_features();

        // Check remaining bits in current block
        if (block_idx < blocks_.size()) {
            block_type current_block = blocks_[block_idx];

            // Mask out bits before search position
            if (bit_idx > 0) {
                current_block = current_block & (~((block_type(1) << bit_idx) - 1));
            }

            // If there's a set bit in current block
            if (current_block != 0) {
                int offset;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanForward64(&index, current_block)) {
                        offset = static_cast<int>(index);
                    } else {
                        offset = ctz_software(current_block);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        offset = _tzcnt_u64(current_block);
                    } else {
                        offset = ctz_software(current_block);
                    }
#else
                    offset = ctz_software(current_block);
#endif
#else
                    offset = __builtin_ctzll(current_block);
#endif
                } else {
                    offset = ctz_software(current_block);
                }

                int64_t found_pos = block_idx * bits_per_block + offset;
                if (found_pos < size_) {
                    return found_pos;
                }
            }
        }

        // Scan subsequent blocks for any set bits
        for (int64_t i = block_idx + 1; i < blocks_.size(); i++) {
            if (blocks_[i] != 0) {
                int offset;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanForward64(&index, blocks_[i])) {
                        offset = static_cast<int>(index);
                    } else {
                        offset = ctz_software(blocks_[i]);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        offset = _tzcnt_u64(blocks_[i]);
                    } else {
                        offset = ctz_software(blocks_[i]);
                    }
#else
                    offset = ctz_software(blocks_[i]);
#endif
#else
                    offset = __builtin_ctzll(blocks_[i]);
#endif
                } else {
                    offset = ctz_software(blocks_[i]);
                }

                int64_t found_pos = i * bits_per_block + offset;
                if (found_pos < size_) {
                    return found_pos;
                }
            }
        }

        return size_; // No more set bits found
    }

    // Find previous set bit starting from position (exclusive)
    int64_t find_prev_set(int64_t start_pos) const
    {
        if (start_pos == 0) {
            return -1;
        }

        int64_t search_pos = start_pos - 1;
        if (search_pos >= size_) {
            search_pos = size_ - 1;
        }

        int64_t block_idx = search_pos / bits_per_block;
        int64_t bit_idx = search_pos % bits_per_block;

        const auto& features = base::system_report::get_cpu_features();

        // Check bits up to search position in current block
        if (block_idx < blocks_.size()) {
            block_type current_block = blocks_[block_idx];

            // Mask out bits after search position
            if (bit_idx != bits_per_block - 1) {
                current_block &= ((block_type(1) << (bit_idx + 1)) - 1);
            }

            // If there's a set bit in current block
            if (current_block != 0) {
                int offset;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanReverse64(&index, current_block)) {
                        offset = static_cast<int>(index);
                    } else {
                        offset = clz_software(current_block);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        offset = bits_per_block - 1 - _lzcnt_u64(current_block);
                    } else {
                        offset = clz_software(current_block);
                    }
#else
                    offset = clz_software(current_block);
#endif
#else
                    offset = bits_per_block - 1 - __builtin_clzll(current_block);
#endif
                } else {
                    offset = clz_software(current_block);
                }

                return block_idx * bits_per_block + offset;
            }
        }

        // Scan previous blocks for any set bits
        for (int64_t i = static_cast<int64_t>(block_idx) - 1; i >= 0; i--) {
            if (blocks_[i] != 0) {
                int offset;

                if (features.has_fast_bitops) {
#ifdef _MSC_VER
                    unsigned long index;
                    if (_BitScanReverse64(&index, blocks_[i])) {
                        offset = static_cast<int>(index);
                    } else {
                        offset = clz_software(blocks_[i]);
                    }
#elif defined(__x86_64__)
#ifdef __BMI1__
                    if (features.has_bmi1) {
                        offset = bits_per_block - 1 - _lzcnt_u64(blocks_[i]);
                    } else {
                        offset = clz_software(blocks_[i]);
                    }
#else
                    offset = clz_software(blocks_[i]);
#endif
#else
                    offset = bits_per_block - 1 - __builtin_clzll(blocks_[i]);
#endif
                } else {
                    offset = clz_software(blocks_[i]);
                }

                return i * bits_per_block + offset;
            }
        }

        return -1; // No previous set bits found
    }

    // Iterator support for efficient traversal of set bits
    class set_bit_iterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = int64_t;
        using difference_type = int64_t;
        using pointer = int64_t*;
        using reference = int64_t&;

    private:
        const bit_vector* bv_;
        int64_t current_position_;

    public:
        set_bit_iterator(const bit_vector* bv, bool at_end = false)
            : bv_(bv)
        {
            if (at_end) {
                current_position_ = bv_->size();
            } else {
                current_position_ = bv_->find_first_set();
            }
        }

        int64_t operator*() const
        {
            ASSERT(current_position_ >= 0);
            return static_cast<int64_t>(current_position_);
        }

        set_bit_iterator& operator++()
        {
            if (current_position_ >= 0) {
                current_position_ = bv_->find_next_set(static_cast<int64_t>(current_position_));
            }
            return *this;
        }

        set_bit_iterator operator++(int)
        {
            set_bit_iterator tmp = *this;
            ++*this;
            return tmp;
        }

        set_bit_iterator& operator--()
        {
            if (current_position_ >= 0) {
                current_position_ = bv_->find_prev_set(static_cast<int64_t>(current_position_));
            }
            return *this;
        }

        set_bit_iterator operator--(int)
        {
            set_bit_iterator tmp = *this;
            --*this;
            return tmp;
        }

        bool operator!=(const set_bit_iterator& other) const
        {
            return current_position_ != other.current_position_;
        }

        bool operator==(const set_bit_iterator& other) const
        {
            return current_position_ == other.current_position_;
        }
    };

    set_bit_iterator begin() const
    {
        return set_bit_iterator(this);
    }
    set_bit_iterator end() const
    {
        return set_bit_iterator(this, true);
    }

    // Utility methods for query engines
    bool any() const
    {
        for (auto block : blocks_) {
            if (block != 0) {
                return true;
            }
        }
        return false;
    }

    bool none() const
    {
        return !any();
    }

    bool all() const
    {
        return count_set_bits() == size_;
    }

    // Memory-efficient serialization
    void serialize(std::vector<uint8_t>& buffer) const
    {
        buffer.resize(sizeof(size_t) + blocks_.size() * sizeof(block_type));

        size_t offset = 0;
        std::memcpy(buffer.data() + offset, &size_, sizeof(size_t));
        offset += sizeof(size_t);

        if (!blocks_.empty()) {
            std::memcpy(buffer.data() + offset, blocks_.data(), blocks_.size() * sizeof(block_type));
        }
    }

    static bit_vector deserialize(const std::vector<uint8_t>& buffer)
    {
        // Check if buffer is large enough to contain the size_t header
        if (buffer.size() < sizeof(size_t)) {
            throw exception("Buffer too small to contain bit_vector header");
        }

        size_t size;
        std::memcpy(&size, buffer.data(), sizeof(size_t));

        bit_vector result(size);
        size_t blocks_size = (buffer.size() - sizeof(size_t)) / sizeof(block_type);

        // Only copy data if there are blocks to copy and the buffer is large enough
        if (blocks_size > 0 && !result.blocks_.empty()) {
            // Validate that the buffer contains enough data for the calculated blocks
            size_t expected_buffer_size = sizeof(size_t) + blocks_size * sizeof(block_type);
            if (buffer.size() < expected_buffer_size) {
                throw std::runtime_error("Buffer too small for bit_vector data");
            }

            std::memcpy(result.blocks_.data(), buffer.data() + sizeof(size_t), blocks_size * sizeof(block_type));
        }

        return result;
    }

    // Direct access to underlying data for advanced operations
    const block_type* data() const
    {
        return blocks_.data();
    }

    block_type* data()
    {
        return blocks_.data();
    }

    // Memory usage
    size_t memory_usage() const
    {
        return sizeof(*this) + blocks_.size() * sizeof(block_type);
    }

    // Conversion utilities - template-based for any iterator range
    template <typename Iterator>
    static bit_vector from_indices(Iterator begin, Iterator end, int64_t max_size)
    {
        bit_vector result(max_size, false);
        for (auto it = begin; it != end; ++it) {
            int64_t index = *it;
            ASSERT(index >= 0 && index < max_size);
            result.set(index);
        }
        return result;
    }

    template <typename Iterator>
    void set_from_indices(Iterator begin, Iterator end)
    {
        for (auto it = begin; it != end; ++it) {
            int64_t index = *it;
            ASSERT(index >= 0 && index < size_);
            int64_t block_idx = index / bits_per_block;
            int64_t bit_idx = index % bits_per_block;
            blocks_[block_idx] |= block_type(1) << bit_idx;
        }
    }

    // Fast utility function to convert 64 consecutive numeric values to uint64_t mask
    template <typename T>
    static uint64_t bools_to_mask_64(const T* values)
    {
        const auto& features = base::system_report::get_cpu_features();

        if (features.has_simd_256) {
#ifdef __x86_64__
            return bools_to_mask_64_avx2(values);
#endif
        } else if (features.has_simd_128) {
#ifdef __x86_64__
            return bools_to_mask_64_sse2(values);
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
            return bools_to_mask_64_neon(values);
#endif
        }

        return bools_to_mask_64_scalar(values);
    }

    // Overload with count parameter for bounds checking
    template <typename T>
    static uint64_t bools_to_mask_64(const T* values, int count)
    {
        const auto& features = base::system_report::get_cpu_features();

        if (features.has_simd_256) {
#ifdef __x86_64__
            return bools_to_mask_64_avx2(values, count);
#endif
        } else if (features.has_simd_128) {
#ifdef __x86_64__
            return bools_to_mask_64_sse2(values, count);
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
            return bools_to_mask_64_neon(values, count);
#endif
        }

        return bools_to_mask_64_scalar(values, count);
    }

    // Make bit_vector_view a friend so it can access private methods
    friend class bit_vector_view;

    // Set bit_vector from std::span of numeric values (converts to bool)
    template <typename T>
    void set_from_span(std::span<const T> values)
    {
        ASSERT(values.size() == size_);

        // Process 64 bits at a time for better performance
        int64_t full_blocks = size_ / bits_per_block;
        int64_t remaining_bits = size_ % bits_per_block;

        // Process complete 64-bit blocks
        for (int64_t block = 0; block < full_blocks; ++block) {
            const T* block_start = values.data() + block * bits_per_block;
            blocks_[block] = bools_to_mask_64(block_start);
        }

        // Process remaining bits in the last block
        if (remaining_bits > 0) {
            const T* last_block_start = values.data() + full_blocks * bits_per_block;
            uint64_t mask = bools_to_mask_64(last_block_start, remaining_bits);
            // Clear unused bits (defensive, should already be handled by count parameter)
            mask &= (uint64_t(1) << remaining_bits) - 1;
            blocks_[full_blocks] = mask;
        }
    }

    // Create a view into a specific range of this bit_vector
    bit_vector_view span(int64_t start_index, int64_t end_index);

    // Add clear_range method that was referenced in bit_vector_view
    void clear_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t start_block = start / bits_per_block;
        int64_t end_block = (end - 1) / bits_per_block;

        if (start_block == end_block) {
            // Range within single block
            int64_t start_bit = start % bits_per_block;
            int64_t end_bit = end % bits_per_block;
            if (end_bit == 0) {
                end_bit = bits_per_block;
            }
            int64_t num_bits = end_bit - start_bit;
            if (num_bits == bits_per_block) {
                blocks_[start_block] = block_type(0);
            } else {
                block_type mask = ~(((block_type(1) << num_bits) - 1) << start_bit);
                blocks_[start_block] &= mask;
            }
        } else {
            // Range spans multiple blocks
            // Clear partial first block
            int64_t start_bit = start % bits_per_block;
            if (start_bit != 0) {
                block_type mask = (block_type(1) << start_bit) - 1;
                blocks_[start_block] &= mask;
                start_block++;
            }

            // Clear complete middle blocks
            for (int64_t i = start_block; i < end_block; i++) {
                blocks_[i] = block_type(0);
            }

            // Clear partial last block
            int64_t end_bit = end % bits_per_block;
            if (end_bit != 0) {
                block_type mask = ~((block_type(1) << end_bit) - 1);
                blocks_[end_block] &= mask;
            } else if (end_block < blocks_.size()) {
                blocks_[end_block] = block_type(0);
            }
        }
    }

private:
    // x86 SSE2 implementations (128-bit)
#ifdef __x86_64__
    __attribute__((target("sse2"))) bit_vector& bitwise_and_sse2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            __m128i a = _mm_loadu_si128((__m128i*)&blocks_[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other.blocks_[i]);
            __m128i result = _mm_and_si128(a, b);
            _mm_storeu_si128((__m128i*)&blocks_[i], result);
        }

        // Handle remaining blocks
        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] &= other.blocks_[i];
        }

        return *this;
    }

    __attribute__((target("sse2"))) bit_vector& bitwise_or_sse2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            __m128i a = _mm_loadu_si128((__m128i*)&blocks_[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other.blocks_[i]);
            __m128i result = _mm_or_si128(a, b);
            _mm_storeu_si128((__m128i*)&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] |= other.blocks_[i];
        }

        return *this;
    }

    __attribute__((target("sse2"))) bit_vector& bitwise_xor_sse2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            __m128i a = _mm_loadu_si128((__m128i*)&blocks_[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other.blocks_[i]);
            __m128i result = _mm_xor_si128(a, b);
            _mm_storeu_si128((__m128i*)&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] ^= other.blocks_[i];
        }

        return *this;
    }

    // x86 AVX2 implementations (256-bit)
    __attribute__((target("avx2"))) bit_vector& bitwise_and_avx2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_256_blocks) * simd_256_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_256_blocks) {
            __m256i a = _mm256_loadu_si256((__m256i*)&blocks_[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other.blocks_[i]);
            __m256i result = _mm256_and_si256(a, b);
            _mm256_storeu_si256((__m256i*)&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] &= other.blocks_[i];
        }

        return *this;
    }

    __attribute__((target("avx2"))) bit_vector& bitwise_or_avx2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_256_blocks) * simd_256_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_256_blocks) {
            __m256i a = _mm256_loadu_si256((__m256i*)&blocks_[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other.blocks_[i]);
            __m256i result = _mm256_or_si256(a, b);
            _mm256_storeu_si256((__m256i*)&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] |= other.blocks_[i];
        }

        return *this;
    }

    __attribute__((target("avx2"))) bit_vector& bitwise_xor_avx2(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_256_blocks) * simd_256_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_256_blocks) {
            __m256i a = _mm256_loadu_si256((__m256i*)&blocks_[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other.blocks_[i]);
            __m256i result = _mm256_xor_si256(a, b);
            _mm256_storeu_si256((__m256i*)&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] ^= other.blocks_[i];
        }

        return *this;
    }

    __attribute__((target("popcnt"))) size_t count_set_bits_x86_popcnt() const
    {
        size_t count = 0;
        for (block_type block : blocks_) {
            count += _mm_popcnt_u64(block);
        }

        // Adjust for unused bits in last block
        if (size_ % bits_per_block != 0) {
            size_t unused_bits = bits_per_block - (size_ % bits_per_block);
            block_type last_block_mask = (block_type(1) << (size_ % bits_per_block)) - 1;
            block_type unused_block = blocks_.back() & ~last_block_mask;
            count -= _mm_popcnt_u64(unused_block);
        }

        return count;
    }
#endif

    // ARM NEON implementations (128-bit)
#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
    bit_vector& bitwise_and_neon(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            uint64x2_t a = vld1q_u64(&blocks_[i]);
            uint64x2_t b = vld1q_u64(&other.blocks_[i]);
            uint64x2_t result = vandq_u64(a, b);
            vst1q_u64(&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] &= other.blocks_[i];
        }

        return *this;
    }

    bit_vector& bitwise_or_neon(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            uint64x2_t a = vld1q_u64(&blocks_[i]);
            uint64x2_t b = vld1q_u64(&other.blocks_[i]);
            uint64x2_t result = vorrq_u64(a, b);
            vst1q_u64(&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] |= other.blocks_[i];
        }

        return *this;
    }

    bit_vector& bitwise_xor_neon(const bit_vector& other)
    {
        size_t simd_blocks = (blocks_.size() / simd_128_blocks) * simd_128_blocks;

        for (size_t i = 0; i < simd_blocks; i += simd_128_blocks) {
            uint64x2_t a = vld1q_u64(&blocks_[i]);
            uint64x2_t b = vld1q_u64(&other.blocks_[i]);
            uint64x2_t result = veorq_u64(a, b);
            vst1q_u64(&blocks_[i], result);
        }

        for (size_t i = simd_blocks; i < blocks_.size(); i++) {
            blocks_[i] ^= other.blocks_[i];
        }

        return *this;
    }

#ifdef __aarch64__
    size_t count_set_bits_arm_popcnt() const
    {
        size_t count = 0;
        for (block_type block : blocks_) {
            count += __builtin_popcountll(block); // ARM64 has efficient builtin
        }

        // Handle unused bits adjustment
        if (size_ % bits_per_block != 0) {
            size_t unused_bits = bits_per_block - (size_ % bits_per_block);
            block_type last_block_mask = (block_type(1) << (size_ % bits_per_block)) - 1;
            block_type unused_block = blocks_.back() & ~last_block_mask;
            count -= __builtin_popcountll(unused_block);
        }

        return count;
    }
#endif
#endif

    // Builtin implementations (fallback for non-x86)
    size_t count_set_bits_builtin() const
    {
        size_t count = 0;

        for (block_type block : blocks_) {
#ifdef _MSC_VER
            count += __popcnt64(block);
#else
            count += __builtin_popcountll(block);
#endif
        }

        // Adjust for unused bits in last block
        if (size_ % bits_per_block != 0) {
            size_t unused_bits = bits_per_block - (size_ % bits_per_block);
            block_type last_block_mask = (block_type(1) << (size_ % bits_per_block)) - 1;
            block_type unused_block = blocks_.back() & ~last_block_mask;
#ifdef _MSC_VER
            count -= __popcnt64(unused_block);
#else
            count -= __builtin_popcountll(unused_block);
#endif
        }

        return count;
    }

    // Software implementations for older hardware
    size_t count_set_bits_software() const
    {
        size_t count = 0;

        for (block_type block : blocks_) {
            count += popcount_software(block);
        }

        // Adjust for unused bits in last block
        if (size_ % bits_per_block != 0) {
            size_t unused_bits = bits_per_block - (size_ % bits_per_block);
            block_type last_block_mask = (block_type(1) << (size_ % bits_per_block)) - 1;
            block_type unused_block = blocks_.back() & ~last_block_mask;
            count -= popcount_software(unused_block);
        }

        return count;
    }

    // Scalar fallback implementations
    bit_vector& bitwise_and_scalar(const bit_vector& other)
    {
        for (size_t i = 0; i < blocks_.size(); i++) {
            blocks_[i] &= other.blocks_[i];
        }
        return *this;
    }

    bit_vector& bitwise_or_scalar(const bit_vector& other)
    {
        for (size_t i = 0; i < blocks_.size(); i++) {
            blocks_[i] |= other.blocks_[i];
        }
        return *this;
    }

    bit_vector& bitwise_xor_scalar(const bit_vector& other)
    {
        for (size_t i = 0; i < blocks_.size(); i++) {
            blocks_[i] ^= other.blocks_[i];
        }
        return *this;
    }

    // Software implementations for missing instructions
    static size_t popcount_software(block_type x)
    {
        // Brian Kernighan's algorithm
        size_t count = 0;
        while (x) {
            x = x & (x - 1);
            count++;
        }
        return count;
    }

    static int ctz_software(block_type x)
    {
        if (x == 0) {
            return bits_per_block;
        }
#ifdef _MSC_VER
        unsigned long index;
        if (_BitScanForward64(&index, x)) {
            return static_cast<int>(index);
        }
        return bits_per_block;
#else
        int count = 0;
        while ((x & 1) == 0) {
            x >>= 1;
            count++;
        }
        return count;
#endif
    }

    static int clz_software(block_type x)
    {
        if (x == 0) {
            return bits_per_block;
        }
#ifdef _MSC_VER
        // On Windows, _BitScanReverse64 is always available, so we use it
        // This should return the bit index (0-63), not CLZ count
        // to match the behavior expected by callers on Windows
        unsigned long index;
        if (_BitScanReverse64(&index, x)) {
            return static_cast<int>(index);
        }
        return -1; // Should never happen if x != 0
#else
        int count = 0;
        block_type mask = block_type(1) << (bits_per_block - 1);
        while ((x & mask) == 0) {
            x <<= 1;
            count++;
        }
        return bits_per_block - 1 - count;
#endif
    }

    // x86 SSE2 implementations (128-bit)
#ifdef __x86_64__
    template <typename T>
    __attribute__((target("sse2"))) static uint64_t bools_to_mask_64_sse2(const T* values, int count = 64)
    {
        uint64_t result = 0;
        int i = 0;

        // For types larger than 1 byte, we need to handle them differently
        if constexpr (sizeof(T) == 1) {
            // Process 16 values at a time using SSE2 for 8-bit types
            for (; i + 16 <= count && i + 16 <= 64; i += 16) {
                __m128i v = _mm_loadu_si128((__m128i*)(values + i));
                __m128i zero = _mm_setzero_si128();
                __m128i mask = _mm_cmpeq_epi8(v, zero);
                __m128i not_mask = _mm_xor_si128(mask, _mm_set1_epi8(0xFF));

                // Extract the mask bits - _mm_movemask_epi8 returns 16 bits for 16 bytes
                uint32_t mask_16 = _mm_movemask_epi8(not_mask);
                // Each bit in mask_16 corresponds to one byte, so we can use it directly
                result |= (uint64_t)mask_16 << i;
            }
        } else if constexpr (sizeof(T) == 2) {
            // Process 8 values at a time using SSE2 for 16-bit types
            for (; i + 8 <= count && i + 8 <= 64; i += 8) {
                __m128i v = _mm_loadu_si128((__m128i*)(values + i));
                __m128i zero = _mm_setzero_si128();
                __m128i mask = _mm_cmpeq_epi16(v, zero);
                __m128i not_mask = _mm_xor_si128(mask, _mm_set1_epi16(0xFFFF));

                // Extract the mask bits (8 bits for 8 16-bit values)
                uint32_t mask_8 = _mm_movemask_epi8(not_mask);
                // Convert to 8-bit mask by taking every other bit
                uint32_t mask_16 = 0;
                for (int j = 0; j < 8; ++j) {
                    if (mask_8 & (1 << (j * 2 + 1))) {
                        mask_16 |= (1 << j);
                    }
                }
                result |= (uint64_t)mask_16 << i;
            }
        } else if constexpr (sizeof(T) == 4) {
            // Process 4 values at a time using SSE2 for 32-bit types
            for (; i + 4 <= count && i + 4 <= 64; i += 4) {
                __m128i v = _mm_loadu_si128((__m128i*)(values + i));
                __m128i zero = _mm_setzero_si128();
                __m128i mask = _mm_cmpeq_epi32(v, zero);
                __m128i not_mask = _mm_xor_si128(mask, _mm_set1_epi32(0xFFFFFFFF));

                // Extract the mask bits (4 bits for 4 32-bit values)
                uint32_t mask_4 = _mm_movemask_ps(_mm_castsi128_ps(not_mask));
                result |= (uint64_t)mask_4 << i;
            }
        } else if constexpr (sizeof(T) == 8) {
            // Process 2 values at a time using SSE2 for 64-bit types
            // SSE2 doesn't have _mm_cmpeq_epi64, so we compare 32-bit parts separately
            for (; i + 2 <= count && i + 2 <= 64; i += 2) {
                __m128i v = _mm_loadu_si128((__m128i*)(values + i));
                __m128i zero = _mm_setzero_si128();

                // Compare high and low 32-bit parts separately
                __m128i low_mask = _mm_cmpeq_epi32(v, zero);
                __m128i high_mask = _mm_cmpeq_epi32(_mm_srli_si128(v, 4), zero);

                // Combine the masks: both high and low must be zero for the 64-bit value to be zero
                __m128i combined_mask = _mm_and_si128(low_mask, high_mask);
                __m128i not_mask = _mm_xor_si128(combined_mask, _mm_set1_epi32(0xFFFFFFFF));

                // Extract the mask bits (2 bits for 2 64-bit values)
                uint32_t mask_2 = _mm_movemask_ps(_mm_castsi128_ps(not_mask));
                result |= (uint64_t)mask_2 << i;
            }
        }

        // Handle remaining values
        for (; i < count && i < 64; ++i) {
            if (values[i] != T{0}) {
                result |= uint64_t(1) << i;
            }
        }

        return result;
    }
#endif

    // x86 AVX2 implementations (256-bit)
#ifdef __x86_64__
    template <typename T>
    __attribute__((target("avx2"))) static uint64_t bools_to_mask_64_avx2(const T* values, int count = 64)
    {
        uint64_t result = 0;
        int i = 0;

        // For types larger than 1 byte, we need to handle them differently
        if constexpr (sizeof(T) == 1) {
            // Process 32 values at a time using AVX2 for 8-bit types
            for (; i + 32 <= count && i + 32 <= 64; i += 32) {
                __m256i v = _mm256_loadu_si256((__m256i*)(values + i));
                __m256i zero = _mm256_setzero_si256();
                __m256i mask = _mm256_cmpeq_epi8(v, zero);
                __m256i not_mask = _mm256_xor_si256(mask, _mm256_set1_epi8(0xFF));

                // Extract the mask bits
                uint32_t mask_32 = _mm256_movemask_epi8(not_mask);
                result |= (uint64_t)mask_32 << i;
            }
        } else if constexpr (sizeof(T) == 2) {
            // Process 16 values at a time using AVX2 for 16-bit types
            for (; i + 16 <= count && i + 16 <= 64; i += 16) {
                __m256i v = _mm256_loadu_si256((__m256i*)(values + i));
                __m256i zero = _mm256_setzero_si256();
                __m256i mask = _mm256_cmpeq_epi16(v, zero);
                __m256i not_mask = _mm256_xor_si256(mask, _mm256_set1_epi16(0xFFFF));

                // Extract the mask bits (16 bits for 16 16-bit values)
                // Use _mm256_movemask_epi8 and convert to 16-bit mask
                uint32_t mask_32 = _mm256_movemask_epi8(not_mask);
                // Convert to 16-bit mask by taking every other bit
                uint32_t mask_16 = 0;
                for (int j = 0; j < 16; ++j) {
                    if (mask_32 & (1 << (j * 2 + 1))) {
                        mask_16 |= (1 << j);
                    }
                }
                result |= (uint64_t)mask_16 << i;
            }
        } else if constexpr (sizeof(T) == 4) {
            // Process 8 values at a time using AVX2 for 32-bit types
            for (; i + 8 <= count && i + 8 <= 64; i += 8) {
                __m256i v = _mm256_loadu_si256((__m256i*)(values + i));
                __m256i zero = _mm256_setzero_si256();
                __m256i mask = _mm256_cmpeq_epi32(v, zero);
                __m256i not_mask = _mm256_xor_si256(mask, _mm256_set1_epi32(0xFFFFFFFF));

                // Extract the mask bits (8 bits for 8 32-bit values)
                uint32_t mask_8 = _mm256_movemask_ps(_mm256_castsi256_ps(not_mask));
                result |= (uint64_t)mask_8 << i;
            }
        } else if constexpr (sizeof(T) == 8) {
            // Process 4 values at a time using AVX2 for 64-bit types
            for (; i + 4 <= count && i + 4 <= 64; i += 4) {
                __m256i v = _mm256_loadu_si256((__m256i*)(values + i));
                __m256i zero = _mm256_setzero_si256();
                __m256i mask = _mm256_cmpeq_epi64(v, zero);
                __m256i not_mask = _mm256_xor_si256(mask, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF));

                // Extract the mask bits (4 bits for 4 64-bit values)
                uint32_t mask_4 = _mm256_movemask_pd(_mm256_castsi256_pd(not_mask));
                result |= (uint64_t)mask_4 << i;
            }
        }

        // Handle remaining values
        for (; i < count && i < 64; ++i) {
            if (values[i] != T{0}) {
                result |= uint64_t(1) << i;
            }
        }

        return result;
    }
#endif

    // ARM NEON implementations (128-bit)
#if defined(__aarch64__) || defined(__arm__)
#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
    template <typename T>
    static uint64_t bools_to_mask_64_neon(const T* values, int count = 64)
    {
        uint64_t result = 0;
        int i = 0;

        // For types larger than 1 byte, we need to handle them differently
        if constexpr (sizeof(T) == 1) {
            // Process 16 values at a time using NEON for 8-bit types
            for (; i + 16 <= count && i + 16 <= 64; i += 16) {
                uint8x16_t v = vld1q_u8((uint8_t*)(values + i));
                uint8x16_t zero = vdupq_n_u8(0);
                uint8x16_t mask = vceqq_u8(v, zero);
                uint8x16_t not_mask = vmvnq_u8(mask);

                // Extract the mask bits - NEON doesn't have movemask, so we need to extract each bit manually
                // We'll extract the high bit of each byte to create our mask
                uint8x16_t high_bits = vshrq_n_u8(not_mask, 7); // Shift right by 7 to get MSB of each byte

                // Extract the 16 bits as individual bytes and combine them
                uint8_t bits[16];
                vst1q_u8(bits, high_bits);

                // Combine the bits into a 16-bit mask
                uint16_t mask_16 = 0;
                for (int j = 0; j < 16; ++j) {
                    if (bits[j] & 1) {
                        mask_16 |= (1 << j);
                    }
                }
                result |= (uint64_t)mask_16 << i;
            }
        } else if constexpr (sizeof(T) == 2) {
            // Process 8 values at a time using NEON for 16-bit types
            for (; i + 8 <= count && i + 8 <= 64; i += 8) {
                uint16x8_t v = vld1q_u16((uint16_t*)(values + i));
                uint16x8_t zero = vdupq_n_u16(0);
                uint16x8_t mask = vceqq_u16(v, zero);
                uint16x8_t not_mask = vmvnq_u16(mask);

                // Extract the mask bits - get the high bit of each 16-bit value
                uint16x8_t high_bits = vshrq_n_u16(not_mask, 15); // Shift right by 15 to get MSB

                // Extract the 8 bits as individual 16-bit values
                uint16_t bits[8];
                vst1q_u16(bits, high_bits);

                // Combine the bits into an 8-bit mask
                uint8_t mask_8 = 0;
                for (int j = 0; j < 8; ++j) {
                    if (bits[j] & 1) {
                        mask_8 |= (1 << j);
                    }
                }
                result |= (uint64_t)mask_8 << i;
            }
        } else if constexpr (sizeof(T) == 4) {
            // Process 4 values at a time using NEON for 32-bit types
            for (; i + 4 <= count && i + 4 <= 64; i += 4) {
                uint32x4_t v = vld1q_u32((uint32_t*)(values + i));
                uint32x4_t zero = vdupq_n_u32(0);
                uint32x4_t mask = vceqq_u32(v, zero);
                uint32x4_t not_mask = vmvnq_u32(mask);

                // Extract the mask bits - get the high bit of each 32-bit value
                uint32x4_t high_bits = vshrq_n_u32(not_mask, 31); // Shift right by 31 to get MSB

                // Extract the 4 bits as individual 32-bit values
                uint32_t bits[4];
                vst1q_u32(bits, high_bits);

                // Combine the bits into a 4-bit mask
                uint8_t mask_4 = 0;
                for (int j = 0; j < 4; ++j) {
                    if (bits[j] & 1) {
                        mask_4 |= (1 << j);
                    }
                }
                result |= (uint64_t)mask_4 << i;
            }
        } else if constexpr (sizeof(T) == 8) {
            // Process 2 values at a time using NEON for 64-bit types
            for (; i + 2 <= count && i + 2 <= 64; i += 2) {
                uint64x2_t v = vld1q_u64((uint64_t*)(values + i));
                uint64x2_t zero = vdupq_n_u64(0);
                uint64x2_t mask = vceqq_u64(v, zero);

                // For 64-bit types, we need to handle the mask differently since vmvnq_u64 doesn't exist
                // Extract the original values to check if they are non-zero
                uint64_t val1 = vgetq_lane_u64(v, 0);
                uint64_t val2 = vgetq_lane_u64(v, 1);

                // If the value is non-zero, set the corresponding bit
                if (val1 != 0) {
                    result |= uint64_t(1) << i;
                }
                if (val2 != 0) {
                    result |= uint64_t(1) << (i + 1);
                }
            }
        }

        // Handle remaining values
        for (; i < count && i < 64; ++i) {
            if (values[i] != T{0}) {
                result |= uint64_t(1) << i;
            }
        }

        return result;
    }
#endif
#endif

    // Scalar fallback for bools_to_mask_64
    template <typename T>
    static uint64_t bools_to_mask_64_scalar(const T* values)
    {
        uint64_t result = 0;
        for (int i = 0; i < 64; ++i) {
            if (values[i] != T{0}) {
                result |= uint64_t(1) << i;
            }
        }
        return result;
    }

    // Scalar fallback with count parameter
    template <typename T>
    static uint64_t bools_to_mask_64_scalar(const T* values, int count)
    {
        uint64_t result = 0;
        for (int i = 0; i < count && i < 64; ++i) {
            if (values[i] != T{0}) {
                result |= uint64_t(1) << i;
            }
        }
        return result;
    }

private:
    friend class bit_vector_view;

    std::vector<block_type> blocks_;
    int64_t size_;
};

// Additional operators for convenience
inline bit_vector operator&(const bit_vector& a, const bit_vector& b)
{
    bit_vector result = a;
    result &= b;
    return result;
}

inline bit_vector operator|(const bit_vector& a, const bit_vector& b)
{
    bit_vector result = a;
    result |= b;
    return result;
}

inline bit_vector operator^(const bit_vector& a, const bit_vector& b)
{
    bit_vector result = a;
    result ^= b;
    return result;
}

inline bit_vector operator~(const bit_vector& a)
{
    bit_vector result = a;
    // Flip all bits
    for (size_t i = 0; i < result.num_blocks(); i++) {
        result.data()[i] = ~result.data()[i];
    }
    // Clear unused bits in last block
    if (result.size() % 64 != 0) {
        size_t unused_bits = 64 - (result.size() % 64);
        result.data()[result.num_blocks() - 1] =
            result.data()[result.num_blocks() - 1] & ((uint64_t(1) << (result.size() % 64)) - 1);
    }
    return result;
}

// Comparison operators
inline bool operator==(const bit_vector& a, const bit_vector& b)
{
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.num_blocks(); i++) {
        if (a.data()[i] != b.data()[i]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const bit_vector& a, const bit_vector& b)
{
    return !(a == b);
}

// Stream operators for debugging
inline std::ostream& operator<<(std::ostream& os, const bit_vector& bv)
{
    for (size_t i = 0; i < bv.size(); i++) {
        os << (bv.get(i) ? '1' : '0');
        if ((i + 1) % 8 == 0 && i + 1 < bv.size()) {
            os << ' ';
        }
    }
    return os;
}

class bit_vector_view
{
private:
    using block_type = uint64_t;
    static constexpr int64_t bits_per_block = 64;

    bit_vector* original_bv_;
    int64_t start_block_;
    int64_t end_block_;
    int64_t end_bit_offset_;
    int64_t size_;

public:
    using value_type = int64_t;

    // Construction from bit_vector span
    bit_vector_view(bit_vector* bv, int64_t start_block, int64_t end_block, int64_t end_bit, int64_t size)
        : original_bv_(bv)
        , start_block_(start_block)
        , end_block_(end_block)
        , end_bit_offset_(end_bit)
        , size_(size)
    {
    }

    // Copy constructor
    bit_vector_view(const bit_vector_view& other) = default;
    bit_vector_view& operator=(const bit_vector_view& other) = default;

    // Basic operations
    int64_t size() const
    {
        return size_;
    }

    int64_t num_blocks() const
    {
        return end_block_ - start_block_;
    }

    // Bit access - maps to original bit_vector
    bool get(int64_t index) const
    {
        ASSERT(index < size_);
        int64_t global_index = start_block_ * bits_per_block + index;
        return original_bv_->get(global_index);
    }

    void set(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t global_index = start_block_ * bits_per_block + index;
        original_bv_->set(global_index);
    }

    void clear(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t global_index = start_block_ * bits_per_block + index;
        original_bv_->clear(global_index);
    }

    void flip(int64_t index)
    {
        ASSERT(index >= 0 && index < size_);
        int64_t global_index = start_block_ * bits_per_block + index;
        original_bv_->flip(global_index);
    }

    // Bulk operations - delegate to original bit_vector
    void set_all()
    {
        set_range(0, size_);
    }

    void clear_all()
    {
        clear_range(0, size_);
    }

    void clear()
    {
        clear_all();
    }

    void flip_all()
    {
        flip_range(0, size_);
    }

    void set_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t global_start = start_block_ * bits_per_block + start;
        int64_t global_end = start_block_ * bits_per_block + end;
        original_bv_->set_range(global_start, global_end);
    }

    void clear_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t global_start = start_block_ * bits_per_block + start;
        int64_t global_end = start_block_ * bits_per_block + end;

        // Clear range in original bit_vector
        original_bv_->clear_range(global_start, global_end);
    }

    void flip_range(int64_t start, int64_t end)
    {
        ASSERT(start <= end && end <= size_);
        if (start == end) {
            return;
        }

        int64_t global_start = start_block_ * bits_per_block + start;
        int64_t global_end = start_block_ * bits_per_block + end;

        // Flip range in original bit_vector
        original_bv_->flip_range(global_start, global_end);
    }

    // Bitwise operations - optimized block-level operations
    bit_vector_view& operator&=(const bit_vector_view& other)
    {
        ASSERT(size_ == other.size_);

        // Handle case where views are of the same bit_vector
        if (original_bv_ == other.original_bv_) {
            // Get direct access to the underlying blocks
            block_type* this_blocks = original_bv_->data() + start_block_;
            const block_type* other_blocks = other.original_bv_->data() + other.start_block_;

            int64_t num_blocks = end_block_ - start_block_;

            // Use the same SIMD optimization logic as bit_vector
            const auto& features = base::system_report::get_cpu_features();

            if (features.has_simd_256 && num_blocks >= 4) {
#ifdef __x86_64__
                bitwise_and_avx2_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else if (features.has_simd_128 && num_blocks >= 2) {
#ifdef __x86_64__
                bitwise_and_sse2_blocks(this_blocks, other_blocks, num_blocks);
#elif defined(__aarch64__) || defined(__arm__)
                bitwise_and_neon_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else {
                bitwise_and_scalar_blocks(this_blocks, other_blocks, num_blocks);
            }

            // Handle the last block if it's partial
            if (end_bit_offset_ != 0) {
                block_type mask = (block_type(1) << end_bit_offset_) - 1;
                this_blocks[num_blocks - 1] &= mask;
            }
        } else {
            // Views are of different bit_vectors - use optimized bit-by-bit approach
            // This is still much faster than the original implementation
            int64_t full_blocks = size_ / bits_per_block;
            int64_t remaining_bits = size_ % bits_per_block;

            // Process complete blocks
            for (int64_t block = 0; block < full_blocks; ++block) {
                int64_t this_global_block = start_block_ + block;
                int64_t other_global_block = other.start_block_ + block;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                original_bv_->data()[this_global_block] = this_block & other_block;
            }

            // Process remaining bits
            if (remaining_bits > 0) {
                int64_t this_global_block = start_block_ + full_blocks;
                int64_t other_global_block = other.start_block_ + full_blocks;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                block_type mask = (block_type(1) << remaining_bits) - 1;
                original_bv_->data()[this_global_block] = (this_block & other_block) & mask;
            }
        }

        return *this;
    }

    bit_vector_view& operator|=(const bit_vector_view& other)
    {
        ASSERT(size_ == other.size_);

        // Handle case where views are of the same bit_vector
        if (original_bv_ == other.original_bv_) {
            // Get direct access to the underlying blocks
            block_type* this_blocks = original_bv_->data() + start_block_;
            const block_type* other_blocks = other.original_bv_->data() + other.start_block_;

            int64_t num_blocks = end_block_ - start_block_;

            // Use the same SIMD optimization logic as bit_vector
            const auto& features = base::system_report::get_cpu_features();

            if (features.has_simd_256 && num_blocks >= 4) {
#ifdef __x86_64__
                bitwise_or_avx2_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else if (features.has_simd_128 && num_blocks >= 2) {
#ifdef __x86_64__
                bitwise_or_sse2_blocks(this_blocks, other_blocks, num_blocks);
#elif defined(__aarch64__) || defined(__arm__)
                bitwise_or_neon_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else {
                bitwise_or_scalar_blocks(this_blocks, other_blocks, num_blocks);
            }

            // Handle the last block if it's partial
            if (end_bit_offset_ != 0) {
                block_type mask = (block_type(1) << end_bit_offset_) - 1;
                this_blocks[num_blocks - 1] &= mask;
            }
        } else {
            // Views are of different bit_vectors - use optimized block-level approach
            int64_t full_blocks = size_ / bits_per_block;
            int64_t remaining_bits = size_ % bits_per_block;

            // Process complete blocks
            for (int64_t block = 0; block < full_blocks; ++block) {
                int64_t this_global_block = start_block_ + block;
                int64_t other_global_block = other.start_block_ + block;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                original_bv_->data()[this_global_block] = this_block | other_block;
            }

            // Process remaining bits
            if (remaining_bits > 0) {
                int64_t this_global_block = start_block_ + full_blocks;
                int64_t other_global_block = other.start_block_ + full_blocks;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                block_type mask = (block_type(1) << remaining_bits) - 1;
                original_bv_->data()[this_global_block] = (this_block | other_block) & mask;
            }
        }

        return *this;
    }

    bit_vector_view& operator^=(const bit_vector_view& other)
    {
        ASSERT(size_ == other.size_);

        // Handle case where views are of the same bit_vector
        if (original_bv_ == other.original_bv_) {
            // Get direct access to the underlying blocks
            block_type* this_blocks = original_bv_->data() + start_block_;
            const block_type* other_blocks = other.original_bv_->data() + other.start_block_;

            int64_t num_blocks = end_block_ - start_block_;

            // Use the same SIMD optimization logic as bit_vector
            const auto& features = base::system_report::get_cpu_features();

            if (features.has_simd_256 && num_blocks >= 4) {
#ifdef __x86_64__
                bitwise_xor_avx2_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else if (features.has_simd_128 && num_blocks >= 2) {
#ifdef __x86_64__
                bitwise_xor_sse2_blocks(this_blocks, other_blocks, num_blocks);
#elif defined(__aarch64__) || defined(__arm__)
                bitwise_xor_neon_blocks(this_blocks, other_blocks, num_blocks);
#endif
            } else {
                bitwise_xor_scalar_blocks(this_blocks, other_blocks, num_blocks);
            }

            // Handle the last block if it's partial
            if (end_bit_offset_ != 0) {
                block_type mask = (block_type(1) << end_bit_offset_) - 1;
                this_blocks[num_blocks - 1] &= mask;
            }
        } else {
            // Views are of different bit_vectors - use optimized block-level approach
            int64_t full_blocks = size_ / bits_per_block;
            int64_t remaining_bits = size_ % bits_per_block;

            // Process complete blocks
            for (int64_t block = 0; block < full_blocks; ++block) {
                int64_t this_global_block = start_block_ + block;
                int64_t other_global_block = other.start_block_ + block;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                original_bv_->data()[this_global_block] = this_block ^ other_block;
            }

            // Process remaining bits
            if (remaining_bits > 0) {
                int64_t this_global_block = start_block_ + full_blocks;
                int64_t other_global_block = other.start_block_ + full_blocks;

                block_type this_block = original_bv_->data()[this_global_block];
                block_type other_block = other.original_bv_->data()[other_global_block];

                block_type mask = (block_type(1) << remaining_bits) - 1;
                original_bv_->data()[this_global_block] = (this_block ^ other_block) & mask;
            }
        }

        return *this;
    }

    // Population count - optimized block-level operation
    int64_t count_set_bits() const
    {
        // Get direct access to the underlying blocks
        const block_type* blocks = original_bv_->data() + start_block_;
        int64_t num_blocks = end_block_ - start_block_;

        int64_t count = 0;
        const auto& features = base::system_report::get_cpu_features();

        // Use the same optimization logic as bit_vector
        if (features.has_popcnt) {
#ifdef __x86_64__
            count = count_set_bits_x86_popcnt_blocks(blocks, num_blocks);
#elif defined(__aarch64__)
            count = count_set_bits_arm_popcnt_blocks(blocks, num_blocks);
#else
            count = count_set_bits_builtin_blocks(blocks, num_blocks);
#endif
        } else {
            count = count_set_bits_software_blocks(blocks, num_blocks);
        }

        // Adjust for unused bits in last block if it's partial
        if (end_bit_offset_ != 0) {
            block_type last_block_mask = (block_type(1) << end_bit_offset_) - 1;
            block_type unused_block = blocks[num_blocks - 1] & ~last_block_mask;
#ifdef _MSC_VER
            count -= __popcnt64(unused_block);
#else
            count -= __builtin_popcountll(unused_block);
#endif
        }

        return count;
    }

    // Find operations - map to global indices
    int64_t find_first_set() const
    {
        for (int64_t i = 0; i < size_; ++i) {
            if (get(i)) {
                return i;
            }
        }
        return size_;
    }

    int64_t find_last_set() const
    {
        for (int64_t i = size_ - 1; i >= 0; --i) {
            if (get(i)) {
                return i;
            }
        }
        return -1;
    }

    int64_t find_next_set(int64_t start_pos) const
    {
        if (start_pos >= size_) {
            return size_;
        }

        for (int64_t i = start_pos + 1; i < size_; ++i) {
            if (get(i)) {
                return i;
            }
        }
        return size_;
    }

    int64_t find_prev_set(int64_t start_pos) const
    {
        if (start_pos == 0) {
            return -1;
        }

        int64_t search_pos = start_pos - 1;
        if (search_pos >= size_) {
            search_pos = size_ - 1;
        }

        for (int64_t i = search_pos; i >= 0; --i) {
            if (get(i)) {
                return i;
            }
        }
        return -1;
    }

    // Iterator support - custom iterator for bit_vector_view
    class set_bit_iterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = int64_t;
        using difference_type = int64_t;
        using pointer = int64_t*;
        using reference = int64_t&;

    private:
        const bit_vector_view* view_;
        int64_t current_position_;

    public:
        set_bit_iterator(const bit_vector_view* view, bool at_end = false)
            : view_(view)
        {
            if (at_end) {
                current_position_ = view_->size();
            } else {
                current_position_ = view_->find_first_set();
            }
        }

        int64_t operator*() const
        {
            ASSERT(current_position_ >= 0);
            return static_cast<int64_t>(current_position_);
        }

        set_bit_iterator& operator++()
        {
            if (current_position_ >= 0) {
                current_position_ = view_->find_next_set(static_cast<int64_t>(current_position_));
            }
            return *this;
        }

        set_bit_iterator operator++(int)
        {
            set_bit_iterator tmp = *this;
            ++*this;
            return tmp;
        }

        set_bit_iterator& operator--()
        {
            if (current_position_ >= 0) {
                current_position_ = view_->find_prev_set(static_cast<int64_t>(current_position_));
            }
            return *this;
        }

        set_bit_iterator operator--(int)
        {
            set_bit_iterator tmp = *this;
            --*this;
            return tmp;
        }

        bool operator!=(const set_bit_iterator& other) const
        {
            return current_position_ != other.current_position_;
        }

        bool operator==(const set_bit_iterator& other) const
        {
            return current_position_ == other.current_position_;
        }
    };

    set_bit_iterator begin() const
    {
        return set_bit_iterator(this);
    }

    set_bit_iterator end() const
    {
        return set_bit_iterator(this, true);
    }

    // Utility methods - optimized block-level operations
    bool any() const
    {
        const block_type* blocks = original_bv_->data() + start_block_;
        int64_t num_blocks = end_block_ - start_block_;

        // Check complete blocks
        for (int64_t i = 0; i < num_blocks - 1; ++i) {
            if (blocks[i] != 0) {
                return true;
            }
        }

        // Check last block (may be partial)
        if (num_blocks > 0) {
            if (end_bit_offset_ != 0) {
                block_type mask = (block_type(1) << end_bit_offset_) - 1;
                if ((blocks[num_blocks - 1] & mask) != 0) {
                    return true;
                }
            } else {
                if (blocks[num_blocks - 1] != 0) {
                    return true;
                }
            }
        }

        return false;
    }

    bool none() const
    {
        return !any();
    }

    bool all() const
    {
        const block_type* blocks = original_bv_->data() + start_block_;
        int64_t num_blocks = end_block_ - start_block_;

        // Check complete blocks
        for (int64_t i = 0; i < num_blocks - 1; ++i) {
            if (blocks[i] != ~block_type(0)) {
                return false;
            }
        }

        // Check last block (may be partial)
        if (num_blocks > 0) {
            if (end_bit_offset_ != 0) {
                block_type mask = (block_type(1) << end_bit_offset_) - 1;
                if ((blocks[num_blocks - 1] & mask) != mask) {
                    return false;
                }
            } else {
                if (blocks[num_blocks - 1] != ~block_type(0)) {
                    return false;
                }
            }
        }

        return true;
    }

    // Direct access to underlying data (read-only)
    const block_type* data() const
    {
        return original_bv_->data() + start_block_;
    }

    // Memory usage
    size_t memory_usage() const
    {
        return sizeof(*this); // Only the view itself, not the data
    }

    // Conversion utilities
    template <typename Iterator>
    void set_from_indices(Iterator begin, Iterator end)
    {
        for (auto it = begin; it != end; ++it) {
            int64_t index = *it;
            ASSERT(index >= 0 && index < size_);
            set(index);
        }
    }

    // Set bit_vector_view from std::span of numeric values (converts to bool)
    template <typename T>
    void set_from_span(std::span<const T> values)
    {
        ASSERT(values.size() == size_);

        // Process 64 bits at a time for better performance
        int64_t full_blocks = size_ / bits_per_block;
        int64_t remaining_bits = size_ % bits_per_block;

        // Process complete 64-bit blocks
        for (int64_t block = 0; block < full_blocks; ++block) {
            const T* block_start = values.data() + block * bits_per_block;
            uint64_t mask = bit_vector::bools_to_mask_64(block_start);
            int64_t global_block_idx = start_block_ + block;
            original_bv_->data()[global_block_idx] = mask;
        }

        // Process remaining bits in the last block
        if (remaining_bits > 0) {
            const T* last_block_start = values.data() + full_blocks * bits_per_block;
            uint64_t mask = bit_vector::bools_to_mask_64(last_block_start, remaining_bits);
            // Clear unused bits (defensive, should already be handled by count parameter)
            mask &= (uint64_t(1) << remaining_bits) - 1;
            int64_t global_block_idx = start_block_ + full_blocks;
            original_bv_->data()[global_block_idx] = mask;
        }
    }

private:
    // Helper method to get global index
    int64_t to_global_index(int64_t local_index) const
    {
        return start_block_ * bits_per_block + local_index;
    }

    // Block-level bitwise operations - optimized implementations
    static void bitwise_and_scalar_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        for (int64_t i = 0; i < num_blocks; i++) {
            this_blocks[i] &= other_blocks[i];
        }
    }

    static void bitwise_or_scalar_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        for (int64_t i = 0; i < num_blocks; i++) {
            this_blocks[i] |= other_blocks[i];
        }
    }

    static void bitwise_xor_scalar_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        for (int64_t i = 0; i < num_blocks; i++) {
            this_blocks[i] ^= other_blocks[i];
        }
    }

#ifdef __x86_64__
    __attribute__((target("sse2"))) static void
    bitwise_and_sse2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            __m128i a = _mm_loadu_si128((__m128i*)&this_blocks[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other_blocks[i]);
            __m128i result = _mm_and_si128(a, b);
            _mm_storeu_si128((__m128i*)&this_blocks[i], result);
        }

        // Handle remaining blocks
        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] &= other_blocks[i];
        }
    }

    __attribute__((target("sse2"))) static void
    bitwise_or_sse2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            __m128i a = _mm_loadu_si128((__m128i*)&this_blocks[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other_blocks[i]);
            __m128i result = _mm_or_si128(a, b);
            _mm_storeu_si128((__m128i*)&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] |= other_blocks[i];
        }
    }

    __attribute__((target("sse2"))) static void
    bitwise_xor_sse2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            __m128i a = _mm_loadu_si128((__m128i*)&this_blocks[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&other_blocks[i]);
            __m128i result = _mm_xor_si128(a, b);
            _mm_storeu_si128((__m128i*)&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] ^= other_blocks[i];
        }
    }

    __attribute__((target("avx2"))) static void
    bitwise_and_avx2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 4) * 4;

        for (size_t i = 0; i < simd_blocks; i += 4) {
            __m256i a = _mm256_loadu_si256((__m256i*)&this_blocks[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other_blocks[i]);
            __m256i result = _mm256_and_si256(a, b);
            _mm256_storeu_si256((__m256i*)&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] &= other_blocks[i];
        }
    }

    __attribute__((target("avx2"))) static void
    bitwise_or_avx2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 4) * 4;

        for (size_t i = 0; i < simd_blocks; i += 4) {
            __m256i a = _mm256_loadu_si256((__m256i*)&this_blocks[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other_blocks[i]);
            __m256i result = _mm256_or_si256(a, b);
            _mm256_storeu_si256((__m256i*)&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] |= other_blocks[i];
        }
    }

    __attribute__((target("avx2"))) static void
    bitwise_xor_avx2_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 4) * 4;

        for (size_t i = 0; i < simd_blocks; i += 4) {
            __m256i a = _mm256_loadu_si256((__m256i*)&this_blocks[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other_blocks[i]);
            __m256i result = _mm256_xor_si256(a, b);
            _mm256_storeu_si256((__m256i*)&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] ^= other_blocks[i];
        }
    }
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
    static void bitwise_and_neon_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            uint64x2_t a = vld1q_u64(&this_blocks[i]);
            uint64x2_t b = vld1q_u64(&other_blocks[i]);
            uint64x2_t result = vandq_u64(a, b);
            vst1q_u64(&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] &= other_blocks[i];
        }
    }

    static void bitwise_or_neon_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            uint64x2_t a = vld1q_u64(&this_blocks[i]);
            uint64x2_t b = vld1q_u64(&other_blocks[i]);
            uint64x2_t result = vorrq_u64(a, b);
            vst1q_u64(&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] |= other_blocks[i];
        }
    }

    static void bitwise_xor_neon_blocks(block_type* this_blocks, const block_type* other_blocks, int64_t num_blocks)
    {
        size_t simd_blocks = (num_blocks / 2) * 2;

        for (size_t i = 0; i < simd_blocks; i += 2) {
            uint64x2_t a = vld1q_u64(&this_blocks[i]);
            uint64x2_t b = vld1q_u64(&other_blocks[i]);
            uint64x2_t result = veorq_u64(a, b);
            vst1q_u64(&this_blocks[i], result);
        }

        for (size_t i = simd_blocks; i < num_blocks; i++) {
            this_blocks[i] ^= other_blocks[i];
        }
    }
#endif

    // Population count helper functions
    static int64_t count_set_bits_software_blocks(const block_type* blocks, int64_t num_blocks)
    {
        int64_t count = 0;
        for (int64_t i = 0; i < num_blocks; i++) {
            count += popcount_software(blocks[i]);
        }
        return count;
    }

    static int64_t count_set_bits_builtin_blocks(const block_type* blocks, int64_t num_blocks)
    {
        int64_t count = 0;
        for (int64_t i = 0; i < num_blocks; i++) {
#ifdef _MSC_VER
            count += __popcnt64(blocks[i]);
#else
            count += __builtin_popcountll(blocks[i]);
#endif
        }
        return count;
    }

#ifdef __x86_64__
    __attribute__((target("popcnt"))) static int64_t count_set_bits_x86_popcnt_blocks(const block_type* blocks,
                                                                                      int64_t num_blocks)
    {
        int64_t count = 0;
        for (int64_t i = 0; i < num_blocks; i++) {
            count += _mm_popcnt_u64(blocks[i]);
        }
        return count;
    }
#endif

#ifdef __aarch64__
    static int64_t count_set_bits_arm_popcnt_blocks(const block_type* blocks, int64_t num_blocks)
    {
        int64_t count = 0;
        for (int64_t i = 0; i < num_blocks; i++) {
#ifdef _MSC_VER
            count += __popcnt64(blocks[i]);
#else
            count += __builtin_popcountll(blocks[i]); // ARM64 has efficient builtin
#endif
        }
        return count;
    }
#endif

    // Software popcount implementation
    static size_t popcount_software(block_type x)
    {
        // Brian Kernighan's algorithm
        size_t count = 0;
        while (x) {
            x = x & (x - 1);
            count++;
        }
        return count;
    }
};

inline bit_vector_view bit_vector::span(int64_t start_index, int64_t end_index)
{
    ASSERT(start_index >= 0 && start_index <= size_);
    ASSERT(end_index >= start_index && end_index <= size_);
    ASSERT(start_index % bits_per_block == 0 || start_index == 0);
    ASSERT(end_index % bits_per_block == 0 || end_index == size_);

    int64_t start_block = start_index / bits_per_block;
    int64_t end_block = (end_index + bits_per_block - 1) / bits_per_block;
    int64_t end_bit = end_index % bits_per_block;
    int64_t view_size = end_index - start_index;

    return bit_vector_view(this, start_block, end_block, end_bit, view_size);
}

// Additional operators for bit_vector_view convenience
inline icm::bit_vector operator&(const icm::bit_vector_view& a, const icm::bit_vector_view& b)
{
    icm::bit_vector result(a.size());
    for (int64_t i = 0; i < a.size(); ++i) {
        if (a.get(i) && b.get(i)) {
            result.set(i);
        }
    }
    return result;
}

inline icm::bit_vector operator|(const icm::bit_vector_view& a, const icm::bit_vector_view& b)
{
    icm::bit_vector result(a.size());
    for (int64_t i = 0; i < a.size(); ++i) {
        if (a.get(i) || b.get(i)) {
            result.set(i);
        }
    }
    return result;
}

inline icm::bit_vector operator^(const icm::bit_vector_view& a, const icm::bit_vector_view& b)
{
    icm::bit_vector result(a.size());
    for (int64_t i = 0; i < a.size(); ++i) {
        if (a.get(i) != b.get(i)) {
            result.set(i);
        }
    }
    return result;
}

// Comparison operators for bit_vector_view
inline bool operator==(const bit_vector_view& a, const bit_vector_view& b)
{
    if (a.size() != b.size()) {
        return false;
    }

    for (int64_t i = 0; i < a.size(); i++) {
        if (a.get(i) != b.get(i)) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const bit_vector_view& a, const bit_vector_view& b)
{
    return !(a == b);
}

// Stream operators for bit_vector_view debugging
inline std::ostream& operator<<(std::ostream& os, const bit_vector_view& bv)
{
    for (int64_t i = 0; i < bv.size(); i++) {
        os << (bv.get(i) ? '1' : '0');
        if ((i + 1) % 8 == 0 && i + 1 < bv.size()) {
            os << ' ';
        }
    }
    return os;
}

} // namespace icm
