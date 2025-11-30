#pragma once

#include <roaring/roaring64map.hh>

#include <cstdint>
#include <vector>

namespace icm {

/**
 * @brief Wrapper around Roaring Bitmaps with icm naming conventions
 * 
 * This class provides a clean interface to Roaring Bitmaps.
 */
class roaring
{
private:
    using roaring_type = ::roaring::Roaring64Map;
    roaring_type roaring_bm_;

public:
    using value_type = uint64_t;
    using size_type = uint64_t;

    // Construction and assignment
    roaring() = default;
    roaring(const roaring& other) = default;
    roaring(roaring&& other) noexcept = default;
    roaring& operator=(const roaring& other) = default;
    roaring& operator=(roaring&& other) noexcept = default;

    // Construct from vector of indices
    explicit roaring(const std::vector<uint64_t>& indices)
    {
        for (uint64_t idx : indices) {
            roaring_bm_.add(idx);
        }
    }

    // Construct from iterator range
    template<typename Iterator>
    roaring(Iterator begin, Iterator end)
    {
        for (auto it = begin; it != end; ++it) {
            roaring_bm_.add(static_cast<uint64_t>(*it));
        }
    }

    // Core Roaring operations
    void add(uint64_t value)
    {
        roaring_bm_.add(value);
    }

    void remove(uint64_t value)
    {
        roaring_bm_.remove(value);
    }

    bool contains(uint64_t value) const
    {
        return roaring_bm_.contains(value);
    }

    uint64_t cardinality() const
    {
        return roaring_bm_.cardinality();
    }

    bool is_empty() const
    {
        return roaring_bm_.isEmpty();
    }

    uint64_t minimum() const
    {
        return roaring_bm_.minimum();
    }

    uint64_t maximum() const
    {
        return roaring_bm_.maximum();
    }

    void add_range(uint64_t min, uint64_t max)
    {
        roaring_bm_.addRange(min, max);
    }

    void remove_range(uint64_t min, uint64_t max)
    {
        roaring_bm_.removeRange(min, max);
    }

    void flip(uint64_t min, uint64_t max)
    {
        roaring_bm_.flip(min, max);
    }

    void clear()
    {
        roaring_bm_.clear();
    }

    // Iterator support
    auto begin() const -> decltype(roaring_bm_.begin())
    {
        return roaring_bm_.begin();
    }

    auto end() const -> decltype(roaring_bm_.end())
    {
        return roaring_bm_.end();
    }

    // Bitwise operations
    roaring& operator&=(const roaring& other)
    {
        roaring_bm_ &= other.roaring_bm_;
        return *this;
    }

    roaring& operator|=(const roaring& other)
    {
        roaring_bm_ |= other.roaring_bm_;
        return *this;
    }

    roaring& operator^=(const roaring& other)
    {
        roaring_bm_ ^= other.roaring_bm_;
        return *this;
    }

    roaring& operator-=(const roaring& other)
    {
        roaring_bm_ -= other.roaring_bm_;
        return *this;
    }

    // Non-member bitwise operations
    friend roaring operator&(const roaring& lhs, const roaring& rhs)
    {
        roaring result = lhs;
        result &= rhs;
        return result;
    }

    friend roaring operator|(const roaring& lhs, const roaring& rhs)
    {
        roaring result = lhs;
        result |= rhs;
        return result;
    }

    friend roaring operator^(const roaring& lhs, const roaring& rhs)
    {
        roaring result = lhs;
        result ^= rhs;
        return result;
    }

    friend roaring operator-(const roaring& lhs, const roaring& rhs)
    {
        roaring result = lhs;
        result -= rhs;
        return result;
    }

    // Comparison operators
    bool operator==(const roaring& other) const
    {
        return roaring_bm_ == other.roaring_bm_;
    }

    bool operator!=(const roaring& other) const
    {
        return roaring_bm_ != other.roaring_bm_;
    }

    // Memory usage
    std::size_t get_size_in_bytes() const
    {
        return roaring_bm_.getSizeInBytes();
    }

    // Serialization
    void serialize(std::vector<char>& buffer) const
    {
        std::size_t size = roaring_bm_.getSizeInBytes();
        buffer.resize(size);
        roaring_bm_.write(buffer.data());
    }

    static roaring deserialize(const std::vector<char>& buffer)
    {
        roaring result;
        if (!buffer.empty()) {
            result.roaring_bm_ = roaring_type::readSafe(buffer.data(), buffer.size());
        }
        return result;
    }

    // Conversion to/from std::vector
    std::vector<uint64_t> to_vector() const
    {
        std::vector<uint64_t> result;
        result.reserve(roaring_bm_.cardinality());
        for (auto value : roaring_bm_) {
            result.push_back(value);
        }
        return result;
    }

    void from_vector(const std::vector<uint64_t>& values)
    {
        roaring_bm_.clear();
        for (uint64_t value : values) {
            roaring_bm_.add(value);
        }
    }

    // Direct access to underlying Roaring bitmap
    const roaring_type& get_roaring() const
    {
        return roaring_bm_;
    }

    roaring_type& get_roaring()
    {
        return roaring_bm_;
    }
};

} // namespace icm
