#pragma once

/**
 * @file ndenumerate.hpp
 * @brief Definition and implementation of `ndenumerate` utility.
 */
#include <algorithm>
namespace nd {

class coord_iterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::vector<uint32_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const std::vector<uint32_t>*;
    using reference = const std::vector<uint32_t>&;

public:
    explicit coord_iterator(const std::vector<uint32_t>& data)
        : data_(data)
        , state_(data.size(), 0)
        , end_(std::ranges::find(data_, 0) != data_.end())
    {}

    explicit coord_iterator(std::vector<uint32_t>&& data)
        : data_(std::move(data))
        , state_(data_.size(), 0)
        , end_(std::ranges::find(data_, 0) != data_.end())
    {}

    reference operator*() const
    {
        return state_;
    }

    pointer operator->() const
    {
        return &state_;
    }

    coord_iterator& operator++()
    {
        int pos = state_.size();
        while (--pos >= 0) {
            if (++state_[pos] != data_[pos]) {
                break;
            }
            state_[pos] = 0;
        }
        end_ = pos < 0;
        return *this;
    }

    bool operator==(const coord_iterator& other) const
    {
        return end_ == other.end_ && state_ == other.state_;
    }

    bool operator!=(const coord_iterator& other) const
    {
        return !(*this == other);
    }

    void jump_to_end()
    {
        end_ = true;
    }

public:
    const std::vector<uint32_t> data_;
    std::vector<uint32_t> state_;
    bool end_;
};

inline auto ndenumerate(const std::vector<uint32_t>& iterable)
{
    struct coord_iterator_wrapper
    {
        std::vector<uint32_t> iterable;

        [[nodiscard]] auto begin() const
        {
            return coord_iterator(iterable);
        }

        [[nodiscard]] auto end() const
        {
            coord_iterator it(iterable);
            it.jump_to_end();
            return it;
        }
    };

    return coord_iterator_wrapper{iterable};
}

} // namespace nd