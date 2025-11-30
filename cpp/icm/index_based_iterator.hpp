#pragma once

#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace icm {
class use_container_index_tag
{
};
/**
 * @brief Generic iterator template for the container which have index based random access.
 *
 * @tparam C Container type.
 * @tparam V Value type.
 * @tparam F Functor which gets container and index as an input and returns value type. Usually it's `C::operator[]`
 * @tparam I Index type.
 */
template <typename C, typename V, typename F, typename I = int64_t>
requires(std::is_invocable_v<F, const C&, I> && std::is_same_v<std::invoke_result_t<F, const C&, I>, V>) ||
        std::is_same_v<F, use_container_index_tag>
class index_based_iterator
{
public:
    using value_type = V;
    using reference = V&;
    using pointer = std::remove_reference_t<V>*;
    using difference_type = I;
    using iterator_category = std::random_access_iterator_tag;

    index_based_iterator() = default;

    inline index_based_iterator(const C& c, int64_t index, F f)
        : container_(&c)
        , index_(index)
        , functor_(std::move(f))
    {
    }

    inline index_based_iterator(const C& c, int64_t index)
        : container_(&c)
        , index_(index)
    {
        static_assert(std::is_same_v<F, use_container_index_tag>, "Functor is not provided");
    }

    inline value_type operator*() const
    {
        if constexpr (std::is_same_v<use_container_index_tag, F>) {
            return (*container_)[index_];
        } else {
            return functor_(*container_, index_);
        }
    }

    inline value_type operator->() const
    {
        if constexpr (std::is_same_v<use_container_index_tag, F>) {
            return (*container_)[index_];
        } else {
            return functor_(*container_, index_);
        }
    }

    inline index_based_iterator& operator++()
    {
        ++index_;
        return *this;
    }

    inline index_based_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    inline index_based_iterator& operator+=(difference_type step)
    {
        index_ += step;
        return *this;
    }

    inline index_based_iterator operator+(difference_type step) const
    {
        auto tmp = *this;
        tmp += step;
        return tmp;
    }

    inline index_based_iterator& operator--()
    {
        --index_;
        return *this;
    }

    inline index_based_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    inline index_based_iterator& operator-=(difference_type step)
    {
        index_ -= step;
        return *this;
    }

    inline index_based_iterator operator-(difference_type step) const
    {
        auto tmp = *this;
        tmp -= step;
        return tmp;
    }

    inline difference_type operator-(const index_based_iterator& it) const
    {
        return index_ - it.index_;
    }

    inline bool operator==(const index_based_iterator& rhs) const
    {
        return container_ == rhs.container_ && index_ == rhs.index_;
    }

private:
    const C* container_ = nullptr;
    I index_ = I();
    F functor_;
};

template <typename C, typename V, typename F, typename I = int64_t>
requires(std::is_invocable_v<F, C&, I> && std::is_same_v<std::invoke_result_t<F, C&, I>, V>) ||
        std::is_same_v<F, use_container_index_tag>
class mutable_index_based_iterator
{
public:
    using value_type = V;
    using reference = V&;
    using pointer = std::remove_reference_t<V>*;
    using difference_type = I;
    using iterator_category = std::random_access_iterator_tag;

    inline mutable_index_based_iterator() = default;
    
    inline mutable_index_based_iterator(C& c, int64_t index, F f)
        : container_(&c), index_(index), functor_(std::move(f))
    {
    }
    inline mutable_index_based_iterator(C& c, int64_t index) : container_(&c), index_(index)
    {
        static_assert(std::is_same_v<F, use_container_index_tag>, "Functor is not provided");
    }

    inline value_type operator*() const
    {
        if constexpr (std::is_same_v<use_container_index_tag, F>) {
            return (*container_)[index_];
        } else {
            return functor_(*container_, index_);
        }
    }

    inline value_type operator->()
    {
        if constexpr (std::is_same_v<use_container_index_tag, F>) {
            return (*container_)[index_];
        } else {
            return functor_(*container_, index_);
        }
    }

    inline mutable_index_based_iterator& operator++()
    {
        ++index_;
        return *this;
    }

    inline mutable_index_based_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    inline mutable_index_based_iterator& operator+=(difference_type step)
    {
        index_ += step;
        return *this;
    }

    inline mutable_index_based_iterator operator+(difference_type step) const
    {
        auto tmp = *this;
        tmp += step;
        return tmp;
    }

    inline mutable_index_based_iterator& operator--()
    {
        --index_;
        return *this;
    }

    inline mutable_index_based_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    inline mutable_index_based_iterator& operator-=(difference_type step)
    {
        index_ -= step;
        return *this;
    }

    inline mutable_index_based_iterator operator-(difference_type step) const
    {
        auto tmp = *this;
        tmp -= step;
        return tmp;
    }

    inline difference_type operator-(const mutable_index_based_iterator& it) const
    {
        return index_ - it.index_;
    }

    inline bool operator==(const mutable_index_based_iterator& rhs) const
    {
        return container_ == rhs.container_ && index_ == rhs.index_;
    }

private:
    C* container_ = nullptr;
    I index_ = I();
    F functor_;
};

} // namespace icm
