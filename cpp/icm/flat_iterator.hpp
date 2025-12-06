#pragma once

#include <base/assert.hpp>

#include <cstdint>
#include <vector>

namespace icm {

/**
 * @brief A random access iterator that iterates over a vector of containers.
 * `flat_iterator` Gets a vector of containers and a vector of offsets, and iterates over the containers as if they were
 * a single container. The offsets vector contains the offset of each container in the flattened container, it can be
 * generated using `std::transform_exclusive_scan`.
 * The vector of containers can be empty
 * The vector of offsets should have the same size as the vector of containers.
 * It's allowed to have empty containers in the vector of containers.
 */
template <typename C>
class flat_iterator
{
public:
    using value_type = typename C::const_iterator::value_type;
    using reference = typename C::const_iterator::reference;
    using pointer = value_type*;
    using difference_type = int64_t;
    using iterator_category = std::random_access_iterator_tag;

    static flat_iterator begin(const std::vector<C>& containers, const std::vector<value_type>& offsets)
    {
        if (containers.empty()) {
            return flat_iterator();
        }
        return flat_iterator(containers, offsets, containers.begin(), containers.begin()->begin());
    }

    static flat_iterator end(const std::vector<C>& containers, const std::vector<value_type>& offsets)
    {
        if (containers.empty()) {
            return flat_iterator();
        }
        return flat_iterator(containers, offsets, containers.end() - 1, containers.back().end());
    }

    reference operator*() const
    {
        return *child_iterator_;
    }

    flat_iterator& operator++()
    {
        ++child_iterator_;
        skip_empty_containers();
        return *this;
    }

    flat_iterator operator++(int)
    {
        flat_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    flat_iterator& operator--()
    {
        while (child_iterator_ == parent_iterator_->begin() && parent_iterator_ != containers_->begin()) {
            --parent_iterator_;
            child_iterator_ = parent_iterator_->end();
        }
        --child_iterator_;
        return *this;
    }

    flat_iterator operator--(int)
    {
        flat_iterator tmp = *this;
        --*this;
        return tmp;
    }

    flat_iterator& operator+=(difference_type n)
    {
        if (n > 0) {
            while (n > 0) {
                auto remaining = std::distance(child_iterator_, parent_iterator_->end());
                if (n <= remaining) {
                    child_iterator_ += n;
                    break;
                }
                n -= remaining;
                ++parent_iterator_;
                child_iterator_ = parent_iterator_->begin();
            }
        } else {
            while (n < 0) {
                auto remaining = std::distance(parent_iterator_->begin(), child_iterator_);
                if (n >= -remaining) {
                    child_iterator_ += n;
                    break;
                }
                n += remaining;
                --parent_iterator_;
                child_iterator_ = parent_iterator_->end();
            }
        }
        skip_empty_containers();
        return *this;
    }

    flat_iterator operator+(difference_type n) const
    {
        flat_iterator tmp = *this;
        tmp += n;
        return tmp;
    }

    friend flat_iterator operator+(difference_type n, const flat_iterator& it)
    {
        return it + n;
    }

    flat_iterator& operator-=(difference_type n)
    {
        return *this += -n;
    }

    flat_iterator operator-(difference_type n) const
    {
        flat_iterator tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const flat_iterator& other) const
    {
        if (containers_ == nullptr) {
            ASSERT(other.containers_ == nullptr);
            return 0;
        }
        difference_type n = 0;
        auto it = *this;
        while (std::distance(it.parent_iterator_, other.parent_iterator_) > 0) {
            n += std::distance(it.parent_iterator_->end(), it.child_iterator_);
            ++it.parent_iterator_;
            it.child_iterator_ = it.parent_iterator_->begin();
        }
        while (std::distance(it.parent_iterator_, other.parent_iterator_) < 0) {
            n += std::distance(it.parent_iterator_->begin(), it.child_iterator_);
            --it.parent_iterator_;
            it.child_iterator_ = it.parent_iterator_->end();
        }
        n += std::distance(other.child_iterator_, it.child_iterator_);
        return n;
    }

    bool operator==(const flat_iterator& other) const
    {
        return child_iterator_ == other.child_iterator_;
    }

    bool operator!=(const flat_iterator& other) const
    {
        return !(*this == other);
    }

private:
    flat_iterator() = default;

    flat_iterator(const std::vector<C>& containers,
                  const std::vector<value_type>& offsets,
                  std::vector<C>::const_iterator parent_iterator,
                  C::const_iterator child_iterator)
        : containers_(&containers)
        , offsets_(&offsets)
        , parent_iterator_(parent_iterator)
        , child_iterator_(child_iterator)
    {
        skip_empty_containers();
    }

    void skip_empty_containers()
    {
        while (parent_iterator_ != containers_->end() - 1 && child_iterator_ == (*parent_iterator_).end()) {
            ++parent_iterator_;
            child_iterator_ = parent_iterator_->begin();
        }
    }

    const std::vector<C>* containers_;
    const std::vector<value_type>* offsets_;
    std::vector<C>::const_iterator parent_iterator_;
    C::const_iterator child_iterator_;
};

}
