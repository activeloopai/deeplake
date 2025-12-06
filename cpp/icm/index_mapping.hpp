#pragma once

/**
 * @file index_mapping.hpp
 * @brief Definition of the index_mapping_t class
 */

#include "index_based_iterator.hpp"
#include "small_vector.hpp"
#include "flat_iterator.hpp"

#include <base/assert.hpp>
#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <variant>
#include <vector>

namespace icm {

namespace impl {

template <typename I>
struct other_int;

template <>
struct other_int<int32_t>
{
    using type = int64_t;
};

template <>
struct other_int<int64_t>
{
    using type = int32_t;
};

template <>
struct other_int<uint32_t>
{
    using type = uint64_t;
};

template <>
struct other_int<uint64_t>
{
    using type = uint32_t;
};

} // namespace impl

/**
 * @class index_mapping_t
 * @brief Class representing a mapping of indices
 * @tparam T Type of the indices
 */
template <typename T>
class index_mapping_t
{
private:
    using other_int_t = typename impl::other_int<T>::type;

public:
    using value_type = T;
    class iterator;
    using const_iterator = iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;

private:
    class merged_index_mapping;
    using variant_t = std::variant<std::monostate,
                                   T,
                                   std::array<T, 3>,
                                   std::shared_ptr<std::vector<T>>,
                                   std::shared_ptr<std::pair<index_mapping_t, index_mapping_t>>,
                                   T,
                                   std::shared_ptr<index_mapping_t<other_int_t>>,
                                   std::array<T, 2>,
                                   std::shared_ptr<merged_index_mapping>>;

public:
    index_mapping_t()
        : state_(std::in_place_index<0>)
    {
    }

    static index_mapping_t single_index(T index)
    {
        return index_mapping_t(variant_t(std::in_place_index<5>, index));
    }

    static index_mapping_t trivial(T size)
    {
        return index_mapping_t(size);
    }

    static index_mapping_t slice(const std::array<T, 3>& indices)
    {
        return index_mapping_t(indices);
    }

    static index_mapping_t repeated(T value, T size)
    {
        return index_mapping_t(std::array<T, 2>{value, size});
    }

    static index_mapping_t list(std::vector<T>&& indices)
    {
        if (indices.empty()) {
            return index_mapping_t();
        }
        return index_mapping_t(std::move(indices));
    }

    static index_mapping_t list(const std::vector<T>& indices)
    {
        if (indices.empty()) {
            return index_mapping_t();
        }
        return index_mapping_t(indices);
    }

    static index_mapping_t list(std::shared_ptr<std::vector<T>> indices)
    {
        return index_mapping_t(std::move(indices));
    }

    static index_mapping_t cast(index_mapping_t<other_int_t>&& index)
    {
        return index_mapping_t(std::move(index));
    }

    static index_mapping_t cast(index_mapping_t&& index)
    {
        return index_mapping_t(std::move(index));
    }

    static index_mapping_t chain(index_mapping_t&& first, index_mapping_t&& second)
    {
        if (second.is_single_index()) {
            return index_mapping_t::single_index(first[second[0]]);
        }
        return index_mapping_t(std::move(first), std::move(second));
    }

    static index_mapping_t chain(const index_mapping_t& first, const index_mapping_t& second)
    {
        if (second.is_single_index()) {
            return index_mapping_t::single_index(first[second[0]]);
        }
        return index_mapping_t(first, second);
    }

    static index_mapping_t merge(std::vector<index_mapping_t>&& indices)
    {
        if (indices.empty()) {
            return index_mapping_t();
        }
        if (indices.size() == 1) {
            return std::move(indices[0]);
        }
        return index_mapping_t(std::make_shared<merged_index_mapping>(std::move(indices)));
    }

public:
    T size() const;

    bool empty() const
    {
        return size() == 0;
    }

    T operator[](T new_index) const;

    std::vector<T> indices() const;

    iterator begin() const;
    iterator end() const;
    iterator cbegin() const;
    iterator cend() const;

    reverse_iterator rbegin() const
    {
        return reverse_iterator(end());
    }

    reverse_iterator rend() const
    {
        return reverse_iterator(begin());
    }

public:
    icm::json to_json() const;

    static index_mapping_t from_json(const icm::const_json& j);

public:
    explicit operator bool() const noexcept
    {
        return state_.index() != 0;
    }

    bool is_single_index() const noexcept
    {
        return state_.index() == 5;
    }

    bool is_trivial() const noexcept
    {
        return state_.index() == 1;
    }

    bool is_slice() const noexcept
    {
        return state_.index() == 2;
    }

    bool is_repeated() const noexcept
    {
        return state_.index() == 7;
    }

    bool is_list() const noexcept
    {
        return state_.index() == 3;
    }

    bool is_chain() const noexcept
    {
        return state_.index() == 4;
    }

    bool is_cast() const noexcept
    {
        return state_.index() == 6;
    }

    bool is_merged() const noexcept
    {
        return state_.index() == 8;
    }

public:
    T single_index() const
    {
        ASSERT(state_.index() == 5);
        return std::get<5>(state_);
    }

    T trivial() const
    {
        ASSERT(state_.index() == 1);
        return std::get<1>(state_);
    }

    std::array<T, 3> slice() const
    {
        ASSERT(state_.index() == 2);
        return std::get<2>(state_);
    }

    std::array<T, 2> repeated() const
    {
        ASSERT(state_.index() == 7);
        return std::get<7>(state_);
    }

    const std::vector<T>& list() const
    {
        ASSERT(state_.index() == 3);
        return *std::get<3>(state_);
    }

    const index_mapping_t<other_int_t>& cast() const
    {
        ASSERT(state_.index() == 6);
        return *std::get<6>(state_);
    }

    const std::pair<index_mapping_t, index_mapping_t>& chain() const
    {
        ASSERT(state_.index() == 4);
        return *std::get<4>(state_);
    }

    const merged_index_mapping& merged() const
    {
        ASSERT(state_.index() == 8);
        return *std::get<8>(state_);
    }

private:
    explicit index_mapping_t(T size)
        : state_(std::in_place_index<1>, size)
    {
    }

    explicit index_mapping_t(std::array<T, 3>&& indices)
        : state_(std::in_place_index<2>, std::move(indices))
    {
    }

    explicit index_mapping_t(const std::array<T, 3>& indices)
        : state_(std::in_place_index<2>, indices)
    {
    }

    explicit index_mapping_t(const std::array<T, 2>& indices)
        : state_(std::in_place_index<7>, indices)
    {
    }

    explicit index_mapping_t(std::vector<T>&& indices)
        : state_(std::in_place_index<3>, std::make_shared<std::vector<T>>(std::move(indices)))
    {
    }

    explicit index_mapping_t(const std::vector<T>& indices)
        : state_(std::in_place_index<3>, std::make_shared<std::vector<T>>(indices))
    {
    }

    explicit index_mapping_t(std::shared_ptr<std::vector<T>> indices)
        : state_(std::in_place_index<3>, std::move(indices))
    {
    }

    explicit index_mapping_t(index_mapping_t<other_int_t>&& index)
        : state_(std::in_place_index<6>, std::make_shared<index_mapping_t<other_int_t>>(std::move(index)))
    {
    }

    index_mapping_t(index_mapping_t&& first, index_mapping_t&& second)
        : state_(first && second
                     ? variant_t(std::in_place_index<4>, std::make_shared<std::pair<index_mapping_t, index_mapping_t>>(
                                                             std::move(first), std::move(second)))
                 : first ? std::move(first.state_)
                         : std::move(second.state_))
    {
    }

    index_mapping_t(const index_mapping_t& first, const index_mapping_t& second)
        : state_(std::in_place_index<4>, std::make_shared<std::pair<index_mapping_t, index_mapping_t>>(first, second))
    {
    }

    explicit index_mapping_t(std::shared_ptr<merged_index_mapping> indices)
        : state_(std::in_place_index<8>, std::move(indices))
    {
    }

    explicit index_mapping_t(variant_t&& state)
        : state_(std::move(state))
    {
    }

private:
    variant_t state_;
};

template <typename T>
using index_mapping_vector_t = icm::small_vector<index_mapping_t<T>>;

using index_mapping = index_mapping_t<int64_t>;
using index_mapping_vector = index_mapping_vector_t<int64_t>;

template <typename T>
class index_mapping_t<T>::iterator
{
public:
    using value_type = T;
    using reference = T;
    using pointer = T*;
    using difference_type = int64_t;
    using iterator_category = std::random_access_iterator_tag;

    iterator() = default;

    iterator(const index_mapping_t& mapping, T index)
        : iterator_(std::in_place_index<0>,
                    index_based_iterator<index_mapping_t, T, use_container_index_tag, T>(mapping, index))
    {
    }

    iterator(std::unique_ptr<typename merged_index_mapping::iterator> iterator)
        : iterator_(std::move(iterator))
    {
    }

    iterator(const std::vector<T>& indices, T index)
        : iterator_(std::in_place_index<2>, vector_iterator_wrapper{indices.begin() + index, &indices})
    {
    }

    iterator(const iterator& other)
        : iterator_(std::visit(
              []<typename It>(const It& arg) -> iterator_type {
                  if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                      return std::make_unique<typename merged_index_mapping::iterator>(*arg);
                  } else {
                      return arg;
                  }
              },
              other.iterator_))
    {
    }

    iterator& operator=(const iterator& other)
    {
        iterator_ = std::visit(
            []<typename It>(const It& arg) -> iterator_type {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    return std::make_unique<typename merged_index_mapping::iterator>(*arg);
                } else {
                    return arg;
                }
            },
            other.iterator_);
        return *this;
    }

    iterator(iterator&& other) noexcept = default;
    iterator& operator=(iterator&& other) noexcept = default;

    value_type operator*() const
    {
        return std::visit(
            []<typename It>(const It& arg) -> value_type {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    return **arg;
                } else {
                    return *arg;
                }
            },
            iterator_);
    }

    iterator& operator++()
    {
        std::visit(
            []<typename It>(It& arg) {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    ++*arg;
                } else {
                    ++arg;
                }
            },
            iterator_);
        return *this;
    }

    iterator operator++(int)
    {
        iterator tmp = *this;
        ++*this;
        return tmp;
    }

    iterator& operator--()
    {
        std::visit(
            []<typename It>(It& arg) {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    --*arg;
                } else {
                    --arg;
                }
            },
            iterator_);
        return *this;
    }

    iterator operator--(int)
    {
        iterator tmp = *this;
        --*this;
        return tmp;
    }

    iterator& operator+=(difference_type n)
    {
        std::visit(
            [n]<typename It>(It& arg) {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    *arg += n;
                } else {
                    arg += n;
                }
            },
            iterator_);
        return *this;
    }

    iterator operator+(difference_type n) const
    {
        iterator tmp = *this;
        return tmp += n;
    }

    iterator& operator-=(difference_type n)
    {
        std::visit(
            [n]<typename It>(It& arg) {
                if constexpr (std::is_same_v<It, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                    *arg -= n;
                } else {
                    arg -= n;
                }
            },
            iterator_);
        return *this;
    }

    iterator operator-(difference_type n) const
    {
        iterator tmp = *this;
        return tmp -= n;
    }

    difference_type operator-(const iterator& other) const
    {
        return std::visit(
            [&other]<typename It1, typename It2>(const It1& arg1, const It2& arg2) -> difference_type {
                if constexpr (std::is_same_v<It1, It2>) {
                    if constexpr (std::is_same_v<It1, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                        return *arg1 - *arg2;
                    } else {
                        return arg1 - arg2;
                    }
                } else {
                    ASSERT_MESSAGE(false, "Cannot subtract different types of iterators");
                    return difference_type();
                }
            },
            iterator_,
            other.iterator_);
    }

    bool operator==(const iterator& other) const
    {
        return std::visit(
            [&other]<typename It1, typename It2>(const It1& arg1, const It2& arg2) {
                if constexpr (std::is_same_v<It1, It2>) {
                    if constexpr (std::is_same_v<It1, std::unique_ptr<typename merged_index_mapping::iterator>>) {
                        return *arg1 == *arg2;
                    } else {
                        return arg1 == arg2;
                    }
                } else {
                    return false;
                }
            },
            iterator_, other.iterator_);
    }

    bool operator!=(const iterator& other) const
    {
        return !(*this == other);
    }

private:
    struct vector_iterator_wrapper
    {
        typename std::vector<T>::const_iterator iter;
        const void* container_id;

        bool operator==(const vector_iterator_wrapper& other) const
        {
            return container_id == other.container_id && iter == other.iter;
        }

        bool operator!=(const vector_iterator_wrapper& other) const
        {
            return !(*this == other);
        }

        vector_iterator_wrapper& operator++()
        {
            ++iter;
            return *this;
        }

        vector_iterator_wrapper& operator--()
        {
            --iter;
            return *this;
        }

        vector_iterator_wrapper& operator+=(difference_type n)
        {
            iter += n;
            return *this;
        }

        vector_iterator_wrapper& operator-=(difference_type n)
        {
            iter -= n;
            return *this;
        }

        difference_type operator-(const vector_iterator_wrapper& other) const
        {
            ASSERT_MESSAGE(container_id == other.container_id, "Cannot subtract iterators from different containers");
            return iter - other.iter;
        }

        T operator*() const
        {
            return *iter;
        }
    };

public:
    using iterator_type = std::variant<index_based_iterator<index_mapping_t, T, use_container_index_tag, T>,
                                       std::unique_ptr<typename merged_index_mapping::iterator>,
                                       vector_iterator_wrapper>;

private:
    iterator_type iterator_;
};

template <typename T>
struct index_mapping_t<T>::merged_index_mapping
{
    merged_index_mapping(std::vector<index_mapping_t>&& indices)
        : indices_(std::move(indices))
    {
        ASSERT(indices_.size() > 1);
        offsets_.reserve(indices_.size());
        std::transform_exclusive_scan(indices_.begin(),
                                      indices_.end(),
                                      std::back_inserter(offsets_),
                                      static_cast<T>(0),
                                      std::plus<T>(),
                                      [](const index_mapping_t& i) {
                                          return i.size();
                                      });
    }

    using iterator = flat_iterator<index_mapping_t>;

    iterator begin() const;

    iterator end() const;

    T size() const
    {
        ASSERT(indices_.size() > 1);
        return offsets_.back() + indices_.back().size();
    }

    T operator[](T index) const
    {
        auto it = std::upper_bound(offsets_.begin(), offsets_.end(), index) - 1;
        auto offset = *it;
        auto i = std::distance(offsets_.begin(), it);
        return indices_[i][index - offset];
    }

    std::vector<index_mapping_t> indices_;
    std::vector<T> offsets_;
};

template <typename T>
inline typename index_mapping_t<T>::iterator index_mapping_t<T>::begin() const
{
    if (is_merged()) {
        return std::make_unique<typename merged_index_mapping::iterator>(merged().begin());
    }
    if (is_list()) {
        return iterator(list(), 0);
    }
    return index_mapping_t<T>::iterator(*this, 0);
}

template <typename T>
inline typename index_mapping_t<T>::iterator index_mapping_t<T>::end() const
{
    if (is_merged()) {
        return std::make_unique<typename merged_index_mapping::iterator>(merged().end());
    }
    if (is_list()) {
        return iterator(list(), size());
    }
    return index_mapping_t<T>::iterator(*this, size());
}

template <typename T>
inline typename index_mapping_t<T>::iterator index_mapping_t<T>::cbegin() const
{
    return begin();
}

template <typename T>
inline typename index_mapping_t<T>::iterator index_mapping_t<T>::cend() const
{
    return end();
}

template <typename T>
inline index_mapping_t<T>::merged_index_mapping::iterator index_mapping_t<T>::merged_index_mapping::begin() const
{
    return iterator::begin(indices_, offsets_);
}

template <typename T>
inline index_mapping_t<T>::merged_index_mapping::iterator index_mapping_t<T>::merged_index_mapping::end() const
{
    return iterator::end(indices_, offsets_);
}

} // namespace icm
