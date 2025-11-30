#pragma once

/**
 * @file colony.hpp
 * @brief Definition and implementation of `colony` template class.
 */

#include <base/assert.hpp>

#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <deque>
#include <variant>

namespace icm {

template <typename T, typename id_type = int>
class colony
{
    static_assert(std::is_signed_v<id_type>);

public:
    colony() = default;

    colony(const colony&) = default;
    colony& operator=(const colony&) = default;

    colony(colony&& other) noexcept
        : elements_(std::move(other.elements_))
        , size_(other.size_)
        , first_(other.first_)
    {
        other.size_ = 0;
        other.first_ = 0;
    }

    colony& operator=(colony&& other) noexcept
    {
        if (this != &other) {
            elements_ = std::move(other.elements_);
            size_ = other.size_;
            first_ = other.first_;
            other.size_ = 0;
            other.first_ = 0;
        }
        return *this;
    }

    inline id_type next_position() const
    {
        if (first_ >= 0) {
            return elements_.size();
        }
        return -first_ - 1;
    }

    template <typename ... Args>
    inline id_type emplace(Args&& ... args)
    {
        ++size_;
        if (first_ >= 0) {
            elements_.emplace_back(std::in_place_index<0>, std::forward<Args>(args) ...);
            return elements_.size() - 1;
        } else {
            int i = -first_ - 1;
            first_ = std::get<1>(elements_[i]);
            elements_[i].template emplace<0>(std::forward<Args>(args) ...);
            return i;
        }
    }

    inline id_type insert(T&& t)
    {
        return emplace(std::move(t));
    }

    inline void erase(id_type id)
    {
        ASSERT(id >= 0 && id < elements_.size());
        ASSERT(elements_[id].index() == 0);
        elements_[id].template emplace<1>(first_);
        --size_;
        first_ = -id - 1;
    }

    inline void clear()
    {
        elements_.clear();
        first_ = 0;
        size_ = 0;
    }

    inline auto size() const noexcept
    {
        return size_;
    }

    inline bool empty() const noexcept
    {
        return size_ == 0;
    }

    inline const T& operator[](id_type id) const
    {
        ASSERT(id >= 0 && id < elements_.size());
        ASSERT(elements_[id].index() == 0);
        return std::get<0>(elements_[id]);
    }

    inline T& operator[](id_type id)
    {
        ASSERT(id >= 0 && id < elements_.size());
        ASSERT(elements_[id].index() == 0);
        return std::get<0>(elements_[id]);
    }

    inline const T& front() const
    {
        return const_cast<colony*>(this)->front();
    }

    inline T& front()
    {
        int i = 0;
        while (elements_[i].index() == 1) {
            ++i;
        }
        return std::get<0>(elements_[i]);
    }

    inline const T& back() const
    {
        return const_cast<colony*>(this)->back();
    }

    inline T& back()
    {
        auto i = elements_.size() - 1;
        while (elements_[i].index() == 1) {
            --i;
        }
        return std::get<0>(elements_[i]);
    }

    inline bool contains(int i) const noexcept
    {
        return i < elements_.size() && elements_[i].index() == 0;
    }

private:
    using element_t = std::variant<T, int>;
    using container_t = std::deque<element_t>;

    static inline bool is_used(const element_t& e) noexcept
    {
        return e.index() == 0;
    }

    static inline T& get_value(element_t& e)
    {
        return std::get<0>(e);
    }

    static inline const T& get_const_value(const element_t& e)
    {
        return std::get<0>(e);
    }

    using filter_iterator = boost::iterators::filter_iterator<decltype(&is_used), typename container_t::iterator>;
    using const_filter_iterator =
        boost::iterators::filter_iterator<decltype(&is_used), typename container_t::const_iterator>;

public:
    using iterator = boost::iterators::transform_iterator<decltype(&get_value), filter_iterator>;
    using const_iterator = boost::iterators::transform_iterator<decltype(&get_const_value), const_filter_iterator>;

    inline iterator begin()
    {
        return boost::iterators::make_transform_iterator(
            boost::iterators::make_filter_iterator(&is_used, elements_.begin(), elements_.end()), &get_value);
    }

    inline iterator end()
    {
        return boost::iterators::make_transform_iterator(
            boost::iterators::make_filter_iterator(&is_used, elements_.end(), elements_.end()), &get_value);
    }

    inline const_iterator begin() const
    {
        return boost::iterators::make_transform_iterator(
            boost::iterators::make_filter_iterator(&is_used, elements_.begin(), elements_.end()), &get_const_value);
    }

    inline const_iterator end() const
    {
        return boost::iterators::make_transform_iterator(
            boost::iterators::make_filter_iterator(&is_used, elements_.end(), elements_.end()), &get_const_value);
    }

    inline const_iterator cbegin() const
    {
        return begin();
    }

    inline const_iterator cend() const
    {
        return end();
    }

private:
    container_t elements_;
    int size_ = 0;
    int first_ = 0;
};

} // namespace icm
