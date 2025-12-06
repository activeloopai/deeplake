#pragma once

#include "../dataset_view.hpp"

#include <iterator>

namespace heimdall {

class column_view;
class dataset_view;

namespace impl {

class dataset_iterator
{
public:
    using value_type = column_view;
    using reference_type = value_type&;
    using pointer_type = value_type*;
    using difference_type = int;
    using iterator_category = std::random_access_iterator_tag;

public:
    dataset_iterator() = default;
    explicit dataset_iterator(dataset_view& ds)
        : dataset_iterator(ds, 0)
    {}

    dataset_iterator(dataset_view& ds, int position)
        : ds_(&ds)
        , position_(position)
    {}

public:
    inline reference_type operator*() const
    {
        return (*ds_)[position_];
    }

    inline pointer_type operator->()
    {
        return &(*ds_)[position_];
    }

    inline dataset_iterator& operator++()
    {
        ++position_;
        return *this;
    }

    inline dataset_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    inline dataset_iterator& operator+=(difference_type step)
    {
        position_ += step;
        return *this;
    }

    dataset_iterator operator+(difference_type step) const
    {
        auto tmp = *this;
        tmp += step;
        return tmp;
    }

    inline dataset_iterator& operator--()
    {
        --position_;
        return *this;
    }

    inline dataset_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    inline dataset_iterator& operator-=(difference_type step)
    {
        position_ -= step;
        return *this;
    }

    inline dataset_iterator operator-(difference_type step) const
    {
        auto tmp = *this;
        tmp -= step;
        return tmp;
    }

    inline difference_type operator-(const dataset_iterator& it) const
    {
        return position_ - it.position_;
    }

    inline friend bool operator==(const dataset_iterator& a, const dataset_iterator& b)
    {
        return a.ds_ == b.ds_ && a.position_ == b.position_;
    }

    inline friend bool operator!=(const dataset_iterator& a, const dataset_iterator& b)
    {
        return a.ds_ != b.ds_ || a.position_ != b.position_;
    }


private:
    dataset_view* ds_;
    int position_;
};

class const_dataset_iterator
{
public:
    using value_type = column_view;
    using reference_type = const value_type&;
    using pointer_type = const value_type*;
    using difference_type = int;
    using iterator_category = std::random_access_iterator_tag;

public:
    explicit const_dataset_iterator(const dataset_view& ds)
        : const_dataset_iterator(ds, 0)
    {}

    const_dataset_iterator(const dataset_view& ds, int position)
        : ds_(&ds)
        , position_(position)
    {}

public:
    reference_type operator*() const
    {
        return (*ds_)[position_];
    }

    pointer_type operator->() const
    {
        return &(*ds_)[position_];
    }

    inline const_dataset_iterator& operator++()
    {
        ++position_;
        return *this;
    }

    inline const_dataset_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    inline const_dataset_iterator& operator+=(difference_type step)
    {
        position_ += step;
        return *this;
    }

    const_dataset_iterator operator+(difference_type step) const
    {
        auto tmp = *this;
        tmp += step;
        return tmp;
    }

    inline const_dataset_iterator& operator--()
    {
        --position_;
        return *this;
    }

    inline const_dataset_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    inline const_dataset_iterator& operator-=(difference_type step)
    {
        position_ -= step;
        return *this;
    }

    inline const_dataset_iterator operator-(difference_type step) const
    {
        auto tmp = *this;
        tmp -= step;
        return tmp;
    }

    inline difference_type operator-(const const_dataset_iterator& it) const
    {
        return position_ - it.position_;
    }

    inline friend bool operator==(const const_dataset_iterator& a, const const_dataset_iterator& b)
    {
        return a.ds_ == b.ds_ && a.position_ == b.position_;
    }

    inline friend bool operator!=(const const_dataset_iterator& a, const const_dataset_iterator& b)
    {
        return a.ds_ != b.ds_ || a.position_ != b.position_;
    }


private:
    const dataset_view* ds_;
    int position_;
};

}

}

namespace std {

template <>
struct iterator_traits<heimdall::impl::dataset_iterator>
{
    using value_type = heimdall::impl::dataset_iterator::value_type;
    using reference_type = heimdall::impl::dataset_iterator::reference_type;
    using pointer_type = heimdall::impl::dataset_iterator::pointer_type;
    using difference_type = heimdall::impl::dataset_iterator::difference_type;
    using iterator_category = heimdall::impl::dataset_iterator::iterator_category;
};

template <>
struct iterator_traits<heimdall::impl::const_dataset_iterator>
{
    using value_type = heimdall::impl::const_dataset_iterator::value_type;
    using reference_type = heimdall::impl::const_dataset_iterator::reference_type;
    using pointer_type = heimdall::impl::const_dataset_iterator::pointer_type;
    using difference_type = heimdall::impl::const_dataset_iterator::difference_type;
    using iterator_category = heimdall::impl::const_dataset_iterator::iterator_category;
};

}
