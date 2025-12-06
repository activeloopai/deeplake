#pragma once

#include <base/assert.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>

#include <cstdint>
#include <optional>

namespace deeplake_core {

struct batch_column
{
    explicit batch_column(std::string name = std::string())
        : name_(std::move(name))
        , data_(nd::dynamic_empty(nd::dtype::uint8, 0))
    {
    }

    explicit batch_column(int64_t size)
        : data_(nd::dynamic_empty(nd::dtype::uint8, static_cast<uint32_t>(size)))
    {
    }

    batch_column(std::string name, nd::array&& a)
        : name_(std::move(name))
        , data_(std::move(a))
    {
    }

    int64_t size() const noexcept
    {
        return array().shape()[0];
    }

    const nd::array& array() const &
    {
        return data_;
    }

    nd::array array() &&
    {
        return std::move(data_);
    }

    const std::string& name() const
    {
        return name_;
    }

private:
    std::string name_;
    nd::array data_;
};

struct sample;

struct batch
{
public:
    using columns_t = std::vector<batch_column>;

public:
    batch() = default;

    explicit batch(int64_t size)
        : columns_{ batch_column(size) }
    {
    }

    batch(int64_t size, int64_t offset)
        : columns_{ batch_column(size) }
        , offset_(offset)
    {
    }

    batch(std::vector<batch_column>&& columns, int64_t offset)
        : columns_(std::move(columns))
        , offset_(offset)
    {
    }

    const auto& columns() const
    {
        return columns_;
    }

    int64_t offset() const
    {
        return offset_;
    }

    int64_t size() const
    {
        return columns_.empty() ? 0L : columns_.front().size();
    }

    bool empty() const
    {
        return columns_.empty() ? true : column_empty(columns_.front());
    }

    struct iterator;

    iterator begin() const;
    iterator end() const;
    iterator cbegin() const;
    iterator cend() const;

    sample operator[](int64_t row) const;

private:
    bool column_empty(const batch_column& c) const
    {
        return c.array().size() == 0;
    }

    friend struct sample;

private:
    std::vector<batch_column> columns_;
    int64_t offset_ = 0L;
};

struct sample
{
    sample() = default;

    explicit sample(int64_t row)
        : row_(row)
    {
    }

    sample(const batch& b, int64_t row)
        : batch_(&b)
        , row_(row)
    {
    }

    const batch& get_batch() const
    {
        ASSERT(batch_ != nullptr);
        return *batch_;
    }

    auto index() const
    {
        return row_ + (batch_ ? batch_->offset_ : 0L);
    }

    nd::array operator[](int batch_column) const
    {
        ASSERT(batch_ != nullptr);
        return batch_->columns_[batch_column].array()[static_cast<int>(row_)];
    }

private:
    const batch* batch_ = nullptr;
    int64_t row_ = -1;
};

struct batch::iterator
{
    using value_type = sample;
    using reference_type = sample&;
    using pointer_type = sample*;
    using difference_type = int64_t;
    using iterator_category = std::random_access_iterator_tag;

    iterator(const batch& b, int64_t row)
        : batch_(&b)
        , row_(row)
    {}

    iterator(const iterator&) = default;
    iterator& operator=(const iterator&) = default;
    iterator(iterator&&) = default;
    iterator& operator=(iterator&&) = default;
    ~iterator() = default;

    value_type operator*() const
    {
        return sample(*batch_, row_);
    }

    value_type operator->() const
    {
        return sample(*batch_, row_);
    }

    iterator& operator++()
    {
        ++row_;
        return *this;
    }

    iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    iterator& operator+=(difference_type step)
    {
        row_ += step;
        return *this;
    }

    iterator operator+(difference_type step) const
    {
        auto tmp = *this;
        tmp += step;
        return tmp;
    }

    iterator& operator--()
    {
        --row_;
        return *this;
    }

    iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    iterator& operator-=(difference_type step)
    {
        row_ -= step;
        return *this;
    }

    iterator operator-(difference_type step) const
    {
        auto tmp = *this;
        tmp -= step;
        return tmp;
    }

    difference_type operator-(const iterator& it) const
    {
        return row_ - it.row_;
    }

    friend bool operator==(const iterator& a, const iterator& b)
    {
        return a.batch_ == b.batch_ && a.row_ == b.row_;
    }

    friend bool operator!=(const iterator& a, const iterator& b)
    {
        return a.batch_ != b.batch_ || a.row_ != b.row_;
    }
private:
    const batch* batch_;
    int64_t row_;
};

inline batch::iterator batch::begin() const
{
    return iterator(*this, 0L);
}

inline batch::iterator batch::end() const
{
    return iterator(*this, size());
}

inline batch::iterator batch::cbegin() const
{
    return iterator(*this, 0L);
}

inline batch::iterator batch::cend() const
{
    return iterator(*this, size());
}

inline sample batch::operator[](int64_t row) const
{
    return sample(*this, row);
}

}
