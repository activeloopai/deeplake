#pragma once

#include "dtype.hpp"

#include <icm/shape.hpp>
#include <icm/vector.hpp>

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace nd {

class array;

class data_iterator
{
public:
    data_iterator(std::shared_ptr<void> owner, std::span<const uint8_t> data, icm::shape shape, dtype dt)
        : owner_(std::move(owner))
        , data_(data)
        , shape_(std::move(shape))
        , dtype_(dt)
    {
        ASSERT(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies{}) * dtype_bytes(dtype_) ==
               data.size());
    }

    array operator*() const;

    void operator++()
    {
        data_ = std::span<const uint8_t>(data_.data() + data_.size(), data_.size());
    }

    void operator--()
    {
        data_ = std::span<const uint8_t>(data_.data() - data_.size(), data_.size());
    }

    void operator+=(std::ptrdiff_t n)
    {
        data_ = std::span<const uint8_t>(data_.data() + n * data_.size(), data_.size());
    }

    void operator-=(std::ptrdiff_t n)
    {
        data_ = std::span<const uint8_t>(data_.data() - n * data_.size(), data_.size());
    }

    std::ptrdiff_t operator-(const data_iterator& other) const
    {
        return (data_.data() - other.data_.data()) / data_.size();
    }

    bool operator==(const data_iterator& other) const
    {
        return data_.data() == other.data_.data();
    }

    bool operator!=(const data_iterator& other) const
    {
        return data_.data() != other.data_.data();
    }

private:
    std::shared_ptr<void> owner_;
    std::span<const uint8_t> data_;
    icm::shape shape_;
    dtype dtype_;
};

using dynamic_vector_iterator = std::vector<array>::const_iterator;
using dynamic_icm_vector_iterator = icm::vector<array>::const_iterator;

class iterator
{
public:
    struct holder
    {
        virtual ~holder() = default;
        virtual array value() const = 0;
        virtual void operator++() = 0;
        virtual void operator+=(int) = 0;
        virtual void operator--() = 0;
        virtual void operator-=(int) = 0;
        virtual std::ptrdiff_t operator-(const holder& other) const = 0;
        virtual bool operator==(const holder& other) const = 0;
    };

    template <typename T>
    class concrete_holder : public holder
    {
    public:
        concrete_holder(T value)
            : value_(std::move(value))
        {
        }

        array value() const override;

        void operator++() override
        {
            ++value_;
        }

        void operator+=(int n) override
        {
            value_ += n;
        }

        void operator--() override
        {
            --value_;
        }

        void operator-=(int n) override
        {
            value_ -= n;
        }

        std::ptrdiff_t operator-(const holder& other) const override
        {
            ASSERT(dynamic_cast<const concrete_holder<T>*>(&other));
            return value_ - static_cast<const concrete_holder<T>&>(other).value_;
        }

        bool operator==(const holder& other) const override
        {
            ASSERT(dynamic_cast<const concrete_holder<T>*>(&other));
            return value_ == static_cast<const concrete_holder<T>&>(other).value_;
        }

    private:
        T value_;
    };

    iterator(data_iterator it)
        : iterator_(std::move(it))
    {
    }

    iterator(dynamic_vector_iterator it)
        : iterator_(std::move(it))
    {
    }

    iterator(dynamic_icm_vector_iterator it)
        : iterator_(std::move(it))
    {
    }

    template <typename T>
    iterator(T it)
        : iterator_(std::make_unique<concrete_holder<T>>(std::move(it)))
    {
    }

    iterator(iterator&&) noexcept = default;
    iterator& operator=(iterator&&) noexcept = default;

    array operator*() const;

    void operator++();

    void operator+=(int n);

    void operator--();

    void operator-=(int n);

    std::ptrdiff_t operator-(const iterator& other) const;

    bool operator==(const iterator& other) const
    {
        ASSERT(iterator_.index() == other.iterator_.index());
        switch (iterator_.index()) {
        case 0:
            return std::get<0>(iterator_) == std::get<0>(other.iterator_);
        case 1:
            return std::get<1>(iterator_) == std::get<1>(other.iterator_);
        case 2:
            return std::get<2>(iterator_) == std::get<2>(other.iterator_);
        case 3:
            return std::get<3>(iterator_)->operator==(*std::get<3>(other.iterator_));
        default:
            ASSERT(false);
            return false;
        }
    }

    bool operator!=(const iterator& other) const
    {
        return !(*this == other);
    }

private:
    std::variant<data_iterator, dynamic_vector_iterator, dynamic_icm_vector_iterator, std::unique_ptr<holder>>
        iterator_;
};

}
