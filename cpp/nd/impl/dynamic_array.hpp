#pragma once

#include "../array.hpp"

#include <memory>
#include <vector>

namespace nd::impl {

template <typename T>
struct dynamic_array
{
    explicit dynamic_array(T&& arrays)
        : arrays_(std::make_shared<T>(std::move(arrays)))
        , shape_(static_cast<uint64_t>(arrays_->size()))
    {
    }

    nd::dtype dtype() const
    {
        return arrays_->empty() ? nd::dtype::uint8 : arrays_->front().dtype();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    T::const_iterator begin() const
    {
        return arrays_->begin();
    }

    T::const_iterator end() const
    {
        return arrays_->end();
    }

    nd::array get(int64_t index) const
    {
        return (*arrays_)[index];
    }

    uint8_t dimensions() const
    {
        return arrays_->empty() ? 1 : static_cast<uint8_t>(arrays_->front().dimensions() + 1);
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<T> arrays_;
    icm::shape shape_;
};

struct dynamic_repeated_array
{
    explicit dynamic_repeated_array(nd::array arr, int64_t size)
        : array_(std::move(arr))
        , shape_(size)
    {
    }

    dynamic_repeated_array(const dynamic_repeated_array&) = default;
    dynamic_repeated_array& operator=(const dynamic_repeated_array&) = delete;
    dynamic_repeated_array& operator=(dynamic_repeated_array&&) = default;
    dynamic_repeated_array(dynamic_repeated_array&& s) = default;
    ~dynamic_repeated_array() = default;

    class iterator
    {
    public:
        iterator(const array& a, int64_t index)
            : array_(a)
            , index_(index)
        {
        }

        void operator++()
        {
            ++index_;
        }

        void operator--()
        {
            --index_;
        }

        void operator+=(std::ptrdiff_t n)
        {
            index_ += n;
        }

        void operator-=(std::ptrdiff_t n)
        {
            index_ -= n;
        }

        std::ptrdiff_t operator-(const iterator& other) const
        {
            return index_ - other.index_;
        }

        bool operator==(const iterator& other) const
        {
            return index_ == other.index_;
        }

        array operator*() const
        {
            return array_;
        }

    private:
        const array& array_;
        int64_t index_ = 0;
    };

    iterator begin() const
    {
        return iterator(array_, 0);
    }

    iterator end() const
    {
        return iterator(array_, shape_[0]);
    }

    nd::dtype dtype() const
    {
        return array_.dtype();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    nd::array get(int64_t) const
    {
        return array_;
    }

    uint8_t dimensions() const
    {
        return static_cast<uint8_t>(array_.dimensions() + 1);
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    nd::array array_;
    icm::shape shape_;
};

}
