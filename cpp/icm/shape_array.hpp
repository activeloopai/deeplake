#pragma once

#include "shape.hpp"

#include <base/memory_buffer.hpp>

namespace icm {

template <typename T>
class shape_array
{
public:
    shape_array() = default;

    shape_array(base::memory_buffer buffer, T count)
        : buffer_(std::move(buffer))
        , span_(buffer_.template span<T>())
        , stride_(count ? span_.size() / count : 0)
    {
        ASSERT((count == 0 && span_.empty()) || span_.size() % count == 0);
    }

    T size() const noexcept
    {
        return span_.empty() ? 0 : span_.back() + 1;
    }

    bool empty() const noexcept
    {
        return size() == 0;
    }

    shape operator[](T index) const noexcept
    {
        ASSERT(index < size());
        auto distance = lower_bound(index);
        return shape(span_.subspan(distance * stride_, stride_ - 1));
    }

private:
    auto lower_bound(T index) const noexcept
    {
        T count = span_.size() / stride_;
        T step = count / 2;
        T first = 0;
        T it = 0;
        while (count > 0) {
            it = first;
            step = count / 2;
            it += step;
            if (span_[(it + 1) * stride_ - 1] < index) {
                first = ++it;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        return first;
    }

private:
    base::memory_buffer buffer_;
    std::span<const T> span_;
    T stride_ = 0;
};

}
