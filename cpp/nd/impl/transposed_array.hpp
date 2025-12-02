#pragma once

#include "../array.hpp"

#include <icm/vector.hpp>

namespace nd::impl {

template <typename T>
class transposed_array
{
public:
    using value_type = T;

public:
    explicit transposed_array(nd::array arr)
        : a_(std::move(arr))
        , shape_{a_.shape()[1], a_.shape()[0]}
    {
        ASSERT(a_.shape().size() == 2);
    }

public:
    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    value_type value(int64_t index) const
    {
        auto y = index / shape_[1];
        auto x = index % shape_[1];
        auto ii = y * shape_[0] + x;
        return a_.value<value_type>(ii);
    }

    nd::array get(int64_t index) const
    {
        icm::vector<value_type> data(static_cast<std::size_t>(shape_[1]));
        for (auto j = 0; j < shape_[1]; ++j) {
            data[j] = a_.value<value_type>(j * shape_[0] + index);
        }
        return nd::adapt(data);
    }

    nd::array eval() const
    {
        nd::array a = nd::eval(nd::cast<dtype_enum_v<T>>(a_));
        auto sp = a.data<T>();
        auto d = icm::vector<T>(static_cast<std::size_t>(shape_[0] * shape_[1]));
        for (auto i = 0; i < shape_[0]; ++i) {
            for (auto j = 0; j < shape_[1]; ++j) {
                d[i * shape_[1] + j] = sp[j * shape_[0] + i];
            }
        }
        return nd::adapt(std::move(d), icm::shape(shape_));
    }

    constexpr bool is_dynamic() const noexcept
    {
        return a_.is_dynamic();
    }

private:
    nd::array a_;
    icm::shape shape_;
};

} // namespace nd::impl
