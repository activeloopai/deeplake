#pragma once

#include "../adapt.hpp"
#include "../array.hpp"

#include <icm/vector.hpp>

namespace nd::impl {

inline nd::array flatten(nd::array a)
{
    if (a.dimensions() <= 1 && !a.is_dynamic()) {
        return a;
    }
    return nd::switch_numeric_dtype(a.dtype(), [&a]<typename T>() {
        icm::vector<T> v;
        v.reserve(a.volume());
        for (int i = 0; i < a.size(); i++) {
            auto arr = flatten(a[i]);
            if (!arr.has_data()) {
                for (int j = 0; j < arr.volume(); ++j) {
                    v.emplace_back(arr.template value<T>(j));
                }
            } else {
                ASSERT(arr.dtype() == a.dtype());
                auto d = arr.template data<T>();
                v.insert(v.end(), d.begin(), d.end());
            }
        }
        return adapt(std::move(v));
    });
}

template <typename T>
class flattened_array
{
public:
    using value_type = T;

public:
    explicit flattened_array(nd::array arr)
        : a_(std::move(arr))
        , shape_(a_.volume())
    {
    }

public:
    enum dtype dtype() const
    {
        return a_.dtype();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    value_type value(int64_t index) const
    {
        return a_.value<value_type>(index);
    }

    array get(int64_t index) const
    {
        return nd::adapt(value(index));
    }

    nd::array eval() const
    {
        if (a_.is_dynamic() || a_.dimensions() > 1) {
            return flatten(a_);
        }
        return nd::eval(a_);
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    nd::array a_;
    icm::shape shape_;
};

} // namespace nd::impl
