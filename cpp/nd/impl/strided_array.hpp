#pragma once

#include "../array.hpp"
#include "../exceptions.hpp"
#include "../shape_utils.hpp"
#include "../stride.hpp"

#include <base/assert.hpp>
#include <icm/index_mapping.hpp>
#include <icm/indexable.hpp>
#include <icm/shape.hpp>
#include <icm/slice.hpp>
#include <icm/slice_shape.hpp>

namespace nd {

namespace impl {

template <typename I>
int new_to_old_offset(int new_offset, const icm::shape& old_shape, I begin, I end)
{
    icm::small_vector<int> old_strides(old_shape.size(), 1);
    icm::small_vector<int> new_strides(old_shape.size(), 1);
    auto os = 1;
    auto ns = 1;
    auto distance = std::distance(begin, end);
    for (auto i = old_shape.size() - 1; i > 0; --i) {
        os *= old_shape[i];
        if (i < distance) {
            ns *= (begin + i)->size();
        } else {
            ns *= old_shape[i];
        }
        old_strides[i - 1] = os;
        new_strides[i - 1] = ns;
    }
    auto old_offset = 0;
    I it = begin;
    for (auto i = 0; i < old_shape.size(); ++i) {
        if (it != end) {
            auto ci = (*it)[new_offset / new_strides[i]];
            if (ci >= old_shape[i]) {
                throw invalid_operation("Subscript index is out of array bounds.");
            }
            old_offset += old_strides[i] * ci;
            ++it;
        } else {
            old_offset += old_strides[i] * (new_offset / new_strides[i]);
        }
        new_offset %= new_strides[i];
    }
    return old_offset;
}

template <typename T, typename I>
class single_strided_array
{
public:
    using value_type = T;

public:
    single_strided_array(array a, icm::index_mapping_t<I> index)
        : shape_(icm::slice_shape(a.shape(), &index, &index + 1))
        , a_(std::move(a))
        , index_(std::move(index))
    {}

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    value_type value(int64_t index) const
    {
        auto o = new_to_old_offset(index, a_.shape(), &index_, &index_ + 1);
        return a_.value<value_type>(o);
    }

    array get(int64_t index) const
    {
        return a_[index_[index]];
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const icm::shape shape_;
    const array a_;
    const icm::index_mapping_t<I> index_;
};

template <typename T>
class strided_array
{
public:
    using value_type = T;

public:
    strided_array(array a, icm::index_mapping_vector indices)
        : shape_(icm::slice_shape(a.shape(), indices.begin(), indices.end()))
        , a_(std::move(a))
        , indices_(std::move(indices))
    {}

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    value_type value(int64_t index) const
    {
        auto o = new_to_old_offset(index, a_.shape(), indices_.begin(), indices_.end());
        return a_.value<value_type>(o);
    }

    array get(int64_t index) const
    {
        return nd::stride(a_[indices_.front()[index]],
                          icm::index_mapping_vector{indices_.begin() + 1, indices_.end()});
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const icm::shape shape_;
    const array a_;
    const icm::index_mapping_vector indices_;
};

template <typename I>
class single_dynamic_strided_array
{
public:
    single_dynamic_strided_array(array a, icm::index_mapping_t<I> index)
        : a_(std::move(a))
        , index_(std::move(index))
        , shape_(index_.size())
    {
        ASSERT(a_.is_dynamic());
    }

    enum dtype dtype() const
    {
        return a_.dtype();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    array get(int64_t index) const
    {
        return a_[index_[index]];
    }

    uint8_t dimensions() const
    {
        return a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    const array a_;
    const icm::index_mapping_t<I> index_;
    const icm::shape shape_;
};

class strided_dynamic_array
{
public:
    strided_dynamic_array(array a, icm::index_mapping index, icm::indexable_vector slices)
        : shape_(index.size())
        , a_(std::move(a))
        , index_(std::move(index))
        , slices_(std::move(slices))
    {
        ASSERT(a_.is_dynamic());
    }

    enum dtype dtype() const
    {
        return a_.dtype();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    array get(int64_t index) const
    {
        return nd::stride(a_[index_[index]], slices_);
    }

    uint8_t dimensions() const
    {
        return a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    const icm::shape shape_;
    const array a_;
    const icm::index_mapping index_;
    const icm::indexable_vector slices_;
};

}

}
