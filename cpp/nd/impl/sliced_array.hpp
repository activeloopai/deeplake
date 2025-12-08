#pragma once

#include "../adapt.hpp"
#include "../array.hpp"
#include "../exceptions.hpp"

#include <icm/slice.hpp>
#include <icm/vector.hpp>

namespace {

template <typename I>
requires std::is_arithmetic_v<I>
struct slice_
{
    I step;
    I offset;
};

template <typename I>
requires std::is_arithmetic_v<I>
class data_resizer
{
public:
    data_resizer(icm::shape shape, icm::shape old_shape, icm::small_vector<slice_<I>> slices)
        : shape_(std::move(shape))
        , old_shape_(std::move(old_shape))
        , slices_(std::move(slices))
    {
        ASSERT(shape_.size() == old_shape_.size());
        if (shape_.size() != old_shape_.size()) {
            throw nd::shape_dimensions_do_not_match(shape_.size(), old_shape_.size());
        }

        ASSERT(slices_.size() == shape_.size());
        if (slices_.size() != shape_.size()) {
            throw nd::shape_dimensions_do_not_match(slices_.size(), shape_.size());
        }
    }

    template <typename T>
    void resize(const uint8_t* from, uint8_t* to)
    {
        auto [old_volume, new_volume] = compute_volumes();
        // start from first (for 3D depth, height, width, etc...)
        resize_nd_array<T>(from, to, 0, old_volume, new_volume);
    }

    int64_t compute_index(int64_t index)
    {
        auto [old_volume, new_volume] = compute_volumes();
        // start from first (for 3D depth, height, width, etc...)
        int64_t new_index = index;
        compute_index_(index, 0, 0, 0, old_volume, new_volume, &new_index);
        return new_index;
    }

private:
    template <typename T>
    void
    resize_nd_array(const uint8_t* from, uint8_t* to, size_t dims_idx, int64_t old_data_volume, int64_t new_data_volume)
    {
        // the if is just a safety net.
        if (dims_idx >= shape_.size()) {
            return;
        }
        // if we are at the last dimension we can simply copy the data.
        if (dims_idx == shape_.size() - 1) {
            auto& slice = slices_[dims_idx];
            // iterate over the row and copy the data.
            for (auto i = 0; i < shape_[dims_idx]; ++i) {
                auto old_offset = sizeof(T) * static_cast<int64_t>(i * slice.step + slice.offset);
                auto new_offset = sizeof(T) * i;
                std::memmove(to + new_offset, from + old_offset, sizeof(T));
            }
            return;
        }
        int d = shape_[dims_idx];
        if (!d) {
            return;
        }
        auto& slice = slices_[dims_idx];
        auto si = slice.step;
        // divide the volume by the current dimension size to get the volume for the current dimension.
        new_data_volume /= d;
        old_data_volume /= old_shape_[dims_idx];
        auto offset = slice.offset;
        for (auto i = 0; i < d; ++i) {
            auto old_offset = sizeof(T) * static_cast<int64_t>(i * si + offset) * old_data_volume;
            auto new_offset = sizeof(T) * i * new_data_volume;
            resize_nd_array<T>(from + old_offset, to + new_offset, dims_idx + 1, old_data_volume, new_data_volume);
        }
    }

    void compute_index_(int64_t index,
                        int64_t dims_idx,
                        int64_t old_offset,
                        int64_t new_offset,
                        int64_t old_data_volume,
                        int64_t new_data_volume,
                        int64_t* index_out)
    {
        ASSERT(index_out);
        // the if is just a safety net.
        if (dims_idx >= shape_.size()) {
            return;
        }
        // if we are at the last dimension we can simply assign the index.
        if (dims_idx == shape_.size() - 1) {
            auto& slice = slices_[dims_idx];
            auto i = index - new_offset;
            // Check if i is within valid bounds
            if (i >= 0 && i < shape_[dims_idx]) {
                *index_out = static_cast<int64_t>(i * slice.step + slice.offset) + old_offset;
            }
            return;
        }
        int d = shape_[dims_idx];
        if (!d) {
            return;
        }
        auto& slice = slices_[dims_idx];
        auto si = slice.step;
        // divide the volume by the current dimension size to get the volume for the current dimension.
        new_data_volume /= d;
        old_data_volume /= old_shape_[dims_idx];
        auto offset = slice.offset;
        for (auto i = 0; i < d; ++i) {
            auto current_old_offset = static_cast<int64_t>(i * si + offset) * old_data_volume + old_offset;
            auto current_new_offset = i * new_data_volume + new_offset;
            compute_index_(index,
                           dims_idx + 1,
                           current_old_offset,
                           current_new_offset,
                           old_data_volume,
                           new_data_volume,
                           index_out);
        }
    }

    std::pair<int64_t, int64_t> compute_volumes()
    {
        // compute stride for current level so we don't compute this recursively for each recursion step.
        // as we have the volume we will simply divide by the dimension size for each level.
        auto old_volume = std::accumulate(old_shape_.begin(), old_shape_.end(), 1, std::multiplies<int64_t>());
        auto new_volume = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());

        return {old_volume, new_volume};
    }

private:
    icm::shape shape_;
    icm::shape old_shape_;
    icm::small_vector<slice_<I>> slices_;
};

} // namespace

namespace nd::impl {

template <typename T, typename I>
requires std::is_arithmetic_v<I>
class sliced_array
{
public:
    using value_type = T;

public:
    explicit sliced_array(nd::array arr, icm::slice_vector_t<I> slices_vec)
        : a_(std::move(arr))
    {
        ASSERT(!a_.is_dynamic());
        std::vector<int64_t> new_shape;
        auto shape = a_.shape();

        if (shape.size() < slices_vec.size()) {
            throw nd::shape_dimensions_do_not_match(shape.size(), slices_vec.size());
        }

        while (slices_vec.size() < shape.size()) {
            slices_vec.push_back(icm::slice_t<I>::all());
        }

        new_shape.resize(shape.size());
        slices_.resize(shape.size());

        bool should_reverse = false;

        for (size_t i = 0; i < shape.size(); ++i) {
            auto& s = slices_vec[i];
            auto step = s.is_index() ? 1 : s.step();
            if (step == 0) {
                throw nd::invalid_operation("slice step cannot be zero");
            }

            I start = s.is_start_known() ? s.start() : ((step > 0) ? 0 : (shape[i] - 1));
            if (start < 0) {
                start += shape[i];
            }
            int64_t dim = 0;
            // hundle the following cases with the if statement:
            // >>> x
            // array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            // >>> x[-100::-1]
            // array([], dtype=int64)
            // >>> x[100::1]
            // array([], dtype=int64)
            // >>> x[100::-1]
            // array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
            // >>> x[-100::1]
            // array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            if (std::abs(start - shape[i]) < shape[i] || ((start * step) <= 0)) {
                start = std::max(std::min(start, static_cast<I>(shape[i] - 1)), static_cast<I>(0));

                auto stop = (!s.is_index() && s.is_known()) ? s.stop() : ((step > 0) ? shape[i] : (-shape[i] - 1));
                if (stop < 0) {
                    stop = shape[i] + stop;
                } else {
                    stop = std::min(stop, shape[i]);
                }

                if (step < 0) {
                    should_reverse = true;
                }
                dim = std::ceil((stop - start) / static_cast<float>(step));
            }

            new_shape[i] = std::max(dim, static_cast<int64_t>(0));
            slices_[i] = {step, start};
        }

        shape_ = icm::shape(std::move(new_shape));

        needs_resize_ = (shape_ != a_.shape()) || should_reverse;
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
        if (!needs_resize_) {
            return a_.value<value_type>(index);
        }
        auto resizer = data_resizer<I>(shape_, a_.shape(), slices_);
        return a_.value<value_type>(resizer.compute_index(index));
    }

    nd::array get(int64_t index) const
    {
        if (!needs_resize_) {
            return a_[index];
        }
        auto& s = slices_.front();
        auto ii = index * s.step + s.offset;

        if (a_.shape().size() > 1) {
            return nd::array(
                sliced_array(a_[ii], {slices_.begin() + 1, slices_.end()}, {shape_.begin() + 1, shape_.end()}));
        }
        return a_[ii];
    }

    nd::array eval() const
    {
        if (!needs_resize_) {
            return a_;
        }
        auto resizer = data_resizer<I>(shape_, a_.shape(), slices_);
        icm::vector<value_type> data;
        auto v = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
        data.resize(v);
        auto a = nd::eval(a_);
        auto d = a.data();
        resizer.template resize<value_type>(d.data(), reinterpret_cast<uint8_t*>(data.data()));
        return nd::adapt(std::move(data), icm::shape(shape_));
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    explicit sliced_array(nd::array arr, icm::small_vector<slice_<I>> slices, icm::shape current_shape)
        : a_(std::move(arr))
        , slices_(std::move(slices))
        , shape_(current_shape)
        , needs_resize_(a_.shape() != shape_)
    {
        if (!needs_resize_) {
            for (auto& slice : slices_) {
                if (slice.step < 0) {
                    needs_resize_ = true;
                    break;
                }
            }
        }
    }

private:
    nd::array a_;
    icm::shape shape_;
    icm::small_vector<slice_<I>> slices_;

    bool needs_resize_ = false;
};

} // namespace nd::impl
