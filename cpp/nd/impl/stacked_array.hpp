#pragma once

#include "../adapt.hpp"
#include "../array.hpp"
#include "../eval.hpp"
#include "../shape_utils.hpp"

#include <base/assert.hpp>
#include <icm/shape.hpp>
#include <icm/small_vector.hpp>
#include <icm/vector.hpp>

#include <atomic>
#include <memory>

namespace nd::impl {

class vstacked_array_base
{
public:
    vstacked_array_base(array a1, array a2)
        : a1_(std::move(a1))
        , a2_(std::move(a2))
        , shape_(a1_.size() + a2_.size())
    {
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    array get(int64_t index) const
    {
        return index < a1_.size() ? a1_[index] : a2_[index - a1_.size()];
    }

protected:
    array a1_;
    array a2_;
    icm::shape shape_;
};

template <typename T>
class vstacked_array : public vstacked_array_base
{
public:
    vstacked_array(array a1, array a2)
        : vstacked_array_base(std::move(a1), std::move(a2))
    {
        ASSERT(a1_.dtype() == a2_.dtype());
        auto a1_shape = a1_.shape();
        auto s1 = std::span<const icm::shape::value_type>(a1_shape.data() + 1, a1_shape.data() + a1_shape.size());
        auto a2_shape = a2_.shape();
        ASSERT(shapes_equal(
            s1, std::span<const icm::shape::value_type>(a2_shape.data() + 1, a2_shape.data() + a2_shape.size())));
        icm::small_vector<icm::shape::value_type> s{a1_.size() + a2_.size()};
        s.insert(s.end(), s1.begin(), s1.end());

        shape_ = icm::shape(s);
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    T value(int64_t index) const
    {
        auto v1 = a1_.volume();
        if (index < v1) {
            return a1_.template value<T>(index);
        }
        return a2_.template value<T>(index - v1);
    }

    array eval() const
    {
        auto volume = std::accumulate(shape_.begin(), shape_.end(), 1L, std::multiplies<int64_t>());
        auto v = icm::vector<T>(volume);
        copy_data(base::span_cast<uint8_t>(std::span<T>(v.data(), v.size())));
        return nd::adapt(std::move(v), icm::shape(shape_));
    }

    void copy_data(std::span<uint8_t> buffer) const
    {
        auto volume = std::accumulate(shape_.begin(), shape_.end(), 1L, std::multiplies<int64_t>()) * sizeof(T);
        auto vol1 = volume * a1_.size() / (a1_.size() + a2_.size());
        auto sp1 = std::span<uint8_t>(buffer.data(), vol1);
        auto sp2 = std::span<uint8_t>(buffer.data() + vol1, volume - vol1);
        nd::copy_data(a1_, base::span_cast<uint8_t>(sp1));
        nd::copy_data(a2_, base::span_cast<uint8_t>(sp2));
    }

    uint8_t dimensions() const
    {
        return static_cast<uint8_t>(shape_.size());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }
};

class dynamic_vstacked_array : public vstacked_array_base
{
public:
    dynamic_vstacked_array(array a1, array a2)
        : vstacked_array_base(std::move(a1), std::move(a2))
    {
        ASSERT(a1_.dtype() == a2_.dtype());
    }

    enum dtype dtype() const
    {
        return a1_.dtype();
    }

    uint8_t dimensions() const
    {
        return static_cast<uint8_t>(a1_.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }
};

class vstacked_list_array_base
{
public:
    vstacked_list_array_base(std::vector<array>&& a, std::shared_ptr<void>&& owner)
        : a_(std::move(a))
        , owner_(std::move(owner))
    {
        std::transform_exclusive_scan(
            a_.begin(), a_.end(), std::back_inserter(offsets_), 0L, std::plus<int64_t>(), [](const array& a) {
                return a.size();
            });
        auto a_shape = a_[0].shape();
        auto s1 = std::span<const icm::shape::value_type>(a_shape.data() + 1, a_shape.data() + a_shape.size());
        auto size = std::transform_reduce(a_.begin(), a_.end(), 0L, std::plus<int64_t>(), [](const array& a) {
            return a.size();
        });
        icm::small_vector<icm::shape::value_type> s{size};
        s.insert(s.end(), s1.begin(), s1.end());
        shape_ = icm::shape(s);
        sub_volume_ = std::accumulate(s1.begin(), s1.end(), 1L, std::multiplies<int64_t>());
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    class iterator
    {
    public:
        iterator(const vstacked_list_array_base* a, int64_t index, int64_t parent_index)
            : a_(a)
            , parent_index_(parent_index)
            , index_(index)
        {
        }

        void operator++()
        {
            ++index_;
            if (index_ >= a_->offsets_[parent_index_] + a_->a_[parent_index_].size()) {
                ++parent_index_;
            }
        }

        void operator--()
        {
            --index_;
            if (index_ < a_->offsets_[parent_index_]) {
                --parent_index_;
            }
        }

        void operator+=(std::ptrdiff_t n)
        {
            index_ += n;
            parent_index_ = std::distance(a_->offsets_.begin(), std::ranges::upper_bound(a_->offsets_, index_)) - 1;
        }

        void operator-=(std::ptrdiff_t n)
        {
            index_ -= n;
            parent_index_ = std::distance(a_->offsets_.begin(), std::ranges::upper_bound(a_->offsets_, index_)) - 1;
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
            return a_->a_[parent_index_][index_ - a_->offsets_[parent_index_]];
        }

    private:
        const vstacked_list_array_base* a_;
        int64_t parent_index_;
        int64_t index_;
    };

    iterator begin() const
    {
        return iterator(this, 0, 0);
    }

    iterator end() const
    {
        return iterator(this, shape_[0], a_.size());
    }

    array get(int64_t index) const
    {
        auto it = std::ranges::upper_bound(offsets_, index) - 1;
        ASSERT(it != offsets_.end());
        const auto b = *it;
        const auto c = std::distance(offsets_.begin(), it);
        return a_[c][index - b];
    }

protected:
    friend class iterator;
    std::vector<array> a_;
    std::shared_ptr<void> owner_;
    std::vector<int64_t> offsets_;
    icm::shape shape_;
    int64_t sub_volume_ = 0L;
};

template <typename T>
class vstacked_list_array : public vstacked_list_array_base
{
public:
    vstacked_list_array(std::vector<array>&& a, std::shared_ptr<void>&& owner)
        : vstacked_list_array_base(std::move(a), std::move(owner))
    {
    }

    vstacked_list_array(const vstacked_list_array& a) = default;
    vstacked_list_array& operator=(const vstacked_list_array& a) = delete;
    vstacked_list_array(vstacked_list_array&&) noexcept = default;
    vstacked_list_array& operator=(vstacked_list_array&& a) = delete;

    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    T value(int64_t index) const
    {
        return get(index / sub_volume_).template value<T>(index % sub_volume_);
    }

    array eval() const
    {
        auto volume = std::accumulate(shape_.begin(), shape_.end(), 1L, std::multiplies<int64_t>());
        auto v = icm::vector<T>(volume);
        copy_data(base::span_cast<uint8_t>(std::span<T>(v.data(), v.size())));
        return nd::adapt(std::move(v), icm::shape(shape_));
    }

    void copy_data(std::span<uint8_t> buffer) const
    {
        auto current_offset = 0L;
        for (const auto& a : a_) {
            auto vol = a.volume() * sizeof(T);
            auto sp = std::span<uint8_t>(buffer.data() + current_offset, vol);
            nd::copy_data(a, base::span_cast<uint8_t>(sp));
            current_offset += vol;
        }
    }

    uint8_t dimensions() const
    {
        return static_cast<uint8_t>(shape_.size());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }
};

template <typename T>
class dynamic_vstacked_list_array : public vstacked_list_array_base
{
public:
    dynamic_vstacked_list_array(std::vector<array>&& a, std::shared_ptr<void>&& owner)
        : vstacked_list_array_base(std::move(a), std::move(owner))
    {
    }

    dynamic_vstacked_list_array(const dynamic_vstacked_list_array& a) = default;
    dynamic_vstacked_list_array& operator=(const dynamic_vstacked_list_array& a) = delete;
    dynamic_vstacked_list_array(dynamic_vstacked_list_array&&) noexcept = default;
    dynamic_vstacked_list_array& operator=(dynamic_vstacked_list_array&& a) = delete;

    enum dtype dtype() const
    {
        return dtype_enum_v<T>;
    }

    uint8_t dimensions() const
    {
        return a_[0].dimensions();
    }

    const std::vector<array>& arrays() const
    {
        return a_;
    }

    const std::vector<int64_t>& offsets() const
    {
        return offsets_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }
};

} // namespace nd::impl
