#pragma once

#include <base/assert.hpp>
#include <nd/array.hpp>

#include <span>
#include <vector>

namespace heimdall_common::impl {

struct subgroup_sample_array
{
    subgroup_sample_array(std::shared_ptr<nd::array> main_array, int64_t start, int64_t length)
        : main_array_(main_array), start_(start), length_(length)
    {
    }

    inline nd::dtype dtype() const
    {
        return main_array_->dtype();
    }

    inline icm::shape shape() const
    {
        return icm::shape(length_);
    }

    inline nd::array get(int64_t index) const
    {
        ASSERT(index >= 0);
        ASSERT(index < length_);
        return (*main_array_)[start_ + index];
    }

    inline uint8_t dimensions() const
    {
        return static_cast<uint8_t>(main_array_->dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<nd::array> main_array_;
    int64_t start_;
    int64_t length_;
};

struct group_sample_array
{
    group_sample_array(nd::array&& main_array, std::span<const int64_t> sequence_lengths)
        : main_array_(std::make_shared<nd::array>(std::move(main_array)))
    {
        auto o = 0;
        for (auto l : sequence_lengths) {
            sequences_.emplace_back(main_array_, o, l);
            o += l;
        }
        shape_ = icm::shape(static_cast<uint32_t>(sequences_.size()));
        ASSERT(o == main_array_->shape()[0]);
    }

    group_sample_array(const group_sample_array&) = default;
    group_sample_array& operator=(const group_sample_array&) = delete;
    group_sample_array(group_sample_array&& s) = default;
    group_sample_array& operator=(group_sample_array&&) = default;

    ~group_sample_array() = default;

    inline nd::dtype dtype() const
    {
        return main_array_->dtype();
    }

    inline const icm::shape& shape() const
    {
        return shape_;
    }

    inline nd::array get(int64_t index) const
    {
        ASSERT(index >= 0);
        ASSERT(index < sequences_.size());
        return nd::array(subgroup_sample_array(sequences_[index]));
    }

    inline uint8_t dimensions() const
    {
        return static_cast<uint8_t>(main_array_->dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<nd::array> main_array_;
    std::vector<subgroup_sample_array> sequences_;
    icm::shape shape_;
};

} // namespace heimdall_common::impl
