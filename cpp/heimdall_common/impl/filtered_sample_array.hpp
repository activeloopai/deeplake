#pragma once

#include <base/assert.hpp>
#include <icm/index_mapping.hpp>
#include <nd/array.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace heimdall_common::impl {

struct filtered_sample_array
{
    filtered_sample_array(nd::array&& main_array, int64_t start, icm::index_mapping_t<int64_t> indices)
        : main_array_(std::move(main_array))
        , indices_(std::move(indices))
        , start_(start)
        , shape_(indices_.size())
    {}

    inline nd::dtype dtype() const
    {
        return main_array_.dtype();
    }

    inline const icm::shape& shape() const
    {
        return shape_;
    }

    inline nd::array get(int64_t index) const
    {
        ASSERT(index >= 0);
        ASSERT(index < shape()[0]);
        return main_array_[static_cast<int64_t>(indices_[index] - start_)];
    }

    inline uint8_t dimensions() const
    {
        return static_cast<uint8_t>(main_array_.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    nd::array main_array_;
    icm::index_mapping_t<int64_t> indices_;
    int64_t start_;
    icm::shape shape_;
};

}
