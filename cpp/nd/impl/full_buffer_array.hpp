#pragma once

#include "../adapt.hpp"
#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"

#include "dict_array.hpp"

#include <base/memory_buffer.hpp>
#include <format/pod_serializable.hpp>
#include <format/serializer.hpp>
#include <format/stl_serializable.hpp>
#include <icm/icm_format/shape_serializable.hpp>
#include <icm/shape.hpp>

#include <span>
#include <vector>

namespace nd::impl {

class buffer_array_with_shape
{
public:
    buffer_array_with_shape(base::memory_buffer data, icm::shape shape, dtype dt)
        : data_(std::move(data))
        , shape_(std::move(shape))
        , dtype_(dt)
    {
    }

    enum dtype dtype() const
    {
        return dtype_;
    }

    std::span<const uint8_t> data() const
    {
        return data_.span();
    }

    const auto& owner() const
    {
        return data_.owner();
    }

    const icm::shape& shape() const
    {
        return shape_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    base::memory_buffer data_;
    icm::shape shape_;
    enum dtype dtype_;
};

class full_buffer_array
{
public:
    full_buffer_array(base::memory_buffer data, dtype dt)
        : data_(std::move(data))
        , dtype_(dt)
    {
        header_ = std::make_shared<nd::array_header>();
        header_->read_header(data_);
    }

    enum dtype dtype() const noexcept
    {
        return dtype_;
    }

    nd::array get(int64_t index) const
    {
        ASSERT(header_ != nullptr);
        auto buffer = data_.chunk(header_->offsets_[index] + header_->data_offset_,
                                  header_->offsets_[index + 1] + header_->data_offset_);
        return header_->get(buffer, dtype_, index);
    }

    icm::shape shape() const noexcept
    {
        return icm::shape(header_->shapes_.size());
    }

    uint8_t dimensions() const
    {
        return (header_->shapes_.empty() ? 1 : static_cast<uint8_t>(header_->shapes_[0].size() + 1));
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<nd::array_header> header_;
    base::memory_buffer data_;
    enum dtype dtype_;
};

class object_buffer_array
{
public:
    explicit object_buffer_array(base::memory_buffer data, uint32_t version, int64_t num_rows)
        : data_(std::move(data))
        , num_rows_(num_rows)
    {
        header_ = std::make_shared<nd::array_header>();
        header_->dtype_ = nd::dtype::object;
        header_->num_rows_ = num_rows_;
        header_->has_schema_ = true;
        header_->version_ = version;
        header_->read_header(data_);
    }

    enum dtype dtype() const noexcept
    {
        return nd::dtype::object;
    }

    nd::array get(int64_t index) const
    {
        icm::string_map<nd::array> key2arr;
        for (auto i = 0; i < header_->keys_.size(); ++i) {
            const auto& [key, dt] = header_->keys_[i];
            const auto start = header_->offsets_[i] + header_->data_offset_;
            const auto end = header_->offsets_[i + 1] + header_->data_offset_;
            auto buffer = data_.chunk(start, end);
            key2arr[key] = header_->get(buffer, dt, i);
        }
        if (header_->version_ < 2U) {
            return nd::impl::dict_array(std::move(key2arr), 1).get(index);
        } else {
            return nd::impl::dict_array(std::move(key2arr), header_->dynamic_rows_[index]).get(index);
        }
    }

    icm::shape shape() const noexcept
    {
        return icm::shape(num_rows_);
    }

    uint8_t dimensions() const
    {
        return 1;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    std::shared_ptr<nd::array_header> header_;
    base::memory_buffer data_;
    const int64_t num_rows_ = 0;
};

// TODO move from impl to public
class array_with_transform
{
public:
    array_with_transform(nd::array arr, std::vector<int32_t>&& transform_info)
        : arr_(std::move(arr))
        , transform_info_(std::move(transform_info))
    {
    }

    enum dtype dtype() const
    {
        return arr_.dtype();
    }

    icm::shape shape() const
    {
        return arr_.shape();
    }

    nd::array get(int64_t index) const
    {
        return arr_[index];
    }

    nd::array eval() const
    {
        return nd::eval(arr_);
    }

    inline const std::vector<int32_t>& transform_info() const
    {
        return transform_info_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return arr_.is_dynamic();
    }

private:
    nd::array arr_;
    std::vector<int32_t> transform_info_;
};

} // namespace nd::impl
