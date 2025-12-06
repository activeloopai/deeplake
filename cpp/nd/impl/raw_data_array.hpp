#pragma once

#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "serialize.hpp"

#include <base/memory_buffer.hpp>
#include <icm/shape.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace nd::impl {

struct raw_data_holder
{
    raw_data_holder(base::memory_buffer&& buffer, bool is_dynamic = false)
        : buffer_(std::move(buffer))
        , shape(buffer_.data() + header_size())
        , data(skip_container(shape))
    {
        auto version = *reinterpret_cast<const uint32_t*>(buffer_.data());
        auto d = container_dtype(shape);

        if (is_dynamic) {
            auto s = container_data(shape);
            auto num_elements = *reinterpret_cast<const uint32_t*>(s);
            auto s1 = container_size(shape);
            shape_ = icm::shape(num_elements);
        } else {
            if (version == 1) {
                shape_ = icm::shape(std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(container_data(shape)),
                                                              container_size(shape)));
            } else if (version == 2) {
                shape_ = icm::shape(std::span<const icm::shape::value_type>(
                    reinterpret_cast<const icm::shape::value_type*>(container_data(shape)), container_size(shape)));
            } else {
                throw nd::invalid_operation("Unsupported version");
            }
        }
    }

    const uint8_t* get_data_pointer() const
    {
        return skip_container(shape);
    }

    base::memory_buffer buffer_;
    const uint8_t* shape;
    const uint8_t* data;
    icm::shape shape_;
};

class raw_data_array
{

public:
    raw_data_array(base::memory_buffer&& buffer)
        : holder_(std::move(buffer))
    {
    }

public:
    enum dtype dtype() const
    {
        return container_dtype(holder_.data);
    }

    std::span<const uint8_t> data() const
    {
        return std::span<const uint8_t>(container_data(holder_.data),
                                        container_size(holder_.data) * dtype_bytes(dtype()));
    }

    const auto& owner() const
    {
        return holder_.buffer_.owner();
    }

    const icm::shape& shape() const
    {
        return holder_.shape_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    raw_data_holder holder_;
};

class raw_dynamic_data_array
{

public:
    raw_dynamic_data_array(base::memory_buffer&& buffer)
        : holder_(std::move(buffer), true)
    {
    }

public:
    enum dtype dtype() const
    {
        return container_dtype(holder_.data);
    }

    std::span<const uint8_t> data() const
    {
        return std::span<const uint8_t>(container_data(holder_.data),
                                        container_size(holder_.data) * dtype_bytes(dtype()));
    }

    const auto& owner() const
    {
        return holder_.buffer_.owner();
    }

    const icm::shape& shape() const
    {
        return holder_.shape_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

    uint8_t dimensions() const
    {
        auto res = static_cast<uint8_t>(holder_.shape_.size());
        return res;
    }

    nd::array get(int64_t index) const
    {
        const uint8_t* ptr = holder_.data + sizeof(nd::dtype) + sizeof(int64_t);
        for (int64_t i = 0; i < index; ++i) {
            std::size_t col_len = *reinterpret_cast<const std::size_t*>(ptr);
            ptr += sizeof(std::size_t) + col_len;
        }

        std::size_t len = *reinterpret_cast<const std::size_t*>(ptr);
        ptr += sizeof(std::size_t);
        std::span<const uint8_t> span(ptr, len);
        base::memory_buffer sub_buffer(holder_.buffer_.owner(), span);
        return nd::adapt(sub_buffer, icm::shape(len), nd::dtype::string);
    }

private:
    raw_data_holder holder_;
};

inline auto create_unowned_raw_data_array(const uint8_t* data, std::size_t size)
{
    return raw_data_array(base::memory_buffer(nullptr, std::span<const uint8_t>(data, size)));
}

inline auto create_memory_raw_data_array(base::memory_buffer&& buffer)
{
    return raw_data_array(std::move(buffer));
}

inline auto create_memory_dynamic_raw_data_array(base::memory_buffer&& buffer)
{
    return raw_dynamic_data_array(std::move(buffer));
}

} // namespace nd::impl
