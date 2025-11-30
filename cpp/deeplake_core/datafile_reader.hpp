#pragma once

#include <async/promise.hpp>
#include <nd/array.hpp>

#include <cstdint>
#include <memory>

namespace deeplake_core {

class datafile_reader : public std::enable_shared_from_this<datafile_reader>
{
public:
    virtual ~datafile_reader() = default;

    virtual bool is_used() const noexcept = 0;

    virtual int64_t memory_size() const noexcept = 0;

    [[nodiscard]] virtual async::promise<nd::array> value(uint32_t offset) = 0;
    [[nodiscard]] virtual async::promise<nd::array> sub_value(uint32_t offset, uint32_t sub_index)
    {
        throw std::runtime_error("sub_value is not implemented");
    }

    [[nodiscard]] virtual async::promise<nd::array> value(uint32_t start, uint32_t end) = 0;

    [[nodiscard]] virtual async::promise<nd::array> full_value() = 0;

    [[nodiscard]] virtual async::promise<nd::array> shape(uint32_t offset) = 0;
    [[nodiscard]] virtual async::promise<nd::array> sub_shape(uint32_t offset, uint32_t sub_index)
    {
        throw std::runtime_error("sub_shape is not implemented");
    }

    [[nodiscard]] virtual async::promise<nd::array> shape(uint32_t start, uint32_t end) = 0;

    [[nodiscard]] virtual async::promise<nd::array> full_shape() = 0;

    [[nodiscard]] virtual async::promise<nd::array> bytes(uint32_t offset) = 0;
    [[nodiscard]] virtual async::promise<nd::array> sub_bytes(uint32_t offset, uint32_t sub_index)

    {
        throw std::runtime_error("sub_bytes is not implemented");
    }

    [[nodiscard]] virtual async::promise<nd::array> bytes(uint32_t start, uint32_t end) = 0;

    [[nodiscard]] virtual async::promise<nd::array> full_bytes() = 0;
};

} // namespace deeplake_core
