#pragma once

#include "../array.hpp"
#include "../dtype.hpp"

#include <base/span_cast.hpp>
#include <icm/shape.hpp>

#include <cstdint>
#include <span>

namespace nd {

namespace impl {

class chained_eval_array
{
private:
    struct holder {
        holder(nd::array owner, icm::shape shape, uint32_t volume)
            : owner_(std::move(owner)), first_child_(owner_[0]),
              value_(first_child_.data().data(), volume * dtype_bytes(owner_.dtype())), shape_(shape)
        {
        }

        holder(const holder&) = delete;
        holder(holder&&) = delete;
        holder& operator=(const holder&) = delete;
        holder& operator=(holder&&) = delete;

        const nd::array owner_;
        const nd::array first_child_;
        const std::span<const uint8_t> value_;
        const icm::shape shape_;
    };
public:
    chained_eval_array(nd::array owner, icm::shape shape, uint32_t volume)
        : holder_(std::make_shared<holder>(std::move(owner), std::move(shape), volume))
    {
    }

    enum dtype dtype() const
    {
        return holder_->owner_.dtype();
    }

    std::span<const uint8_t> data() const
    {
        return holder_->value_;
    }

    const icm::shape& shape() const
    {
        return holder_->shape_;
    }

    const auto& owner() const
    {
        return holder_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    std::shared_ptr<holder> holder_;
};

} // namespace impl

} // namespace nd
