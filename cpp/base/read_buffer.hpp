#pragma once

#include "span_cast.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/container/vector.hpp>

namespace base {

class read_buffer
{
public:
    using value_type = uint8_t;

    read_buffer() = default;

    explicit read_buffer(std::string&& s)
    {
        auto h = std::make_shared<std::string>(std::move(s));
        data_ = span_cast<const uint8_t>(std::span<const char>(h->data(), h->size()));
        holder_ = std::move(h);
    }

    template <typename T>
    explicit read_buffer(std::vector<T>&& b)
        : data_(span_cast<const uint8_t>(std::span<T>(b.data(), b.size())))
        , holder_(std::make_shared<std::vector<T>>(std::move(b)))
    {
    }

    template <typename T>
    explicit read_buffer(boost::container::vector<T>&& b)
        : data_(span_cast<const uint8_t>(std::span<T>(b.data(), b.size())))
        , holder_(std::make_shared<boost::container::vector<T>>(std::move(b)))
    {
    }

    read_buffer(std::shared_ptr<void> holder, std::span<const uint8_t> data)
        : data_(data)
        , holder_(std::move(holder))
    {
    }

    const uint8_t* data() const noexcept
    {
        return data_.data();
    }

    size_t size() const noexcept
    {
        return data_.size();
    }

    bool empty() const noexcept
    {
        return size() == 0;
    }

    template <typename T>
    std::span<const T> span() const noexcept
    {
        return span_cast<const T>(span());
    }

    std::span<const uint8_t> span() const noexcept
    {
        return data_;
    }

    std::string_view string_view() const noexcept
    {
        return base::string_view_cast(base::span_cast<const char>(span()));
    }

    read_buffer chunk(int64_t start, int64_t end) const noexcept
    {
        return read_buffer(holder_, std::span<const uint8_t>(data_.begin() + start, data_.begin() + end));
    }

    template <typename T>
    read_buffer chunk(int64_t start, int64_t end) const noexcept
    {
        return read_buffer(holder_, span_cast<const uint8_t>(span<T>().subspan(start, end - start)));
    }

    const auto& operator[](size_t i) const noexcept
    {
        return span()[i];
    }

    auto begin() const noexcept
    {
        return span().begin();
    }

    auto end() const noexcept
    {
        return span().end();
    }

    auto rbegin() const noexcept
    {
        return span().rbegin();
    }

    auto rend() const noexcept
    {
        return span().rend();
    }

    bool is_shared() const noexcept
    {
        return holder_.use_count() > 1;
    }

    int64_t use_count() const noexcept
    {
        return holder_.use_count();
    }

    const std::shared_ptr<void>& owner() const noexcept
    {
        return holder_;
    }

private:
    std::span<const uint8_t> data_ = {};
    std::shared_ptr<void> holder_ = nullptr;
};

} // namespace base
