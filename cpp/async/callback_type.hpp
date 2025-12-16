#pragma once

/**
 * @file callback_type.hpp
 * @brief Definition of the `callback_type`
 */

#include "async.hpp"

#include <base/function.hpp>

#include <concepts>

namespace async {

template <typename T>
class result;

/// @brief Callback type used in promises.
template <typename T>
struct callback_type
{
public:
    using self_t = callback_type<T>;
    using underlying_t = base::function<void(result<T>&& v)>;

public:
    callback_type() noexcept = default;

    template <typename F>
    requires (!std::same_as<self_t, std::remove_cvref_t<F>>)
    inline callback_type(F&& f, queue* q = &async::main())
        : callback_(std::forward<F>(f))
        , queue_(q)
    {
    }

    inline operator bool() const noexcept
    {
        return static_cast<bool>(callback_);
    }

    inline void operator()(result<T>&& v)
    {
#if defined(AL_ASSERTIONS)
        auto sl = callback_.get_source_location();
#endif
        base::function<void()> c(
            [cb = std::move(callback_), v = std::move(v)]() mutable {
                cb(std::move(v));
            }
#if defined(AL_ASSERTIONS)
            ,
            std::move(sl)
#endif
        );
        if (queue_) {
            queue_->run_or_submit(std::move(c));
        } else {
            c();
        }
    }

private:
    underlying_t callback_;
    queue* queue_ = nullptr;
};

} // namespace async
