#pragma once

/**
 * @file promise.hpp
 * @brief Definition and implementation of the `promise` template class and some async utility functions.
 */

#include "callback_type.hpp"
#include "handle_base.hpp"
#include "impl/chained_promise.hpp"
#include "impl/fulfilled_promise.hpp"
#include "impl/mpl.hpp"
#include "impl/variadic_promises.hpp"
#include "run.hpp"

#include <base/base.hpp>

#include <future>
#include <memory>
#include <utility>

namespace async {

namespace impl {

template <typename T>
struct holder_
{
    virtual ~holder_() = default;
    virtual bool cancel() = 0;
    virtual float progress() const = 0;
    virtual void set_priority(int priority) = 0;
    virtual bool is_cancelled() const = 0;
    virtual bool has_exception() const = 0;
    virtual bool is_ready() const = 0;
    virtual T get() = 0;
    virtual void set_callback(callback_type<T>) = 0;
    virtual void set_cancel_callback(result<T>::cancel_callback_t) = 0;
    virtual bool has_cancel_callback() const = 0;
};

template <typename T, typename I>
requires(!std::is_reference_v<I>)
struct concrete_holder_ final : public holder_<T>
{
    explicit concrete_holder_(I&& i)
        : impl_(std::move(i))
    {
    }

    explicit concrete_holder_(const I& i)
        : impl_(i)
    {
    }

    inline ~concrete_holder_() = default;

    inline bool cancel() override
    {
        return impl_.cancel();
    }

    inline float progress() const override
    {
        if constexpr (has_<I>::progress) {
            return impl_.progress();
        } else {
            return is_ready();
        }
    }

    inline void set_priority(int priority) override
    {
        if constexpr (has_<I>::set_priority) {
            impl_.set_priority(priority);
        }
    }

    inline bool is_cancelled() const override
    {
        return impl_.is_cancelled();
    }

    inline bool has_exception() const override
    {
        return impl_.has_exception();
    }

    inline bool is_ready() const override
    {
        return impl_.is_ready();
    }

    inline T get() override
    {
        return std::move(impl_).get();
    }

    inline void set_callback(callback_type<T> c) override
    {
        impl_.set_callback(std::move(c));
    }

    inline void set_cancel_callback(result<T>::cancel_callback_t c) override
    {
        impl_.set_cancel_callback(std::move(c));
    }

    inline bool has_cancel_callback() const override
    {
        return impl_.has_cancel_callback();
    }

private:
    I impl_;
};

} // namespace impl

/**
 * @brief Represents generic class for any asynchronous promise which returns object of type `T`.
 * `promise` can be used to:
 * - check the state of the promise
 * - change the priority of the promise
 * - cancel the promise
 * - register callback, to be called when promise is ready
 *
 * The promise can be chained with other promises, using `then` method. If the then callback returns new promise, the
 * result promise will be the type of the returned promise.
 *
 * @tparam T Type of the result of the promise. If the promise is void, the type should be `void`.
 */
template <typename T>
class promise
{
public:
    using result_type = T;
    using cancel_callback_t = typename result<T>::cancel_callback_t;

public:
    /// @brief Default constructor. Creates null promise.
    promise() = default;

    promise(const promise& s) = delete;
    promise& operator=(const promise& s) = delete;
    promise(promise&&) noexcept = default;
    promise& operator=(promise&& h) noexcept = default;

    template <typename I>
    requires(!std::is_same_v<promise, std::decay_t<I>>)
    explicit promise(I&& impl)
        : impl_(std::make_unique<impl::concrete_holder_<T, std::remove_cvref_t<I>>>(std::forward<I>(impl)))
    {
    }

    ~promise() = default;

public:
    /// @brief Checks if the promise is null or not.
    inline explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

public:
    /**
     * @brief Cancels the promise.
     * This function tries to cancel any ongoing async task under the hood. It ensures that the
     * callback will not be called.
     */
    inline bool cancel()
    {
        return (impl_ ? impl_->cancel() : true);
    }

    /// Checks if the promise is cancelled.
    inline bool is_cancelled() const
    {
        return impl_ ? impl_->is_cancelled() : false;
    }

    /**
     * @brief Set the priority of the promise.
     *
     * @param priority
     */
    inline void set_priority(int priority)
    {
        if (impl_) {
            return impl_->set_priority(priority);
        }
    }

    /**
     * @brief Checks if the promise is ready, so the result can be get.
     *
     * @return true
     * @return false
     */
    inline bool is_ready() const
    {
        return impl_->is_ready();
    }

    /**
     * @brief Checks if the promise is ready and has a successful result (not an exception).
     * @pre The promise should be ready.
     *
     * @return true If the promise is ready and has a successful result.
     */
    bool has_exception() const
    {
        return impl_->has_exception();
    }

    /**
     * @brief Returns the result of the promise.
     * @pre The promise should be ready.
     *
     * @return T The result.
     */
    inline T get() &&
    {
        return impl_->get();
    }

    /**
     * @brief Returns the progress of the promise.
     */
    float progress() const noexcept
    {
        return impl_->progress();
    }

    /**
     * @brief Returns a future object whose resolved value is the result of the pormise,
     * or re-raises the exception in case of failure.
     *
     * @pre The promise should have no callback registered. This is a mutually exclusive interface with set_callback.
     * @warning The future will be forever locked if the promise will be cancelled.
     */
    [[nodiscard]] inline std::future<T> get_future()
    {
        std::promise<T> p;
        auto future = p.get_future();
        set_callback([p = std::move(p)](result<T>&& value) mutable {
            if (value.succeeded()) {
                if constexpr (std::is_void_v<T>) {
                    p.set_value();
                } else {
                    p.set_value(std::move(value).value());
                }
            } else {
                p.set_exception(std::move(value).error());
            }
        });
        return future;
    }

    /**
     * @brief Set the callback to the handle, which will be called when the promise is fulfilled.
     * The callback can be set only once and it can't be reset.
     *
     * @param c Callback.
     */
    inline void set_callback(callback_type<T> c)
    {
        return impl_->set_callback(std::move(c));
    }

    /**
     * @brief Set the cancel callback to the handle, which will be called when the promise is cancelled and the result
     * is set.
     */
    inline void set_cancel_callback(cancel_callback_t c)
    {
        impl_->set_cancel_callback(std::move(c));
    }

    /// Checks if the cancel callback was set or inherited from the promise chain.
    inline bool has_cancel_callback() const
    {
        return impl_->has_cancel_callback();
    }

public:
    /**
     * Ensures the span ends after this promise ends.
     *
     * WARNING: The span starts object is created, so DO NOT construct the span when passing it to this function,
     * otherwise it may not line up correctly with spans created inside the promise being added to
     */
    [[nodiscard]] inline auto with(std::unique_ptr<base::logging_span_holder> span) &&
    {
        if (span == nullptr) {
            return std::move(*this);
        }
        if constexpr (std::is_void_v<T>) {
            return std::move(*this).then([span = std::move(span)]() {
            });
        } else {
            return std::move(*this).then([span = std::move(span)](T&& r) {
                return std::move(r);
            });
        }
    }

    /**
     * @brief Chains the given promise with the given functor, which is being synchronously applied
     * to the result of input promise. The output of the functor will be the result of the
     * returned promise. Optionally, chains the exception pointer with an additional functor when the
     * promise is unsuccessful.
     */
    template <typename Converter,
              typename ExceptionConverter = std::monostate,
              typename CancelCallback = std::monostate>
    requires((!std::is_void_v<T> && std::is_invocable_v<Converter, T &&>) ||
             (std::is_void_v<T> && std::is_invocable_v<Converter>))
    [[nodiscard]] inline auto then(Converter&& f,
                                   ExceptionConverter&& g = ExceptionConverter(),
                                   CancelCallback&& c = CancelCallback(),
                                   async::queue* q = &main()) &&
    {
        static_assert(std::is_same_v<ExceptionConverter, std::monostate> ||
                          std::is_nothrow_invocable_v<ExceptionConverter, std::exception_ptr>,
                      "The exception converter should not throw exceptions.");

        using To = std::remove_cvref_t<impl::invoke_result_maybe_void_t<Converter, T>>;
        using R = std::conditional_t<impl::is_promise_v<To>, To, promise<To>>;
        return R(impl::chained_promise<T, To>(std::move(*this),
                                              std::forward<Converter>(f),
                                              std::forward<ExceptionConverter>(g),
                                              std::forward<CancelCallback>(c),
                                              q));
    }

    /// Chain a function onto this promise, but run the function on bg queue.
    template <typename Converter, typename ExceptionConverter = std::monostate>
    [[nodiscard]] inline auto then_async(Converter&& f, ExceptionConverter&& g = ExceptionConverter()) &&
    {
        return std::move(*this).then(std::forward<Converter>(f), std::forward<ExceptionConverter>(g), {}, &bg());
    }

    /// Chain a function onto this promise that can be run on any queue.
    template <typename Converter, typename ExceptionConverter = std::monostate>
    [[nodiscard]] inline auto then_any(Converter&& f, ExceptionConverter&& g = ExceptionConverter()) &&
    {
        return std::move(*this).then(std::forward<Converter>(f), std::forward<ExceptionConverter>(g), {}, nullptr);
    }

private:
    std::unique_ptr<impl::holder_<T>> impl_;
};

} // namespace async

#include "impl/multiple_promises.hpp"

namespace async {

/**
 * @brief Returns fulfilled promise with the given value.
 */
template <typename T>
requires(!std::is_void_v<T> && !std::is_same_v<T, std::exception_ptr>)
[[nodiscard]] inline promise<T> fulfilled(T v)
{
    return promise<T>(impl::fulfilled_promise<T>(std::move(v)));
}

/**
 * @brief Returns fulfilled promise with the default value.
 */
template <typename T>
requires(!std::is_void_v<T> && !std::is_same_v<T, std::exception_ptr>)
[[nodiscard]] inline promise<T> fulfilled()
{
    return promise<T>(impl::fulfilled_promise<T>());
}

/**
 * @brief Returns fulfilled promise with the given value.
 */
[[nodiscard]] inline promise<void> fulfilled()
{
    return promise<void>(impl::fulfilled_promise<void>());
}

/**
 * @brief Returns fulfilled promise failed with the given error.
 */
template <typename T, typename E>
requires(!std::is_same_v<T, std::exception_ptr>)
[[nodiscard]] inline promise<T> error(E e)
{
    if constexpr (std::is_same_v<E, std::exception_ptr>) {
        return promise<T>(impl::error_promise<T>(e));
    } else {
        return promise<T>(impl::error_promise<T>(std::make_exception_ptr(std::move(e))));
    }
}

/**
 * @brief Converts the list of promises with the same type into single promise, with the result
 * of all input promise results.
 */
template <typename T>
requires(!std::is_void_v<T>)
[[nodiscard]] inline promise<std::vector<T>> combine(std::vector<promise<T>>&& promises)
{
    if (promises.empty()) {
        return fulfilled<std::vector<T>>();
    }
    return promise<std::vector<T>>(impl::multiple_promises<T>(std::move(promises)));
}

/**
 * @brief Converts the list of promises with the same type into single promise, with the result
 * of all input promise results.
 */
template <typename T>
requires(std::is_void_v<T>)
[[nodiscard]] inline promise<void> combine(std::vector<promise<T>>&& promises)
{
    if (promises.empty()) {
        return fulfilled();
    }
    return promise<void>(impl::multiple_promises<T>(std::move(promises)));
}

/**
 * @brief Converts the list of variadic promises into single promise with the result of all input promise results
 * as tuple. of all input promise results.
 */
template <typename... Args>
[[nodiscard]] inline auto combine(promise<Args>&&... promises)
{
    if constexpr (sizeof...(Args) == 0) {
        return fulfilled(std::tuple<>());
    } else if constexpr (sizeof...(Args) == 1) {
        return std::move(promises...);
    } else {
        auto h = impl::variadic_promises<Args...>(std::make_tuple(std::move(promises)...));
        return promise<std::tuple<Args...>>(std::move(h));
    }
}

} // namespace async
