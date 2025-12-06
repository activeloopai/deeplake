#pragma once

#include "fulfilled_promise.hpp"
#include "mpl.hpp"

#include <memory>
#include <optional>
#include <variant>

namespace async::impl {

template <typename From, typename ConverterResult>
class chained_promise
{
    using result_type = std::conditional_t<is_promise_v<ConverterResult>, ConverterResult, result<ConverterResult>>;
    using To = result_type::result_type;

public:
    template <typename ResultConverter, typename ExceptionConverter, typename CancelCallback>
    chained_promise(promise<From>&& source, ResultConverter&& f, ExceptionConverter&& g, CancelCallback&& c, queue* q)
        : data_(std::make_shared<data>(std::move(source), std::forward<CancelCallback>(c)))
    {
        std::get<0>(data_->source)
            .set_callback({[d = data_, f = std::forward<ResultConverter>(f), g = std::forward<ExceptionConverter>(g)](
                               result<From>&& v) mutable {
                               if (store_cancel_callback_of_from(d, v)) {
                                   result_type r = convert(d, std::move(v), std::move(f), std::move(g));
                                   set_result_(std::move(d), std::move(r));
                               }
                           },
                           q});
    }

    chained_promise(const chained_promise&) = delete;
    chained_promise& operator=(const chained_promise&) = delete;
    chained_promise(chained_promise&&) noexcept = default;
    chained_promise& operator=(chained_promise&&) noexcept = delete;
    ~chained_promise() = default;

    inline void set_priority(int priority)
    {
        std::scoped_lock lock(data_->lock);
        if (data_->source.index() == 0) {
            std::get<0>(data_->source).set_priority(priority);
        }
    }

    inline bool cancel()
    {
        std::scoped_lock lock(data_->lock);
        if (data_->is_cancelled) {
            return true;
        }
        if (data_->is_done) {
            return false;
        }
        if (data_->source.index() == 0) {
            std::get<0>(data_->source).cancel();
        } else {
            take_result(data_).cancel();
        }
        data_->is_cancelled = true;
        data_->callback.reset();
        return true;
    }

    inline bool is_cancelled() const
    {
        std::scoped_lock lock(data_->lock);
        return data_->is_cancelled;
    }

    inline float progress() const
    {
        std::scoped_lock lock(data_->lock);
        if (data_->source.index() == 0) {
            return std::get<0>(data_->source).progress();
        }
        if constexpr (is_promise_v<result_type>) {
            return std::get<1>(data_->source).progress() * 0.5f + 0.5f;
        } else {
            return 1.0f;
        }
    }

    inline bool is_ready() const
    {
        std::scoped_lock lock(data_->lock);
        if (data_->is_cancelled || data_->source.index() != 1) {
            return false;
        }
        if constexpr (is_promise_v<result_type>) {
            return std::get<1>(data_->source).is_ready();
        } else {
            return true;
        }
    }

    inline bool has_exception() const
    {
        std::scoped_lock lock(data_->lock);
        if (data_->is_cancelled || data_->source.index() != 1) {
            return false;
        }
        return std::get<1>(data_->source).has_exception();
    }

    inline To get() &&
    {
        ASSERT(is_ready());
        std::scoped_lock lock(data_->lock);
        return take_result(data_).get();
    }

    inline void set_callback(callback_type<To> c)
    requires (is_promise_v<result_type>)
    {
        std::unique_lock lock(data_->lock);
        if (data_->is_cancelled) {
            return;
        }
        ASSERT(data_->is_done == false);
        if (data_->source.index() == 0) {
            data_->callback = std::move(c);
        } else {
            result_type&& p = take_result(data_);
            data_->is_done = true;
            lock.unlock();
            p.set_callback(std::move(c));
        }
    }

    void set_callback(callback_type<To> c)
    requires (!is_promise_v<result_type>)
    {
        std::unique_lock lock(data_->lock);
        if (data_->is_cancelled) {
            return;
        }
        ASSERT(data_->is_done == false);
        if (data_->source.index() == 0) {
            data_->callback = std::move(c);
        } else {
            auto r = take_result(data_);
            data_->is_done = true;
            lock.unlock();
            c(std::move(r));
        }
    }

    void set_cancel_callback(result_type::cancel_callback_t c)
    {
        std::scoped_lock lock(data_->lock);
        if (!data_->is_cancelled && !data_->is_done) {
            data_->to_cancel = std::move(c);
        }
    }

    bool has_cancel_callback() const
    {
        std::scoped_lock lock(data_->lock);
        if (data_->from_cancel || data_->to_cancel) {
            return true;
        }
        if (data_->source.index() == 0) {
            return false;
        }
        return std::get<1>(data_->source).has_cancel_callback();
    }

private:
    struct data;

    static inline bool store_cancel_callback_of_from(const std::shared_ptr<data>& d, result<From>& v)
    {
        std::scoped_lock lock(d->lock);
        if (d->is_cancelled) {
            v.cancel();
            return false;
        }
        d->from_cancel = v.get_cancel_callback();
        return true;
    }

    static inline void handle_converter_exception(const std::shared_ptr<data>& d)
    {
        std::scoped_lock lock(d->lock);
        if (d->from_cancel) {
            d->from_cancel();
            d->from_cancel = nullptr;
        } else if (d->to_cancel) {
            d->to_cancel();
            d->to_cancel = nullptr;
        }
    }

    template <typename ResultConverter, typename ExceptionConverter>
    static inline result_type
    convert(const std::shared_ptr<data>& d, result<From>&& v, ResultConverter&& f, ExceptionConverter&& e)
    requires(is_promise_v<result_type>)
    {
        promise<To> p;
        if (v.succeeded()) {
            try {
                if constexpr (std::is_void_v<From>) {
                    p = f();
                } else {
                    p = f(std::move(v).value());
                }
            } catch (...) {
                handle_converter_exception(d);
                if constexpr (std::is_same_v<ExceptionConverter, std::monostate>) {
                    p = promise<To>(error_promise<To>(std::current_exception()));
                } else {
                    p = e(std::current_exception());
                }
            }
        } else {
            if constexpr (std::is_same_v<ExceptionConverter, std::monostate>) {
                p = promise<To>(error_promise<To>(std::move(v).error()));
            } else {
                p = e(std::move(v).error());
            }
        }
        return p;
    }

    template <typename ResultConverter, typename ExceptionConverter>
    static inline result_type
    convert(const std::shared_ptr<data>& d, result<From>&& v, ResultConverter&& f, ExceptionConverter&& e)
    requires(!is_promise_v<result_type>)
    {
        result<To> r;
        const bool was_succeeded = v.succeeded();
        try {
            if constexpr (std::is_void_v<From> && std::is_void_v<To>) {
                std::move(v).get();
                f();
                r = result<void>(true);
            } else if constexpr (std::is_void_v<From>) {
                std::move(v).get();
                r = f();
            } else if constexpr (std::is_void_v<To>) {
                f(std::move(v).get());
                r = result<void>(true);
            } else {
                r = f(std::move(v).get());
            }
        } catch (...) {
            if (was_succeeded) {
                handle_converter_exception(d);
            }
            if constexpr (std::is_same_v<ExceptionConverter, std::monostate>) {
                r = std::current_exception();
            } else {
                try {
                    if constexpr (std::is_void_v<std::invoke_result_t<ExceptionConverter, std::exception_ptr>>) {
                        e(std::current_exception());
                        r = result<void>(true);
                    } else {
                        r = e(std::current_exception());
                    }
                } catch (...) {
                    r = std::current_exception();
                }
            }
        }
        return r;
    }

    static inline void set_cancel_callback(const std::shared_ptr<data>& d, result_type& r)
    {
        if (d->to_cancel) {
            r.set_cancel_callback(std::move(d->to_cancel));
            d->from_cancel = nullptr;
        } else if (r.has_cancel_callback()) {
            d->from_cancel = nullptr;
        } else if (d->from_cancel) {
            if constexpr (std::is_void_v<To>) {
                r.set_cancel_callback(std::move(d->from_cancel));
            } else {
                r.set_cancel_callback(std::move(d->from_cancel));
            }
        }
    }

    static inline result_type&& take_result(const std::shared_ptr<data>& d)
    {
        ASSERT(d->source.index() == 1);
        auto& v = std::get<1>(d->source);
        set_cancel_callback(d, v);
        return std::move(v);
    }

    // data should be passed by value because the source modification may destroy the passed lambda which is storing the
    // shared pointer to data
    static inline void set_result_(std::shared_ptr<data> d, result_type&& value)
    {
        std::unique_lock lock(d->lock);
        // the following line will destroy calling callback
        d->source.template emplace<1>(std::move(value));
        if (d->is_cancelled) {
            take_result(d).cancel();
        } else if (d->callback.has_value()) {
            auto c = std::move(*(d->callback));
            // v is not moved in this case
            auto&& v = take_result(d);
            ASSERT(d->is_done == false);
            d->is_done = true;
            lock.unlock();
            done(c, v);
        }
    }

    static inline void done(callback_type<To>& c, result_type& v)
    {
        if constexpr (is_promise_v<result_type>) {
            v.set_callback(std::move(c));
        } else {
            c(std::move(v));
        }
    }

private:
    struct data
    {
        data(promise<From>&& s, std::monostate)
            : source(std::in_place_index_t<0>(), std::move(s))
        {
        }

        data(promise<From>&& s, result_type::cancel_callback_t c)
            : source(std::in_place_index_t<0>(), std::move(s))
            , to_cancel(std::move(c))
        {
        }

        std::variant<promise<From>, result_type> source;
        std::optional<callback_type<To>> callback;
        base::function<void()> from_cancel;
        result_type::cancel_callback_t to_cancel;
        base::spin_lock lock;
        bool is_cancelled = false;
        bool is_done = false;
    };

    std::shared_ptr<data> data_;
};

}
