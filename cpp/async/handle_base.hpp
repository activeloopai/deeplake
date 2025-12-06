#pragma once

/**
 * @file handle_base.hpp
 * @brief Definition and implementation of the `handle_base` template class.
 */

#include "callback_type.hpp"
#include "result.hpp"

#include <async/async.hpp>
#include <base/assert.hpp>
#include <base/spin_lock.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace async {

template <typename T>
class promise;

template <typename Result, typename State = std::monostate>
class handle_base
{
    template <typename T>
    struct is_recursive
    {
        static constexpr bool value = false;
    };

    template <typename T>
    struct is_recursive<handle_base<T>>
    {
        static constexpr bool value = true;
    };

    template <typename T>
    struct is_recursive<promise<T>>
    {
        static constexpr bool value = true;
    };

    static_assert(
        !is_recursive<State>::value,
        "The class isn't supposed for storing promises recursively. Use promise::then instead.");

protected:
    struct data_type
    {
        explicit data_type(result<Result>&& r = result<Result>())
            : result_(std::move(r))
        {
        }

        explicit data_type(State&& s, result<Result>&& r = result<Result>())
            : result_(std::move(r))
            , state_(std::move(s))
        {
        }

        std::optional<result<Result>> result_;
        callback_type<Result> callback_;
        State state_;
        base::spin_lock lock_;
    };


    static inline bool cancel(const std::shared_ptr<data_type>& data)
    {
        if (is_cancelled(data)) {
            return true;
        }
        if (is_finished(data)) {
            return false;
        }
        data->result_->cancel();
        data->callback_ = nullptr;
        return true;
    }

    static inline bool is_initial(const std::shared_ptr<data_type>& data) noexcept
    {
        return data->result_.has_value() && data->result_->is_initial();
    }

    static inline bool has_result(const std::shared_ptr<data_type>& data) noexcept
    {
        return data->result_.has_value() && data->result_->fulfilled();
    }

    static inline bool is_cancelled(const std::shared_ptr<data_type>& data) noexcept
    {
        return data->result_.has_value() && data->result_->is_cancelled();
    }

    static inline bool is_finished(const std::shared_ptr<data_type>& data) noexcept
    {
        return !data->result_.has_value();
    }

    static inline bool is_ready(const std::shared_ptr<data_type>& data) noexcept
    {
        return has_result(data) || is_finished(data);
    }

    /// @name Types
    /// @{
public:
    using cancel_callback_t = result<Result>::cancel_callback_t;
    /// @}

    /// @name Special member functions
    /// @{
public:
    handle_base()
        : data_(std::make_shared<data_type>())
    {
        ASSERT(is_initial(data_));
    }

    explicit handle_base(cancel_callback_t c)
        : handle_base()
    {
        data_->result_->set_cancel_callback(std::move(c));
    }

    explicit handle_base(State s)
        : data_(std::make_shared<data_type>(std::move(s)))
    {
        ASSERT(is_initial(data_));
    }

    handle_base(State s, cancel_callback_t c)
        : handle_base(std::move(s))
    {
        data_->result_->set_cancel_callback(std::move(c));
    }

    explicit handle_base(result<Result>&& res)
        : data_(std::make_shared<data_type>(std::move(res)))
    {
        ASSERT(has_result(data_));
    }

    handle_base(result<Result>&& res, cancel_callback_t c)
        : handle_base(std::move(res))
    {
        ASSERT(has_result(data_));
        data_->result_->set_cancel_callback(std::move(c));
    }

    template <typename T>
    requires(!std::is_void_v<Result> && std::is_nothrow_convertible_v<T, Result>)
    explicit handle_base(T v)
        : handle_base(result<Result>(std::move(v)))
    {
    }

    explicit handle_base(std::exception_ptr ex)
        : handle_base(result<Result>(std::move(ex)))
    {
    }

    void destroy()
    {
        data_.reset();
    }
    /// @}

public:
    inline bool cancel()
    {
        std::scoped_lock lock(data_->lock_);
        return cancel(data_);
    }

    inline void set_priority(int) const
    {
        // Default implementation does not handle priority.
    }

    inline bool has_exception() const
    {
        std::scoped_lock lock(data_->lock_);
        return has_result(data_) && !data_->result_->succeeded();
    }

    inline Result get() &&
    {
        ASSERT(!is_finished(data_));

        // Return the result on success
        if (has_result(data_)) {
            auto r = std::move(*data_->result_);
            data_->result_.reset();
            return std::move(r).get();
        }

        // Should not call .get() more than once
        throw std::domain_error("Request already handled");
    }

    inline void set_callback(callback_type<Result> c)
    {
        std::unique_lock lock(data_->lock_);
        if (is_cancelled(data_)) {
            return;
        }
        ASSERT(static_cast<bool>(data_->callback_) == false);
        data_->callback_ = std::move(c);
        if (has_result(data_)) {
            call(std::move(lock), data_);
        }
    }

    inline void set_cancel_callback(cancel_callback_t c)
    {
        std::scoped_lock lock(data_->lock_);
        if (!is_finished(data_)) {
            data_->result_->set_cancel_callback(std::move(c));
        }
    }

    inline bool has_cancel_callback() const noexcept
    {
        std::scoped_lock lock(data_->lock_);
        return data_->result_.has_value() && data_->result_->has_cancel_callback();
    }

#ifdef _WIN32
    template <typename T>
    requires(!std::is_void_v<Result>)
    inline void set_value(T&& value)
    {
        std::unique_lock lock(data_->lock_);
        ASSERT(!has_result(data_));
        data_->result_ = Result(std::forward<T>(value));
        on_result_updated(std::move(lock), data_);
    }
#else
    template <typename T>
    requires(!std::is_void_v<Result> && std::is_nothrow_convertible_v<T, Result>)
    inline void set_value(T&& value)
    {
        std::unique_lock lock(data_->lock_);
        ASSERT(!has_result(data_));
        data_->result_ = Result(std::forward<T>(value));
        on_result_updated(std::move(lock), data_);
    }
#endif

    inline void set_value()
    requires(std::is_void_v<Result>)
    {
        std::unique_lock lock(data_->lock_);
        ASSERT(!has_result(data_));
        data_->result_->set_fulfilled();
        on_result_updated(std::move(lock), data_);
    }

    inline void set_exception(std::exception_ptr ex)
    {
        std::unique_lock lock(data_->lock_);
        ASSERT(!has_result(data_) || !data_->result_->succeeded());
        data_->result_ = std::move(ex);
        on_result_updated(std::move(lock), data_);
    }

    inline bool is_ready() const noexcept
    {
        std::scoped_lock lock(data_->lock_);
        return is_ready(data_);
    }

    inline bool is_cancelled() const noexcept
    {
        std::scoped_lock lock(data_->lock_);
        return is_cancelled(data_);
    }

    inline bool is_finished() const noexcept
    {
        std::scoped_lock lock(data_->lock_);
        return is_finished(data_);
    }

    inline bool is_unresolved() const noexcept
    {
        std::scoped_lock lock(data_->lock_);
        return is_initial(data_);
    }

    inline const State& state() const noexcept
    {
        return data_->state_;
    }

    inline State& state() noexcept
    {
        return data_->state_;
    }

    inline void set_state(State&& s)
    {
        data_->state_ = std::move(s);
    }

    inline void set_state(const State& s)
    {
        data_->state_ = s;
    }

    inline const std::shared_ptr<data_type>& data() const noexcept
    {
        return data_;
    }

private:
    static inline void on_result_updated(std::unique_lock<base::spin_lock> lock, const std::shared_ptr<data_type>& data)
    {
        ASSERT(lock.owns_lock());
        ASSERT(!is_finished(data));
        if (data->callback_ && !is_cancelled(data)) {
            ASSERT(has_result(data));
            call(std::move(lock), data);
        }
    }

    static inline void call(std::unique_lock<base::spin_lock> lock, const std::shared_ptr<data_type>& data)
    {
        ASSERT(!is_cancelled(data));
        callback_type<Result> callback = std::move(data->callback_);
        ASSERT(has_result(data));
        result<Result> res = std::move(*data->result_);
        data->result_.reset();

        lock.unlock();
        callback(std::move(res));
    }

private:
    std::shared_ptr<data_type> data_;
};

} // namespace async
