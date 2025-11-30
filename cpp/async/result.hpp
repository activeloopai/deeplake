#pragma once

/**
 * @file result.hpp
 * @brief Definition and implementation of the `result` template class.
 */

#include <base/assert.hpp>
#include <base/function.hpp>

#include <exception>
#include <type_traits>
#include <variant>

namespace async {

namespace impl {

struct initial_state
{
};

struct cancelled_state
{
};


class result_base
{
public:
    using cancel_callback_t = base::function<void()>;

    /// Access to the cancel callback.
    void set_cancel_callback(cancel_callback_t c) noexcept
    {
        cancel_callback_ = std::move(c);
    }

    /// Checks if the cancel callback was set.
    inline bool has_cancel_callback() const noexcept
    {
        return cancel_callback_ != nullptr;
    }

    /// Returns the cancel callback.
    cancel_callback_t get_cancel_callback() noexcept
    {
        return std::move(cancel_callback_);
    }

protected:
    result_base() noexcept = default;

    inline void do_cancel()
    {
        if (cancel_callback_) [[unlikely]] {
            auto c = std::move(cancel_callback_);
            c();
        }
    }

private:
    cancel_callback_t cancel_callback_;
};

} // namespace impl

/**
 * @brief Represents the result obtained by async call. The result can be accessed by `get` function.
 * If the async call was failed then the corresponding exception being rethrown by `get` function.
 *
 * @tparam T 
 */
template <typename T>
class result : public impl::result_base
{
    static_assert(std::is_reference_v<T> == false,
                  "Currently reference type is not allowed as a result of async call.");
    static_assert(!std::is_same_v<T, std::exception_ptr>,
                  "`std::exception_ptr` is not allowed as a result of async call.");

public:
    using result_type = T;

public:
    /// @brief Default constructor.
    result() noexcept = default;
    
    /// @brief Constructor by result.
    result(T&& v) noexcept
        : data_(std::move(v))
    {
    }

    /// @brief Constructor by result.
    result(const T& v)
    requires std::is_copy_constructible_v<T>
        : data_(v)
    {
    }

    /// @brief Constructor by error.
    result(std::exception_ptr e) noexcept
        : data_(std::move(e))
    {
    }

    /// @brief Assignment by result.
    const result& operator=(T&& v) noexcept
    {
        if (is_cancelled()) {
            do_cancel();
        } else {
            data_ = std::move(v);
        }
        return *this;
    }

    /// @brief Assignment by error.
    const result& operator=(std::exception_ptr e) noexcept
    {
        if (!is_cancelled()) {
            data_ = std::move(e);
        }
        return *this;
    }

public:
    /// @brief Checks if the state is initial.
    inline bool is_initial() const noexcept
    {
        return data_.index() == 0;
    }

    /// @brief Checks if the result is cancelled.
    inline bool is_cancelled() const noexcept
    {
        return data_.index() == 1;
    }

    /// @brief Checks if the result is not empty.
    inline bool fulfilled() const noexcept
    {
        return data_.index() > 1;
    }

    /// @brief Checks if the result was succeeded and can be obtained.
    inline bool succeeded() const noexcept
    {
        return data_.index() == 2;
    }

    /// @brief Checks if the result has an exception.
    inline bool has_exception() const noexcept
    {
        return data_.index() == 3;
    }

    /// @brief Returns error.
    /// @pre The result should not be succeeded.
    std::exception_ptr error() && noexcept
    {
        ASSERT(data_.index() == 3);
        return std::move(std::get<3>(data_));
    }

    /// @brief Returns error.
    /// @pre The result should not be succeeded.
    const std::exception_ptr& error() const & noexcept
    {
        ASSERT(data_.index() == 3);
        return std::get<3>(data_);
    }

    /// @brief Returns result.
    /// @pre The result should be succeeded.
    T value() && noexcept
    {
        ASSERT(data_.index() == 2);
        return std::move(std::get<2>(data_));
    }

    /// @brief Returns result.
    /// @pre The result should be succeeded.
    const T& value() const & noexcept
    {
        ASSERT(data_.index() == 2);
        return std::get<2>(data_);
    }

    /// @brief Returns result.
    /// @pre The result should be succeeded.
    T& value() & noexcept
    {
        ASSERT(data_.index() == 2);
        return std::get<2>(data_);
    }

    /// @brief Returns the result if the result was succeeded. Otherwise throws the corresponding error.
    T get() &&
    {
        ASSERT(data_.index() > 1);
        if (data_.index() == 3) {
            std::rethrow_exception(std::move(std::get<3>(data_)));
        }
        return std::move(std::get<2>(data_));
    }

    /// @brief Returns the result if the result was succeeded. Otherwise throws the corresponding error.
    const T& get() const &
    {
        ASSERT(data_.index() > 1);
        if (data_.index() == 3) {
            std::rethrow_exception(std::get<3>(data_));
        }
        return std::get<2>(data_);
    }

    /// @brief Returns the result if the result was succeeded. Otherwise throws the corresponding error.
    T& get() &
    {
        ASSERT(data_.index() > 1);
        if (data_.index() == 3) {
            std::rethrow_exception(std::move(std::get<3>(data_)));
        }
        return std::get<2>(data_);
    }

    /**
     * @brief Returns the result if the result was succeeded. Otherwise returns the passed result.
     * @param val Value to return in case of error.
     */
    T get_default(T val = T()) 
    requires(std::is_copy_constructible_v<T>)
    {
        ASSERT(data_.index() > 1);
        if (data_.index() == 3) {
            return val;
        } else {
            return std::get<2>(data_);
        }
    }

    /// Call the cancel callback if it was set.
    void cancel()
    {
        if (succeeded()) {
            do_cancel();
        }
        data_ = impl::cancelled_state();
    }

private:
    std::variant<impl::initial_state, impl::cancelled_state, T, std::exception_ptr> data_;
    cancel_callback_t cancel_callback_;
};

/**
 * @brief Represents the result of the `void` async call.
 */
template <>
class result<void> : public impl::result_base
{
public:
    using result_type = void;

public:
    /// @brief Default constructor.
    result() noexcept = default;
    
    /// @brief Constructor by error.
    result(std::exception_ptr e) noexcept
        : data_(std::move(e))
    {
    }

    /**
     * @brief Constructor by fulfilled state.
     * @param fulfilled If the results was fulfilled or not. `false` result is same as default constructor.
     */
    result(bool fulfilled) noexcept
    {
        if (fulfilled) {
            set_fulfilled();
        }
    }

    /// @brief Set the fulfilled state.
    inline void set_fulfilled() noexcept
    {
        data_ = std::exception_ptr();
    }

    /// @brief Assign by error.
    const result& operator=(std::exception_ptr e) noexcept
    {
        ASSERT(fulfilled() == false);
        if (is_cancelled()) {
            if (e == nullptr) {
                do_cancel();
            }
        } else {
            data_ = std::move(e);
        }
        return *this;
    }

public:
    /// @brief Checks if the state is initial.
    inline bool is_initial() const noexcept
    {
        return data_.index() == 0;
    }
    
    /// @brief Checks if the state is cancelled.
    inline bool is_cancelled() const noexcept
    {
        return data_.index() == 1;
    }

    /// @brief Checks whether result is not empty.
    inline bool fulfilled() const noexcept
    {
        return data_.index() == 2;
    }

    /// @brief Checks if the result was succeeded.
    inline bool succeeded() const noexcept
    {
        return fulfilled() && std::get<2>(data_) == nullptr;
    }

    /// @brief Checks if the result has an exception.
    inline bool has_exception() const noexcept
    {
        return fulfilled() && std::get<2>(data_) != nullptr;
    }

    /// @brief Returns error.
    /// @pre The result should not be succeeded.
    std::exception_ptr error() && noexcept
    {
        ASSERT(fulfilled());
        ASSERT(std::get<2>(data_) != nullptr);
        return std::move(std::get<2>(data_));
    }

    /// @brief Returns error.
    /// @pre The result should not be succeeded.
    const std::exception_ptr& error() const & noexcept
    {
        ASSERT(fulfilled());
        ASSERT(std::get<2>(data_) != nullptr);
        return std::get<2>(data_);
    }

    /// @brief Throws the error if result was failed. Otherwise does nothing.
    void get() &&
    {
        ASSERT(fulfilled());
        if (std::get<2>(data_) != nullptr) {
            std::rethrow_exception(std::move(std::get<2>(data_)));
        }
    }

    /// @brief Throws the error if result was failed. Otherwise does nothing.
    void get() const &
    {
        ASSERT(fulfilled());
        if (std::get<2>(data_) != nullptr) {
            std::rethrow_exception(std::get<2>(data_));
        }
    }

    /// Call the cancel callback if it was set.
    void cancel()
    {
        if (succeeded()) {
            do_cancel();
        }
        data_ = impl::cancelled_state();
    }

private:
    std::variant<impl::initial_state, impl::cancelled_state, std::exception_ptr> data_;
    cancel_callback_t cancel_callback_;
};

}
