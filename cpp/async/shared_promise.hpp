#pragma once

#include "exceptions.hpp"
#include "impl/shared_promise_handle.hpp"
#include "promise.hpp"
#include "result.hpp"

#include <icm/colony.hpp>

#include <memory>
#include <variant>

namespace async {

/**
 * @brief A shared promise is a promise which can serve multiple callbacks and can be used to notify multiple listeners.
 */
template <typename T>
class shared_promise
{
public:
    shared_promise() = default;

    shared_promise(promise<T> promise)
        : impl_(std::make_shared<impl_t>(std::move(promise)))
    {
        std::get<0>(impl_->promise_).set_callback([impl = impl_](result<T>&& result) {
            impl->set_result(std::move(result));
        });
    }

    promise<T> listen()
    {
        return impl_->listen();
    }

    void cancel()
    {
        impl_->cancel();
    }

    int num_listeners() const
    {
        return impl_->num_listeners();
    }

    bool has_listeners() const
    {
        return impl_->num_listeners() > 0;
    }

    explicit operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

private:
    struct impl_t
    {
        using result_type = T;

        explicit impl_t(promise<T> promise)
            : promise_(std::move(promise))
        {
        }

        static auto get_handle_value_setter(const result<T>& r)
        {
            if constexpr (std::is_void_v<T>) {
                return [](impl::shared_promise_handle<impl_t>& handle) {
                    handle.set_value();
                };
            } else {
                return [val = r.value()](impl::shared_promise_handle<impl_t>& handle) {
                    handle.set_value(T(val));
                };
            }
        }

        void set_result(result<T>&& result)
        {
            icm::colony<impl::shared_promise_handle<impl_t>> handles;
            std::unique_lock lock(lock_);
            std::swap(handles_, handles);
            if (is_cancelled()) {
                ASSERT(handles.empty());
                return;
            }
            if (result.succeeded()) {
                const auto value_setter = get_handle_value_setter(result);
                promise_ = std::move(result);
                lock.unlock();
                for (auto& handle : handles) {
                    value_setter(handle);
                }
            } else {
                const auto error = result.error();
                promise_ = std::move(result);
                lock.unlock();
                for (auto& handle : handles) {
                    handle.set_exception(error);
                }
            }
        }

        promise<T> listen()
        {
            std::scoped_lock lock(lock_);
            if (is_cancelled()) {
                throw shared_promise_cancelled();
            }
            if (promise_.index() == 1) {
                const result<T>& r = std::get<1>(promise_);
                if (r.succeeded()) {
                    if constexpr (std::is_void_v<T>) {
                        return fulfilled();
                    } else {
                        return fulfilled(r.value());
                    }
                } else {
                    return error<T>(r.error());
                }
            }
            auto i = handles_.emplace(this);
            handles_[i].set_index(i);
            return promise<T>(handles_[i]);
        }

        void cancel()
        {
            std::scoped_lock lock(lock_);
            if (promise_.index() == 0) {
                if (!handles_.empty()) {
                    throw cannot_cancel_shared_promise();
                }
                std::get<0>(promise_).cancel();
            }
            promise_ = std::monostate();
        }

        void set_handle_priority(int, int priority)
        {
            std::scoped_lock lock(lock_);
            if (promise_.index() == 0) {
                std::get<0>(promise_).set_priority(priority);
            }
        }

        int num_listeners()
        {
            std::scoped_lock lock(lock_);
            return handles_.size();
        }

        void revoke_handle(int index)
        {
            std::scoped_lock lock(lock_);
            if (!handles_.empty()) {
                handles_.erase(index);
            }
        }

        bool is_cancelled() const
        {
            return std::holds_alternative<std::monostate>(promise_);
        }

        std::variant<promise<T>, result<T>, std::monostate> promise_;
        icm::colony<impl::shared_promise_handle<impl_t>> handles_;
        base::spin_lock lock_;
    };

    friend class impl::shared_promise_handle<impl_t>;

private:
    std::shared_ptr<impl_t> impl_;
};

}
