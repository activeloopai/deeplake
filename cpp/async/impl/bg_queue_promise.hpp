#pragma once

#include <async/async.hpp>
#include <async/context.hpp>
#include <async/handle_base.hpp>
#include <async/queue.hpp>

#include <atomic>
#include <tuple>

namespace async::impl {

struct bg_queue_state_t
{
    bg_queue_state_t() = default;

    async::queue::id_type id_;
    async::context context_;
};

template <typename Result>
class bg_queue_promise : public handle_base<Result, bg_queue_state_t>
{
    using base_t = handle_base<Result, bg_queue_state_t>;
    using cancel_callback_t = typename base_t::cancel_callback_t;

public:
    bg_queue_promise() = default;

    explicit bg_queue_promise(cancel_callback_t&& c)
        : base_t(std::move(c))
    {
    }

    template <typename Functor>
    requires (std::is_invocable_v<Functor> || std::is_invocable_v<Functor, context&>)
    inline void call(Functor&& f)
    {
        if (base_t::is_cancelled()) {
            return;
        }
        try {
            if constexpr (std::is_void_v<Result>) {
                if constexpr (std::is_invocable_v<Functor>) {
                    f();
                } else {
                    f(base_t::state().context_);
                }
                base_t::set_value();
            } else {
                if constexpr (std::is_invocable_v<Functor>) {
                    base_t::set_value(f());
                } else {
                    base_t::set_value(f(base_t::state().context_));
                }
            }
        } catch (...) {
            base_t::set_exception(std::current_exception());
        }
    }

    void set_priority(int priority)
    {
        id().set_priority(priority);
    }

    bool cancel()
    {
        const bool res = base_t::cancel();
        if (res) {
            base_t::state().context_.is_cancelled_ = true;
        }
        id().remove();
        return res;
    }

    float progress() const noexcept
    {
        return base_t::state().context_.progress;
    }

    queue::id_type& id() noexcept
    {
        return base_t::state().id_;
    }
};

}
