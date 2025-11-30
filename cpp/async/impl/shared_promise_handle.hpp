#pragma once

#include "../handle_base.hpp"

namespace async::impl {

template <typename promise_type>
class shared_promise_handle : public handle_base<typename promise_type::result_type, promise_type*>
{
    using R = typename promise_type::result_type;
    using base_t = handle_base<R, promise_type*>;

public:
    shared_promise_handle() = default;

    explicit shared_promise_handle(promise_type* t)
        : base_t(t)
    {
    }

    void set_index(int i)
    {
        index_ = i;
    }

    void set_priority(int priority)
    {
        if (!base_t::is_cancelled()) {
            ASSERT(promise());
            promise()->set_handle_priority(index_, priority);
        }
    }

    bool cancel()
    {
        if (base_t::cancel()) {
            promise()->revoke_handle(index_);
            return true;
        }
        return false;
    }

private:
    inline promise_type* promise()
    {
        return base_t::state();
    }

private:
    int index_ = 0L;
};

}
