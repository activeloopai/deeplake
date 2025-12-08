#pragma once

#include "../handle_base.hpp"

#include <tuple>

namespace async::impl {

template <typename... Args>
struct variadic_promises_state_t
{
    using promises_type = std::tuple<promise<Args>...>;
    using result_type = std::tuple<Args...>;

    explicit variadic_promises_state_t(promises_type&& ps)
        : promises_(std::move(ps))
        , results_(result_type(Args()...))
    {
    }

    variadic_promises_state_t(variadic_promises_state_t&& other) noexcept
        : promises_(std::move(other.promises_))
        , results_(std::move(other.results_))
        , fulfilled_count_(other.fulfilled_count_.load())
    {
    }

    promises_type promises_;
    result_type results_;
    std::atomic<std::size_t> fulfilled_count_ = 0;
};

template <typename... Args>
class variadic_promises : public handle_base<std::tuple<Args...>, variadic_promises_state_t<Args...>>
{
    using state_t = variadic_promises_state_t<Args...>;
    using promises_type = typename state_t::promises_type;
    using result_type = typename state_t::result_type;
    using base_t = handle_base<result_type, state_t>;

    using typename base_t::data_type;

    static constexpr auto total_size = sizeof...(Args);

    template <int id>
    using element_type = typename std::tuple_element<id, result_type>::type;

public:
    explicit variadic_promises(promises_type&& ps)
        : base_t(state_t(std::move(ps)))
    {
        for_each_element(promises(), [this]<int id>(auto& ps) {
            std::get<id>(ps).set_callback({[*this](auto v) mutable {
                                               if (base_t::is_unresolved()) {
                                                   try {
                                                       set_result_<id>(std::move(v).get());
                                                   } catch (...) {
                                                       base_t::set_exception(std::current_exception());
                                                   }
                                               }
                                               /// Manually reseting shared_ptr to avoid cyclic dependency.
                                               base_t::destroy();
                                           },
                                           nullptr});
        });
    }

    variadic_promises(const variadic_promises&) = default;
    variadic_promises& operator=(const variadic_promises&) = delete;
    variadic_promises(variadic_promises&&) noexcept = default;
    variadic_promises& operator=(variadic_promises&&) = delete;
    ~variadic_promises() = default;

    void set_priority(int priority)
    {
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        if (base_t::is_cancelled(data)) {
            return;
        }
        for_each_element(promises(), [priority]<int id>(auto& ps) {
            auto& p = std::get<id>(ps);
            if (p) {
                p.set_priority(priority);
            }
        });
    }

    bool cancel()
    {
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        if (!base_t::cancel(data)) {
            return false;
        }
        for_each_element(promises(), []<int id>(auto& ps) {
            auto& p = std::get<id>(ps);
            if (p) {
                p.cancel();
                p = {};
            }
        });
        return true;
    }

    float progress() const
    {
        float p = 0.0f;
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        for_each_element(promises(), [&p]<int id>(const auto& ps) {
            p += std::get<id>(ps).progress();
        });
        return p / sizeof...(Args);
    }

private:
    template <int id>
    void set_result_(element_type<id>&& r)
    {
        std::get<id>(results()) = std::move(r);
        const auto count = ++(base_t::state().fulfilled_count_);
        ASSERT(count <= total_size);
        if (count == total_size) {
            base_t::set_value(std::move(results()));
        }
    }

    std::tuple<async::promise<Args>...>& promises()
    {
        return base_t::state().promises_;
    }

    const std::tuple<async::promise<Args>...>& promises() const
    {
        return base_t::state().promises_;
    }

    std::tuple<Args...>& results()
    {
        return base_t::state().results_;
    }

    template <int id, typename T, typename F>
    static void for_each_element_recursive(T& t, const F& f)
    {
        if constexpr (id == sizeof...(Args)) {
            return;
        } else {
            f.template operator()<id>(t);
            for_each_element_recursive<id + 1>(t, f);
        }
    }

    template <typename T, typename F>
    static void for_each_element(T& t, const F& f)
    {
        for_each_element_recursive<0>(t, f);
    }
};

} // namespace async::impl
