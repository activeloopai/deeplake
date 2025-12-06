#pragma once

#include "context.hpp"
#include "promise.hpp"
#include "impl/bg_queue_promise.hpp"
#include "impl/mpl.hpp"

namespace async {

namespace impl {

template <typename F, bool>
struct result_type;

template <typename F>
struct result_type<F, true>
{
    using type = std::remove_cvref_t<std::invoke_result_t<F>>;
};

template <typename F>
struct result_type<F, false>
{
    using type = std::remove_cvref_t<std::invoke_result_t<F, async::context&>>;
};

template <typename F>
using result_type_t = typename result_type<F, std::is_invocable_v<F>>::type;

template <typename Result, typename Cancel>
inline auto create_bg_promise_handle(Cancel&& c)
{
    if constexpr (std::is_same_v<Cancel, std::nullptr_t>) {
        return bg_queue_promise<Result>();
    } else {
        using handle_t = bg_queue_promise<Result>;
        using c_t = typename handle_t::cancel_callback_t;
        return handle_t(c_t(std::forward<Cancel>(c)));
    }
}

template <typename Functor, typename Cancel, typename SubmitFunc>
inline auto run_(Functor&& f, Cancel&& c, SubmitFunc&& submit_func)
{
    static_assert(std::is_invocable_v<Functor> || std::is_invocable_v<Functor, async::context&>,
                  "The functor should be invocable without arguments or with async::context&.");
    using Result = result_type_t<Functor>;
    //auto&& promise_impl = create_bg_promise_handle<Result>(std::forward<Cancel>(c));
    bg_queue_promise<Result> promise_impl(std::forward<Cancel>(c));
    std::forward<SubmitFunc>(submit_func)(
        [promise_impl, f = std::forward<Functor>(f)]() mutable {
            promise_impl.call(std::move(f));
        },
        &promise_impl.id());

    promise<Result> result_promise(std::move(promise_impl));
    if constexpr (impl::is_promise_v<Result>) {
        return std::move(result_promise).then_any([](auto p) {
            return p;
        });
    } else {
        return result_promise;
    }
}

}

/**
 * @brief Chains the given promise with the given functor, which is being asynchronously applied
 * to the result of input promise. The output of the functor will be the result of the
 * returned promise.
 */
template <typename Functor, typename Cancel = std::nullptr_t>
inline auto run(Functor&& f, queue& queue, int priority = default_priority, Cancel&& cancel = Cancel())
{
    return impl::run_(
        std::forward<Functor>(f), std::forward<Cancel>(cancel), [&queue, priority](auto&& c, queue::id_type* id) {
            queue.submit(std::forward<decltype(c)>(c), priority, id);
        });
}

template <typename Functor, typename Cancel = std::nullptr_t>
inline auto run_on_main(Functor&& f, Cancel&& cancel = Cancel())
{
    return impl::run_(std::forward<Functor>(f), std::forward<Cancel>(cancel), [](auto&& c, queue::id_type* id) {
        main().run_or_submit(std::forward<decltype(c)>(c), default_priority, id);
    });
}

template <typename Functor, typename Cancel = std::nullptr_t>
inline auto run_on_bg(Functor&& f, int priority = default_priority, Cancel&& cancel = Cancel())
{
    return run(std::forward<Functor>(f), bg(), priority, std::forward<Cancel>(cancel));
}

}
