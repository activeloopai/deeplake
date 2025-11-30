#pragma once

#include <type_traits>

namespace async {

template <typename T>
class promise;

namespace impl {

template <typename T>
constexpr bool is_promise_v = false;

template <typename T>
constexpr bool is_promise_v<async::promise<T>> = true;

template <typename F, typename T, bool = std::is_void_v<T>>
struct invoke_result_maybe_void
{
    using type = std::invoke_result_t<F, T>;
};

template <typename F, typename T>
struct invoke_result_maybe_void<F, T, true>
{
    using type = std::invoke_result_t<F>;
};

template <typename F, typename T>
using invoke_result_maybe_void_t = typename invoke_result_maybe_void<F, T>::type;


template <typename C>
class has_
{
    template <typename C_> static int f_progress(decltype(static_cast<float(C_::*)() const>(&C_::progress)));
    template <typename C_> static int f_set_priority(decltype(static_cast<void(C_::*)(int)>(&C_::set_priority)));
    
    template <typename C_> static char f_progress(...);
    template <typename C_> static char f_set_priority(...);

public:
    static constexpr bool progress = sizeof(decltype(f_progress<C>(nullptr))) == sizeof(int);
    static constexpr bool set_priority = sizeof(decltype(f_set_priority<C>(nullptr))) == sizeof(int);
};

}

}
