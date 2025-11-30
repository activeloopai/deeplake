#pragma once

/**
 * @file function.hpp
 * @brief Definition of the `function` class
 */ 
#include "source_location_holder.hpp"

#include <absl/functional/any_invocable.h>

namespace base {

#ifdef AL_ASSERTIONS

template <typename F>
class function_with_location;

template <typename R, typename... Args>
class function_with_location<R(Args...)> : public absl::AnyInvocable<R(Args...)>, public source_location_holder
{
public:
    function_with_location(source_location location = source_location::current())
        : source_location_holder(std::move(location))
    {
    }

    template <typename Callable>
    function_with_location(Callable&& callable, source_location location = source_location::current())
        : absl::AnyInvocable<R(Args...)>(std::forward<Callable>(callable))
        , source_location_holder(std::move(location))
    {
    }

    template <typename Callable>
    function_with_location& operator=(Callable&& callable)
    {
        absl::AnyInvocable<R(Args...)>::operator=(std::forward<Callable>(callable));
        return *this;
    }

    function_with_location& operator=(std::nullptr_t)
    {
        absl::AnyInvocable<R(Args...)>::operator=(nullptr);
        return *this;
    }

    using absl::AnyInvocable<R(Args...)>::operator();

    explicit operator bool() const
    {
        return absl::AnyInvocable<R(Args...)>::operator bool();
    }
};

template <typename R, typename... Args>
class function_with_location<R(Args...) const> : public absl::AnyInvocable<R(Args...) const>, public source_location_holder
{
public:
    function_with_location(source_location location = source_location::current())
        : source_location_holder(std::move(location))
    {
    }

    template <typename Callable>
    function_with_location(Callable&& callable, source_location location = source_location::current())
        : absl::AnyInvocable<R(Args...) const>(std::forward<Callable>(callable))
        , source_location_holder(std::move(location))
    {
    }

    template <typename Callable>
    function_with_location& operator=(Callable&& callable)
    {
        absl::AnyInvocable<R(Args...) const>::operator=(std::forward<Callable>(callable));
        return *this;
    }

    function_with_location& operator=(std::nullptr_t)
    {
        absl::AnyInvocable<R(Args...) const>::operator=(nullptr);
        return *this;
    }

    using absl::AnyInvocable<R(Args...) const>::operator();

    explicit operator bool() const
    {
        return absl::AnyInvocable<R(Args...) const>::operator bool();
    }
};

template <typename F>
using function = function_with_location<F>;

#else

template <typename F>
using function = absl::AnyInvocable<F>;

#endif

}
