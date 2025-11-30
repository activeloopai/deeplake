#pragma once

/**
 * @file exceptions.hpp
 * @brief Definition of the `exception` class.
 */

#include <base/exception.hpp>

namespace async {

/// @brief Base error type for `async` module.
class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class wait_on_same_thread : public exception
{
public:
    wait_on_same_thread()
        : exception("Cannot wait on the same thread")
    {
    }
};

class cannot_cancel_shared_promise : public exception
{
public:
    cannot_cancel_shared_promise()
        : exception("Cannot cancel the shared_promise, because it has listeners. Cancel the listeners first.")
    {
    }
};

class shared_promise_cancelled : public exception
{
public:
    shared_promise_cancelled()
        : exception("The shared_promise has been already cancelled.")
    {
    }
};

} // namespace async
