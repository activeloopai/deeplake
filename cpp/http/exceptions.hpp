#pragma once

/**
 * @file exceptions.hpp
 * @brief Definitions of the exceptions for `hub` module.
 */

#include <base/concat.hpp>
#include <base/exception.hpp>

namespace http {

class exception : public base::exception
{
public:
    exception(std::string&& what, params_t&& params)
        : base::exception(std::move(what), std::move(params))
    {
    }

    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class request_failed : public exception
{
public:
    request_failed(std::string method, std::string resource, int error_code, std::string message)
        : exception(base::concat(" ", "Method failed:", method, "resource:", resource, "message:", message),
                    {{"resource", resource}, {"errorCode", std::to_string(error_code)}, {"message", message}})
    {
    }
};

class unknown_error : public exception
{
public:
    explicit unknown_error()
        : exception(std::string("Request failed with unknown error."))
    {
    }
};

class body_is_missing : public exception
{
public:
    explicit body_is_missing()
        : exception(std::string("The response has no body."))
    {
    }
};

class body_is_not_json : public exception
{
public:
    explicit body_is_not_json()
        : exception(std::string("The body of the response is not valid json."))
    {
    }
};

class invalid_uri : public exception
{
public:
    invalid_uri(std::string uri)
        : exception("Invalid URI format. Expected: scheme://container/object_path. Got: " + uri)
    {
    }
};

} // namespace http
