#pragma once

/**
 * @file exceptions.hpp
 * @brief Definitions of the exceptions for `hub` module.
 */

#include <base/concat.hpp>
#include <base/exception.hpp>

namespace storage {

template <typename T, typename U>
concept not_self_constructable = !std::is_same_v<std::decay_t<T>, U>;

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

class invalid_operation : public exception
{
public:
    invalid_operation(std::string&& what)
        : exception(std::move(what), {})
    {
    }
};

class reader_error : public exception
{
public:
    reader_error(std::string resource, int error_code, std::string message)
        : exception(base::concat(" ", "Storage reader error:", "resource:", resource, "message:", message),
                    {{"resource", resource}, {"errorCode", std::to_string(error_code)}, {"message", message}})
    {
    }
};

class writer_error : public exception
{
public:
    writer_error(std::string resource, int error_code, std::string message)
        : exception(
              base::concat(" ", "Storage reader error:", "resource:", resource, "message:", message),
              {{"resource", resource}, {"errorCode", std::to_string(error_code)}, {"message", std::move(message)}})
    {
    }
};

class expired_token_exception : public exception
{
public:
    explicit expired_token_exception()
        : exception("The provided token has expired.")
    {
    }
};

class write_exception : public exception
{
public:
    explicit write_exception(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class curl_initialization_failed : public exception
{
public:
    explicit curl_initialization_failed(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class gcs_storage_provider_failed : public exception
{
public:
    explicit gcs_storage_provider_failed(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class invalid_azure_credentials_chain : public exception
{
public:
    explicit invalid_azure_credentials_chain()
        : exception("Azure credentials chain is invalid")
    {
    }
};

class unknown_memory_policy : public exception
{
public:
    explicit unknown_memory_policy()
        : exception("Unknown memory policy")
    {
    }
};

class read_only_error : public exception
{
public:
    explicit read_only_error()
        : exception("The given storage provider is read only.")
    {
    }
};

class storage_access_denied : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_access_denied>)
    explicit storage_access_denied(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_key_not_found : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_key_not_found>)
    explicit storage_key_not_found(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_sign_url_failed : public exception
{
public:
    explicit storage_sign_url_failed(std::string&& what) : exception(std::move(what))
    {
    }
};

class storage_provider_error : public exception
{
public:
    storage_provider_error(std::string resource, int error_code, std::string message)
        : exception(base::concat(" ",
                                 "Storage provider error:",
                                 "resource=",
                                 resource,
                                 "code=",
                                 std::to_string(error_code),
                                 "message=",
                                 message),
                    {{"resource", std::move(resource)},
                     {"errorCode", std::to_string(error_code)},
                     {"message", std::move(message)}})
    {
    }
};

class storage_writer_failed : public exception
{
public:
    storage_writer_failed(const std::string& path, int code)
        : exception(fmt::format("Failed to write to {} with code: {}", path, code))
    {
    }
};

class storage_key_already_exists : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_key_already_exists>)
    explicit storage_key_already_exists(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_network_connection : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_network_connection>)
    explicit storage_network_connection(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_already_destoryed : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_already_destoryed>)
    explicit storage_already_destoryed(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_other_error : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_other_error>)
    explicit storage_other_error(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

class storage_exeeded_block_limit : public exception
{
public:
    template <typename... Args>
    requires(... && not_self_constructable<Args, storage_exeeded_block_limit>)
    explicit storage_exeeded_block_limit(Args&&... args)
        : exception(base::concat(" ", std::forward<Args>(args)...))
    {
    }
};

} // namespace storage
