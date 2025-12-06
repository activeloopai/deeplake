#pragma once

/**
 * @file exception.hpp
 * @brief Definition of the `exception` class.
 */

#include "format.hpp"

#include <exception>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace base {

/**
 * @brief Base class for all indra exceptions.
 */
class exception : public std::exception
{
public:
    using params_t = std::map<std::string, std::string, std::less<>>;

public:
    explicit exception(std::string&& what)
        : what_(std::move(what))
    {
    }

    exception(std::string&& what, params_t&& params)
        : what_(std::move(what))
        , params_(std::move(params))
    {
    }

    const char* what() const noexcept override
    {
        return what_.c_str();
    }

    const std::string& message() const noexcept
    {
        return what_;
    }

    const auto& params() const noexcept
    {
        return params_;
    }

private:
    std::string what_;
    params_t params_;
};

class mmap_failed : public exception
{
public:
    explicit mmap_failed(const std::string& file)
        : exception(std::string("Can't open file '") + file + "' for mmap.")
    {
    }
};

class wrong_sequence_htype : public exception
{
public:
    explicit wrong_sequence_htype(const std::string& ht)
        : exception(fmt::format("Wrong sequence htype {}. Maybe you forgot the last ']'", ht))
    {
    }
};

} // namespace base
