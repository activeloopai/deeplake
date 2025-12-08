#pragma once

/**
 * @file exceptions.hpp
 * @brief Definitions of the exceptions for `bifrost` module.
 */

#include <base/exception.hpp>

#include <string>
#include <utility>

namespace bifrost {

class exception : public base::exception 
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {}
};

class stop_iteration : exception
{
public:
    stop_iteration()
        : exception(std::string("stop"))
    {}
};

}