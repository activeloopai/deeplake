#pragma once

#include <base/exception.hpp>

#include <string>
#include <utility>

namespace icm {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what) : base::exception(std::move(what))
    {
    }
};

class out_of_range : public exception
{
public:
    explicit out_of_range(std::string&& what) : exception(std::move(what))
    {
    }
};
} // namespace icm
