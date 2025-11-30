#pragma once

#include <base/exception.hpp>
#include <base/format.hpp>
#include <nd/dtype.hpp>

namespace query_core {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class invalid_order_function : public exception
{
public:
    invalid_order_function()
        : exception(std::string("invalid_order_function"))
    {
    }
};

class expr_error : public exception
{
public:
    explicit expr_error(std::string&& what)
        : exception(std::move(what))
    {
    }
};

} // namespace query_core
