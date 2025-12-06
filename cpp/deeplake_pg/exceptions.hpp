#pragma once

#include <base/exception.hpp>

namespace pg {

class exception : public base::exception
{
public:
    explicit exception(std::string&& message)
        : base::exception(std::move(message))
    {
    }
};

} /// pg namespace
