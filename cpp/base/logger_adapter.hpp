#pragma once

#include "logger.hpp"

#include <string>

namespace base {
class logger_adapter
{
public:
    virtual ~logger_adapter() = default;

    [[nodiscard]] virtual const std::string name() const = 0;
    virtual void log(log_level level, const std::string& channel, const std::string& message) = 0;
};
} // namespace base
