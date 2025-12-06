#pragma once

/**
 * @file logger.hpp
 * @brief Definition and implementation of the `logger` class.
 */

#include "assert.hpp"
#include "format.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace base {
class logger_adapter;

enum class log_level : unsigned char
{
    debug,
    info,
    warning,
    error
};

inline std::string log_level_to_str(const log_level t)
{
    switch (t) {
    case log_level::debug:
        return "debug";
    case log_level::info:
        return "info";
    case log_level::warning:
        return "warning";
    case log_level::error:
        return "error";
    default:
        ASSERT_MESSAGE(false, "Unknown log level");
        return "unknown";
    }
}

inline log_level str_to_log_level(const std::string& level)
{
    auto final_string = level;
    std::ranges::transform(final_string, final_string.begin(), toupper);
    if (final_string == "DEBUG") {
        return log_level::debug;
    }
    if (final_string == "INFO") {
        return log_level::info;
    }
    if (final_string == "WARN" || final_string == "WARNING") {
        return log_level::warning;
    }
    if (final_string == "ERROR") {
        return log_level::error;
    }

    ASSERT_MESSAGE(false, "Unknown level: " + level);
    return log_level::error;
}

class logger
{
public:
    logger() = default;

    logger(const logger&) = delete;
    logger& operator=(const logger&) = delete;
    logger& operator=(logger&&) = delete;
    ~logger() = default;

    void log(log_level level, const std::string& channel, const std::string& message) const;

    void log(log_level level, const std::string& channel, const std::string& message,
             const std::map<std::string, std::string, std::less<>>& params) const;

    void log(log_level level, const std::string& channel, const std::string& message, const fmt::format_args& args) const;

    void add(std::shared_ptr<logger_adapter> adapter)
    {
        adapters_.push_back(std::move(adapter));
    }

    void remove(const std::string& id);

    void clear()
    {
        adapters_.clear();
    }

private:
    std::vector<std::shared_ptr<logger_adapter>> adapters_;
};

} // namespace base
