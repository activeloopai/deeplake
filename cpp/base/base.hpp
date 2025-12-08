#pragma once

#include "log_channel.hpp"
#include "logger_adapter.hpp"
#include "logging_span_holder.hpp"

/**
 * @file base.hpp
 * @brief Definition of base module level functions.
 */

/**
 * @defgroup base
 * @{
 * @brief Defines lowest level functionality and utilities for deeplake.
 *
 * @}
 */

namespace base {

/**
 * @brief initialize the base module.
 *
 * @param adapter
 */
void initialize(std::shared_ptr<logger_adapter> adapter);

/**
 * @brief deinitialize the base module.
 *
 */
void deinitialize();

void cleanup();

bool is_initialized();

logger& get_logger();

/**
* Logs a start/end of a process. Message is initially logged when the method is called.
* A finish version of the message along with the total time is logged when the returned holder is destructed.
*
* Generally you can use SPAN_NAME() for the "name" argument which computes a name from the surrounding function,
* but for more complex code locations you may need to manually specify it.
* SPAN_NAME() will fail in dev builds if it does not generate a valid name.
*
* NOTE: IF USED IN A METHOD WITH ASYNC LAMBDAS, STD::MOVE THE SPAN INTO THE .WITH() CALL TO INCLUDE THE LAMBDA IT IN THE SPAN
*/
std::unique_ptr<logging_span_holder> log_span(const log_channel& channel, const std::string& message);

/**
 * @param channel Log channel to use
 * @param message Message to log.
 * @param args Arguments to format message with.
 */
template <typename... Args>
inline void log_debug(const log_channel& channel, const std::string& message, Args&&... args)
{
    get_logger().log(log_level::debug, channel.channel(), message, fmt::make_format_args(args...));
}

/**
 * @param channel Log channel to use
 * @param message Message to log.
 * @param args Arguments to format message with.
 */
template <typename... Args>
inline void log_info(const log_channel& channel, const std::string& message, Args&&... args)
{
    get_logger().log(log_level::info, channel.channel(), message, fmt::make_format_args(args...));
}

/**
 * @param channel Log channel to use
 * @param message Message to log.
 * @param args Arguments to format message with.
 */
template <typename... Args>
inline void log_warning(const log_channel& channel, const std::string& message, Args&&... args)
{
    get_logger().log(log_level::warning, channel.channel(), message, fmt::make_format_args(args...));
}

/**
 * @param channel Log channel to use
 * @param message Message to log.
 * @param args Arguments to format message with.
 */
template <typename... Args>
inline void log_error(const log_channel& channel, const std::string& message, Args&&... args)
{
    get_logger().log(log_level::error, channel.channel(), message, fmt::make_format_args(args...));
}
} // namespace base
