#pragma once

#include "log_channel.hpp"

#include <chrono>
#include <memory>
#include <string>

namespace base {

/**
* Holds a span that logs its lifetime to "debug" level.
*/
class logging_span_holder
{
public:
    logging_span_holder(const log_channel& log_channel, std::string log_message);

    logging_span_holder(const logging_span_holder& other) = delete;
    logging_span_holder& operator=(const logging_span_holder& other) = delete;
    logging_span_holder& operator=(logging_span_holder&& other) = delete;

    logging_span_holder(logging_span_holder&& other) noexcept
        : log_channel_(other.log_channel_)
        , log_message_(std::move(other.log_message_))
        , start_time_(other.start_time_)
        , ended_(other.ended_)
    {
        other.ended_ = true;
    }

    virtual ~logging_span_holder()
    {
        logging_span_holder::end();
    }

    virtual void end();

private:
    const log_channel log_channel_;
    const std::string log_message_;
    const std::chrono::high_resolution_clock::time_point start_time_;
    bool ended_ = false;
};
} // namespace base
