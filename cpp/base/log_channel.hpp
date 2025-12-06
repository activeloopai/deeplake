#pragma once

#include <string>

namespace base {

class log_channel
{
public:
    [[nodiscard]] std::string channel() const
    {
        return channel_;
    }

    static const log_channel async;
    static const log_channel client;
    static const log_channel generic;
    static const log_channel index;
    static const log_channel deepmemory;
    static const log_channel storage_azure;
    static const log_channel storage_generic;
    static const log_channel storage_google;
    static const log_channel storage_local;
    static const log_channel storage_memory;
    static const log_channel storage_s3;
    static const log_channel tql;
    static const log_channel visualizer;

private:
    explicit log_channel(std::string channel)
        : channel_(std::move(channel))
    {
    }

    std::string channel_;
};
} // namespace log_channel