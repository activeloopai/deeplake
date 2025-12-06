#pragma once

#include <base/logger.hpp>
#include <base/logger_adapter.hpp>

#include <mutex>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/elog.h>

#ifdef __cplusplus
} /// extern "C"
#endif

namespace pg {

class logger_adapter : public base::logger_adapter
{
public:
    logger_adapter() = default;
    ~logger_adapter() override = default;

    [[nodiscard]] const std::string name() const override
    {
        return "pg_deeplake";
    }

    void log(base::log_level level, const std::string& channel, const std::string& message) override
    {
        std::lock_guard lock(mutex_);
        switch (level) {
        case base::log_level::debug:
            elog(DEBUG1, "%s", message.c_str());
            break;
        case base::log_level::info:
            elog(INFO, "%s", message.c_str());
            break;
        case base::log_level::warning:
            elog(WARNING, "%s", message.c_str());
            break;
        case base::log_level::error:
            elog(NOTICE, "Error from DeepLake: %s", message.c_str());
            break;
        }
    }

private:
    std::mutex mutex_;
};

} // namespace pg
