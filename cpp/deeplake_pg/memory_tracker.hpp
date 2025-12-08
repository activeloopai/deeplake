#pragma once

#include "base/memory_info.hpp"
#include "base/system_report.hpp"

#include <cstdint>

namespace pg {

class memory_tracker
{
public:
    static void check_memory_limit();

    static int64_t get_memory_usage_bytes() noexcept
    {
        base::memory_info info;
        base::system_report::get_meminfo(info);
        return info.process_vm_rss;
    }

    static int64_t get_memory_limit_bytes() noexcept
    {
        return memory_limit_mb_ * 1024L * 1024L;
    }

    static bool has_memory_limit() noexcept
    {
        return memory_limit_mb_ > 0;
    }

    static bool is_memory_limit_exceeded() noexcept
    {
        const auto limit = get_memory_limit_bytes();
        if (limit <= 0) {
            return false;
        }
        return get_memory_usage_bytes() > limit;
    }

    static void ensure_memory_available(int64_t needed_bytes);

    static void log_memory_stats();

    static void initialize_guc_parameters();

private:
    static int32_t memory_limit_mb_;
};

} // namespace pg