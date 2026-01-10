#include "memory_tracker.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>
#include <miscadmin.h>
#include <postmaster/bgworker.h>
#include <storage/proc.h>
#include <utils/elog.h>
#include <utils/guc.h>
#include <utils/memutils.h>

#ifdef __cplusplus
}
#endif

namespace pg {

constexpr int64_t BYTES_PER_KB = 1024;
constexpr int64_t BYTES_PER_MB = BYTES_PER_KB * 1024;

int32_t memory_tracker::memory_limit_mb_{0};

void memory_tracker::initialize_guc_parameters()
{
    //base::memory_info info;
    //base::system_report::get_meminfo(info);
    const auto default_memory_limit = 0;///(info.system_total / BYTES_PER_MB * 0.90); /// consider parallel workers
    DefineCustomIntVariable(
        "pg_deeplake.memory_limit_mb",
        "Memory limit for pg_deeplake operations in MB. Set to 0 to disable limit.",
        "This parameter controls the maximum amount of PostgreSQL memory that pg_deeplake operations can use. "
        "When the limit is exceeded, queries will error out to prevent memory exhaustion.",
        &memory_tracker::memory_limit_mb_,    // linked C variable
        default_memory_limit,                 // default value (90% of system memory)
        0,                                    // min value (0 = unlimited)
        INT_MAX / BYTES_PER_MB,               // max value
        PGC_USERSET,                          // context (USERSET, SUSET, etc.)
        GUC_UNIT_MB,                          // flags - treat as MB
        nullptr,                              // check_hook
        nullptr,                              // assign_hook
        nullptr                               // show_hook
    );
    elog(DEBUG1, "PostgreSQL memory tracker initialized with limit: %d MB", memory_tracker::memory_limit_mb_);
}

void memory_tracker::check_memory_limit()
{
    if (is_memory_limit_exceeded()) {
        const auto current = get_memory_usage_bytes();
        const auto limit = get_memory_limit_bytes();
        elog(ERROR, "PostgreSQL memory limit exceeded: current %ld MB, limit %ld MB",
                     current / BYTES_PER_MB, limit / BYTES_PER_MB);
    }
}

void memory_tracker::ensure_memory_available(int64_t needed_bytes)
{
    const auto limit = get_memory_limit_bytes();
    if (limit <= 0) {
        return;
    }
    const auto current_usage = get_memory_usage_bytes();
    if ((current_usage + needed_bytes) > limit) {
        elog(ERROR, "Insufficient memory: need %zu bytes, current %ld MB, limit %ld MB",
                     needed_bytes, current_usage / BYTES_PER_MB, limit / BYTES_PER_MB);
    }
}

void memory_tracker::log_memory_stats()
{
    const auto current_usage = get_memory_usage_bytes();
    const auto limit = get_memory_limit_bytes();
    elog(INFO, "PostgreSQL memory stats: usage %ld MB, limit %ld MB (%.1f%% used)",
                current_usage / BYTES_PER_MB,
                limit / BYTES_PER_MB,
                limit > 0 ? (current_usage * 100.0 / limit) : 0.0);
}

} // namespace pg
