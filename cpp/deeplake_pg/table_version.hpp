#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>
#include <storage/lwlock.h>
#include <storage/shmem.h>

#ifdef __cplusplus
}
#endif

#include <cstdint>

namespace pg {

// Per-table version tracking entry
struct table_version_entry
{
    Oid table_oid;
    uint64_t version;
};

// Shared memory structure for table version tracking
struct table_version_data
{
    LWLock* lock;
    int32_t max_tables;
    int32_t num_tables;
    table_version_entry entries[FLEXIBLE_ARRAY_MEMBER];
};

class table_version_tracker
{
public:
    // Initialize shared memory for version tracking
    static void initialize();

    // Get required shared memory size
    static Size get_shmem_size();

    // Increment version for a table (called on writes)
    static void increment_version(Oid table_oid);

    // Get current version for a table (called on reads)
    static uint64_t get_version(Oid table_oid);

    // Drop version tracking for a table
    static void drop_table(Oid table_oid);

    // Clear all version tracking entries
    static void clear_all_versions();
private:
    static constexpr int32_t MAX_TRACKED_TABLES = 1024;
    static table_version_data* version_data_;

    // Find or create entry for table
    static table_version_entry* find_entry(Oid table_oid, bool create);
};

} // namespace pg
