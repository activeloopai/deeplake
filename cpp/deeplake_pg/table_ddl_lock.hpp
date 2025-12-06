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

namespace pg {

/**
 * Simple global lock for table DDL operations (CREATE/DROP TABLE).
 *
 * This prevents race conditions when multiple backends execute
 * CREATE TABLE IF NOT EXISTS or DROP TABLE concurrently.
 */
class table_ddl_lock
{
public:
    // Initialize shared memory for the DDL lock
    static void initialize();

    // Get required shared memory size
    static Size get_shmem_size();

    // Get the global DDL lock (for RAII wrapper)
    static LWLock* get_lock();

private:
    struct lock_data
    {
        LWLock* lock;
    };

    static lock_data* data_;
};

/**
 * RAII wrapper for table DDL lock.
 * Automatically acquires lock in constructor and releases in destructor.
 */
class table_ddl_lock_guard
{
public:
    table_ddl_lock_guard();
    ~table_ddl_lock_guard();

    // Disable copy/move
    table_ddl_lock_guard(const table_ddl_lock_guard&) = delete;
    table_ddl_lock_guard& operator=(const table_ddl_lock_guard&) = delete;

private:
    LWLock* lock_;
};

} // namespace pg
