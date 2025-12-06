#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>
#include <storage/ipc.h>
#include <storage/lwlock.h>
#include <storage/shmem.h>

#ifdef __cplusplus
}
#endif

#include "table_version.hpp"

namespace {

struct locker
{
    LWLock* lwlock;

    locker(LWLock* lock, LWLockMode m)
        : lwlock(lock)
    {
        LWLockAcquire(lwlock, m);
    }

    ~locker()
    {
        LWLockRelease(lwlock);
    }
};

} // unnamed namespace

namespace pg {

extern bool use_shared_mem_for_refresh;

table_version_data* table_version_tracker::version_data_ = nullptr;

Size table_version_tracker::get_shmem_size()
{
    Size size = MAXALIGN(sizeof(table_version_data));
    size = add_size(size, mul_size(MAX_TRACKED_TABLES, sizeof(table_version_entry)));
    return size;
}

void table_version_tracker::initialize()
{
    bool found = false;

    locker lock(AddinShmemInitLock, LW_EXCLUSIVE);

    version_data_ = (table_version_data*)ShmemInitStruct(
        "deeplake_table_versions",
        get_shmem_size(),
        &found
    );

    if (!found) {
        // First time initialization
        version_data_->lock = &(GetNamedLWLockTranche("deeplake_versions")->lock);
        version_data_->max_tables = MAX_TRACKED_TABLES;
        version_data_->num_tables = 0;

        // Initialize all entries
        for (int32_t i = 0; i < MAX_TRACKED_TABLES; ++i) {
            version_data_->entries[i].table_oid = InvalidOid;
            version_data_->entries[i].version = 0;
        }
    }
}

table_version_entry* table_version_tracker::find_entry(Oid table_oid, bool create)
{
    if (version_data_ == nullptr || !pg::use_shared_mem_for_refresh) {
        return nullptr;
    }

    // Linear search for the table
    for (int32_t i = 0; i < version_data_->num_tables; ++i) {
        if (version_data_->entries[i].table_oid == table_oid) {
            return &version_data_->entries[i];
        }
    }

    if (create && version_data_->num_tables < version_data_->max_tables) {
        // Create new entry
        int32_t idx = version_data_->num_tables++;
        version_data_->entries[idx].table_oid = table_oid;
        version_data_->entries[idx].version = 0;
        return &version_data_->entries[idx];
    }

    return nullptr;
}

void table_version_tracker::increment_version(Oid table_oid)
{
    if (version_data_ == nullptr || !pg::use_shared_mem_for_refresh) {
        return;
    }

    locker lock(version_data_->lock, LW_EXCLUSIVE);

    table_version_entry* entry = find_entry(table_oid, true);
    if (entry != nullptr) {
        ++entry->version;
    }
}

uint64_t table_version_tracker::get_version(Oid table_oid)
{
    if (version_data_ == nullptr || !pg::use_shared_mem_for_refresh) {
        return 0;
    }

    locker lock(version_data_->lock, LW_SHARED);

    uint64_t version = 0;
    table_version_entry* entry = find_entry(table_oid, false);
    if (entry != nullptr) {
        version = entry->version;
    }

    return version;
}

void table_version_tracker::drop_table(Oid table_oid)
{
    if (version_data_ == nullptr || !pg::use_shared_mem_for_refresh) {
        return;
    }

    locker lock(version_data_->lock, LW_EXCLUSIVE);

    for (int32_t i = 0; i < version_data_->num_tables; ++i) {
        if (version_data_->entries[i].table_oid == table_oid) {
            // Move last entry to this position
            version_data_->entries[i] = version_data_->entries[version_data_->num_tables - 1];
            version_data_->entries[version_data_->num_tables - 1].table_oid = InvalidOid;
            version_data_->entries[version_data_->num_tables - 1].version = 0;
            --version_data_->num_tables;
            break;
        }
    }
}

void table_version_tracker::clear_all_versions()
{
    if (version_data_ == nullptr || !pg::use_shared_mem_for_refresh) {
        return;
    }

    locker lock(version_data_->lock, LW_EXCLUSIVE);

    version_data_->num_tables = 0;
    for (int32_t i = 0; i < version_data_->max_tables; ++i) {
        version_data_->entries[i].table_oid = InvalidOid;
        version_data_->entries[i].version = 0;
    }
}

} // namespace pg
