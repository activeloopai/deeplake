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

#include "table_ddl_lock.hpp"

namespace pg {

// Static pointer to shared memory lock data - must be non-const as it's initialized at runtime
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
table_ddl_lock::lock_data* table_ddl_lock::data_ = nullptr;

Size table_ddl_lock::get_shmem_size()
{
    return MAXALIGN(sizeof(lock_data));
}

void table_ddl_lock::initialize()
{
    bool found = false;

    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

    data_ = static_cast<lock_data*>(ShmemInitStruct(
        "deeplake_table_ddl",
        get_shmem_size(),
        &found
    ));

    if (!found) {
        // First time initialization - assign lock from named tranche
        data_->lock = &(GetNamedLWLockTranche("deeplake_table_ddl")->lock);
    }

    LWLockRelease(AddinShmemInitLock);
}

LWLock* table_ddl_lock::get_lock()
{
    if (data_ == nullptr) {
        return nullptr;
    }
    return data_->lock;
}

// RAII guard implementation
table_ddl_lock_guard::table_ddl_lock_guard()
    : lock_(table_ddl_lock::get_lock())
{
    if (lock_ != nullptr) {
        LWLockAcquire(lock_, LW_EXCLUSIVE);
    }
}

table_ddl_lock_guard::~table_ddl_lock_guard()
{
    if (lock_ != nullptr) {
        LWLockRelease(lock_);
    }
}

} // namespace pg
