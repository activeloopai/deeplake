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

// GUC variables for sync worker configuration
extern int deeplake_sync_interval_ms;

namespace pg {

class pending_install_queue {
public:
    static Size get_shmem_size();
    static void initialize();
    static bool enqueue(const char* dbname);
    static void drain_and_install();

private:
    static constexpr int32_t MAX_PENDING = 64;

    struct entry {
        char db_name[NAMEDATALEN];
    };

    struct queue_data {
        LWLock* lock;
        int32_t count;
        entry entries[FLEXIBLE_ARRAY_MEMBER];
    };

    static queue_data* data_;
};

} // namespace pg
