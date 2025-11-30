#pragma once

#include <cstdint>

namespace base {

/**
 * @struct memory_info
 * @brief Represents system and process memory information.
 */
struct memory_info
{
    uint64_t system_total = 0;
    uint64_t system_available = 0;

    // The total amount of virtual memory allocated to the process, including all
    // code, data, and shared libraries, plus pages that are swapped out.
    uint64_t process_vm_size = 0;

    // The portion of a process's memory held in RAM.
    uint64_t process_vm_rss = 0;

    // The peak virtual memory size
    uint64_t process_peak_mem = 0;

    // The maximum virtual memory set size.
    uint64_t process_max_vm_size = 0;

    // The maximum resident set size used.
    uint64_t process_max_vm_rss = 0;
};

} // namespace base
