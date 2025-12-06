#pragma once

/**
 * @file spin_lock.hpp
 * @brief Definition and implementation of the `spli_lock` class.
 */

#include <atomic>

namespace base {

class spin_lock
{
public:
    spin_lock() noexcept = default;

    spin_lock(const spin_lock& other) = delete;
    spin_lock& operator=(const spin_lock& other) = delete;

    spin_lock(spin_lock&& other) = delete;
    spin_lock& operator=(spin_lock&& other) = delete;

public:
    inline bool try_lock() noexcept
    {
        return !lock_.test_and_set(std::memory_order_acquire);
    }

    inline void lock() noexcept
    {
        while (lock_.test_and_set(std::memory_order_acquire));
    }

    inline void unlock() noexcept
    {
        lock_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag lock_;
};

}
