#pragma once

#include "agreement_handler.hpp"

#include <async/queue.hpp>
#include <base/system_report.hpp>

#include <cstdint>
#include <thread>

/**
 * @file storage.hpp
 * @brief Storage module level functions definitions.
 */

/**
 * @defgroup storage
 * @{
 * @brief storage module.
 * 
 * key-value storage abstraction and common implementations.
 *
 * @}
 */

namespace storage {

class storage
{
public:
    static void initialize(int32_t num_threads = 3 * base::system_report::cpu_cores());

    static void deinitialize();

    static storage& instance()
    {
        static storage instance_;
        return instance_;
    }

    async::queue& queue()
    {
        return *queue_;
    }

    auto agreement() const
    {
        return agreement_handler_;
    }

    inline int32_t concurrency() const noexcept
    {
        if (queue_ == nullptr) {
            return 0;
        }
        return queue_->num_threads();
    }

    inline void concurrency(int32_t num_threads) noexcept
    {
        if (queue_ != nullptr) {
            queue_->change_num_threads(num_threads);
        }
    }

private:
    async::bg_queue* queue_;
    agreement_handler agreement_handler_ = agreement_handler();
};

/**
 * @brief Check if io_uring is available at runtime
 * @return true if io_uring can be initialized, false otherwise
 */
bool is_io_uring_available();

}
