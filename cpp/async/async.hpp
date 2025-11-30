#pragma once

#include "queue.hpp"

#include <base/function.hpp>

#include <vector>

/**
 * @defgroup async
 * @{
 * @brief Asynchronous tasks/requests.
 * 
 * async module defines asynchronous task functionality. The main class is @a promise, this class
 * represents promise similar to `std::promise`, but with the additional functionality:
 * - Set callback to the promise - `promise::set_callback`
 * - Change the priority of the async task - `promise::set_priority`
 * - Cancel the task - `promise::cancel`
 * - Chain promises - `promise::then`
 * - Check the progress of the async task - `promise::progress`
 * @}
 */

namespace async {

template <typename T>
class promise;

/**
 * @brief Provides the input needed for http client to work.
 *
 * @param bg_queue The queue which is used to submit heavy tasks.
 */
void initialize(bg_queue& q);

/**
 * @brief Deinitializes the module.
 */
void deinitialize();

/**
 * @brief Access to the predefined main queue. This queue is not the main thread of the application.
 * Main queue is single threaded queue, where async task callbacks are always being submitted. Main queue is used to
 * synchronize async tasks through it, as an alternative to lock and mutexes.
 * Compute heavy tasks should not be submitted to the main queue.
 */
main_queue& main() noexcept;

/**
 * @brief Access to the predefined background queue.
 * Background queue is multithreaded queue, where compute heavy tasks are submitted. The number of threads for the queue
 * varies, but usually it's number of CPU cores. Client code can modify number of threads.
 */
bg_queue& bg() noexcept;

/// Utility to check if the current thread is main or not.
bool is_this_thread_main() noexcept;

/// Submits or runs the function in the main thread.
void submit_in_main(base::function<void()> f);

/// Adds the promise to the list of detached promises.
void add_detached_promise(promise<void>&& p);

/// Takes the list of detached promises.
std::vector<promise<void>> take_detached_promises() noexcept;

}
