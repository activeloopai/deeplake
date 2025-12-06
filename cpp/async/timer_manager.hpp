#pragma once

#include "priority.hpp"

#include <base/function.hpp>

#include <chrono>
#include <cstdint>
#include <memory>

namespace async {

class timer_manager
{
    /// @name Type definitions
    /// @{
public:
    /// Unique identifier for each timer, 0 is invalid timer id
    using timer_id_t = uint64_t;
    using duration_t = std::chrono::milliseconds;
    using callback_t = base::function<void()>;
    using clock_t = std::chrono::steady_clock;
    /// @}

    /// Returns singleton instance of timer_manager
    static timer_manager& instance();

    /// Schedule one-shot timer that fires once after specified duration
    timer_id_t schedule_once(duration_t interval, callback_t callback, int priority = async::default_priority);

    /// Schedule periodic timer that fires repeatedly with specified interval
    timer_id_t schedule_periodic(duration_t interval, callback_t callback, int priority = async::default_priority);

    /// Schedule periodic timer that fires repeatedly with specified interval and duration
    timer_id_t schedule_periodic(duration_t interval,
                                 duration_t duration,
                                 callback_t callback,
                                 int priority = async::default_priority);

    /// Cancel previously scheduled timer
    void cancel(timer_id_t id) noexcept;

    /// Pause all timers. Their remaining time is preserved
    void pause() noexcept;

    /// Resume all previously paused timers
    void resume() noexcept;

    /// In a single-threaded environment, we cannot use background threads,
    /// so we need to call this method periodically to process timers.
    void tick();

private:
    timer_manager();
    ~timer_manager() noexcept;

    timer_manager(const timer_manager&) = delete;
    timer_manager& operator=(const timer_manager&) = delete;

private:
    class impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace async
