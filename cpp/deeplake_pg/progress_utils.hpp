#pragma once

#include <async/promise.hpp>

#include <string_view>

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/elog.h>

#ifdef __cplusplus
} /// extern "C"
#endif

namespace pg::utils {

/**
 * @brief Introduced custom progress bar for the console
 * @details The progress bar is displayed in the console and updated in real-time.
 * @note The progress bar is introduced to clean previous progress line and update the current progress,
 *       as the progress elog/ereport message printers are always printing the message in a new line.
 *       Inspired by the progress_display class from the Boost's Timer library.
 */
class progress_display
{
public:
    progress_display() = default;
    explicit progress_display(uint64_t expected_count, std::string pre_msg = "")
    {
        restart(expected_count, std::move(pre_msg));
    }

    void restart(uint64_t expected_count, std::string pre_msg = "")
    {
        count_ = 0;
        next_tic_count_ = 0;
        tic_ = 0;
        expected_count_ = expected_count;
        if (expected_count_ == 0) {
            expected_count_ = 1;
        }
        if (!pre_msg.empty()) {
            elog(INFO, "\n%s", pre_msg.c_str());
        }
        msg_ = default_msg_;
        const std::string msg = "\n0%   10   20   30   40   50   60   70   80   90   100%\n"
                                "|----|----|----|----|----|----|----|----|----|----|\n";
        elog(INFO, "%s", msg.c_str());
    }

    inline uint64_t operator+=(uint64_t increment) noexcept
    {
        if ((count_ += increment) >= next_tic_count_) {
            display_tic();
        }
        return count_;
    }

    inline uint64_t operator++() noexcept
    {
        return operator+=(1);
    }

    inline uint64_t count() const noexcept
    {
        return count_;
    }

private:
    void display_tic()
    {
        const auto tics_needed = static_cast<uint32_t>((static_cast<double>(count_) /
                                 static_cast<double>(expected_count_)) * 50.0);
        do {
            msg_ += '*';
        } while (++tic_ < tics_needed);
        next_tic_count_ = static_cast<uint64_t>((tic_ / 50.0) * static_cast<double>(expected_count_));
        if (count_ == expected_count_ && tic_ < 51) {
            msg_ += '*';
        }
        elog(INFO, "%s", msg_.c_str());
    }

    constexpr static std::string_view default_msg_ = "\r\e[1A\e[K";
    std::string msg_;
    uint64_t count_ = 0;
    uint64_t expected_count_ = 0;
    uint64_t next_tic_count_ = 0;
    uint32_t tic_ = 0;
};

inline void print_progress_and_wait(async::promise<void> promise, std::string_view message)
{
    elog(INFO, "%s", message.data());
    progress_display progress_bar{100};
    auto prev_progress = 0u;
    while (!promise.is_ready()) {
        const auto progress = static_cast<uint64_t>(promise.progress() * 100);
        progress_bar += (progress - prev_progress);
        prev_progress = progress;
        pg_usleep(10000);
    }
    progress_bar += (100 - progress_bar.count());
    promise.get_future().get();
}

template<typename F>
inline void run_and_print_progress(F&& progress_callback, uint64_t total, std::string_view message)
{
    elog(INFO, "%s", message.data());
    progress_display progress_bar{total};
    for (auto i = 0u; i < total; ++i) {
        ++progress_bar;
        progress_callback(i);
    }
}

} /// pg::utils namespace
