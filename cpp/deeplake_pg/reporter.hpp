#pragma once

#include <chrono>
#include <csignal>
#include <string>

namespace pg {

extern bool print_runtime_stats;

inline static void signal_handler(int signal)
{
    _exit(signal);
}

inline static void install_signal_handlers()
{
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_SIGINFO;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}

template<typename PERIOD = std::milli>
struct runtime_printer
{
    runtime_printer(std::string_view task_name = "Execution time")
        : task_name_(task_name)
    {
        if (pg::print_runtime_stats) {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~runtime_printer() = default;

private:
    std::string_view task_name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace pg
