#pragma once

#include <base/backtrace.hpp>
#include <base/memory_info.hpp>
#include <base/system_report.hpp>

#include <chrono>
#include <csignal>
#include <string>

namespace pg {

extern bool print_runtime_stats;

inline static void print_memory_report()
{
    base::memory_info info;

    base::system_report::get_meminfo(info);
    const auto msg = fmt::format("Memory Report:\n"
                                 "RSS: ({:.2f} MB)"
                                 ", VM: ({:.2f} MB)"
                                 ", Peak: ({:.2f} MB)",
                                 info.process_vm_rss / (1024.0 * 1024.0),
                                 info.process_vm_size / (1024.0 * 1024.0),
                                 info.process_peak_mem / (1024.0 * 1024.0));
    elog(INFO, "%s", msg.c_str());
}

inline static void signal_handler(int signal)
{
    elog(NOTICE, "Caught signal %d (%s)\n", signal, strsignal(signal));
    elog(NOTICE, "%s", base::backtrace().c_str());
    print_memory_report();
    fflush(stderr);
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

    ~runtime_printer()
    {
        if (pg::print_runtime_stats) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, PERIOD>(end_time - start_time_).count();
            std::string period_name = "ms";
            if constexpr (std::is_same_v<PERIOD, std::micro>) {
                period_name = "µs";
            } else if constexpr (std::is_same_v<PERIOD, std::nano>) {
                period_name = "ns";
            } else if constexpr (std::is_same_v<PERIOD, std::ratio<1>>) {
                period_name = "s";
            }
            elog(INFO, "%s: %.2f %s", task_name_.data(), duration, period_name.data());
        }
    }

private:
    std::string_view task_name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace pg
