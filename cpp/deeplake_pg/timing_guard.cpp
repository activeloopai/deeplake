#include "timing_guard.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace base {

std::ofstream& get_timing_log_file();
std::chrono::high_resolution_clock::time_point get_session_start_time();
// needs to be non-const as it's modified in timing_guard constructor/destructor
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local int timing_indent_level = 0;

std::chrono::high_resolution_clock::time_point get_session_start_time()
{
    thread_local auto session_start = std::chrono::high_resolution_clock::now();
    return session_start;
}

std::ofstream& get_timing_log_file()
{
    thread_local std::ofstream log_file;
    if (!log_file.is_open()) {
        // Initialize session start time for this thread
        get_session_start_time();

        std::ostringstream filename;
        filename << "/home/ubuntu/deeplake_timing_" << std::this_thread::get_id() << ".log";
        log_file.open(filename.str());
    }

    return log_file;
}

timing_guard::timing_guard(const std::string& operation)
    : operation_(operation)
    , start_time_(std::chrono::high_resolution_clock::now())
{
    timing_indent_level++;
}

timing_guard::~timing_guard()
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
    auto start_timestamp_us =
        std::chrono::duration_cast<std::chrono::microseconds>(start_time_ - get_session_start_time()).count();

    constexpr int INDENT_SPACES_PER_LEVEL = 2;
    constexpr int TIMESTAMP_WIDTH = 8;
    std::string indent(static_cast<std::string::size_type>(timing_indent_level * INDENT_SPACES_PER_LEVEL), ' ');

    get_timing_log_file() << "[" << std::setw(TIMESTAMP_WIDTH) << start_timestamp_us << "] " << indent << operation_ << " "
                          << duration_us << '\n';

    timing_indent_level--;
}

} // namespace base
