#pragma once

#include <chrono>
#include <string>

namespace base {

class timing_guard
{
public:
    timing_guard(const std::string& operation);
    ~timing_guard();

private:
    std::string operation_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace base
