#pragma once

#include <limits>

namespace async {

constexpr int max_priority = std::numeric_limits<int>::max();
constexpr int extreme_priority = 5000000;
constexpr int high_priority = 1000000;
constexpr int high_medium_priority = 501;
constexpr int low_medium_priority = 500;
constexpr int low_priority = 0;

constexpr int default_priority = max_priority;

}
