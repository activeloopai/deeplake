#pragma once

#include <functional>
#include <map>
#include <string>

namespace icm {

template <typename T = std::string>
using string_map = std::map<std::string, T, std::less<>>;

}
