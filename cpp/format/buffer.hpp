#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace format {

using buffer_t = std::vector<uint8_t>;
using buffer_view_t = std::span<uint8_t>;

}