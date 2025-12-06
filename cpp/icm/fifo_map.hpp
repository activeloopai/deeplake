#pragma once

#include <nlohmann/fifo_map.hpp>

namespace icm {

template <
    class Key,
    class T,
    class Compare = nlohmann::fifo_map_compare<Key>,
    class Allocator = std::allocator<std::pair<const Key, T>>
    >
using fifo_map = nlohmann::fifo_map<Key, T, Compare, Allocator>;

} // namespace icm
