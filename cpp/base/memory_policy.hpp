#pragma once

namespace base {

/**
 * @brief Defines the policy of how to use RAM.
 */
enum class memory_policy : unsigned char
{
    aggressive,
    restrictive
};

}
