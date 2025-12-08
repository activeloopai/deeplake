#pragma once

#include <cstdint>

namespace base {

struct cpu_features
{
    bool has_simd_128 = false;  // SSE2/NEON support
    bool has_simd_256 = false;  // AVX2 support
    bool has_simd_512 = false;  // AVX-512 support
    bool has_popcnt = false;    // Population count instruction
    bool has_fast_bitops = false; // Fast bit operations (CTZ, CLZ)
    bool has_bmi1 = false;      // Bit Manipulation Instructions 1
    bool has_bmi2 = false;      // Bit Manipulation Instructions 2
    bool has_lzcnt = false;     // Leading Zero Count
    bool has_tzcnt = false;     // Trailing Zero Count
};

/**
 * @brief Get CPU features for the current platform
 * @return cpu_features structure with detected capabilities
 */
const cpu_features& get_cpu_features() noexcept;

} // namespace base 