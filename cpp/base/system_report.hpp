#pragma once

#include "cpu_features.hpp"
#include "memory_info.hpp"

#include <cstdint>
#include <cstdlib>

#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER

#include <intrin.h>

static void cpuid(int32_t out[4], int32_t eax, int32_t ecx)
{
    __cpuidex(out, eax, ecx);
}

static __int64 xgetbv(unsigned int x)
{
    return _xgetbv(x);
}

#else

#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>

static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx)
{
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}

static uint64_t xgetbv(unsigned int index)
{
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}

#endif

#if defined(USE_AVX512)

#include <immintrin.h>

#endif

#if defined(__GNUC__)

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

#else

#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))

#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK 0

#endif

namespace base::system_report {

/// Gets the system and process memory information.
bool get_meminfo(memory_info& info) noexcept;

uint64_t total_memory() noexcept;

uint64_t total_available_memory() noexcept;

int cpu_cores() noexcept;

int process_id() noexcept;

bool has_avx();

bool has_avx512();

bool has_avx512_vpopcntdq();

/**
 * @brief Get CPU features for the current platform
 * @return cpu_features structure with detected capabilities
 */
const cpu_features& get_cpu_features() noexcept;

} // namespace base::system_report
