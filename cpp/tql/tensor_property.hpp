#pragma once

#include <cstdint>

namespace tql {

enum class tensor_property : uint8_t
{
    data,
    shape,
    sample_info,
};

} // namespace tql
