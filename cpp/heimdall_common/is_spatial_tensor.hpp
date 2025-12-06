#pragma once

#include <heimdall/column_view.hpp>

namespace heimdall_common {
bool is_spatial_tensor(const heimdall::column_view& t);
} // namespace heimdall_common
