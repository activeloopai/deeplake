#pragma once

#include <heimdall/column_view.hpp>

#include "impl/spatial_tensor.hpp"

#include <vector>

namespace heimdall_common {

using impl::spatial_tensor;

/**
 * @brief Creates a spatial tensor from the source tensor.
 * @param s The source tensor.
 * @param lengths The lengths of the spatial groups.
 * @return column_view_ptr The sequence tensor.
 */
heimdall::column_view_ptr create_spatial_tensor(heimdall::column_view& s, const std::vector<int64_t>& lengths);

}
