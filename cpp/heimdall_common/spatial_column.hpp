#pragma once

#include <heimdall/column_view.hpp>

#include "impl/spatial_column.hpp"

#include <vector>

namespace heimdall_common {

using impl::spatial_column;

/**
 * @brief Creates a spatial column from the source column.
 * @param s The source column.
 * @param lengths The lengths of the spatial groups.
 * @return column_view_ptr The sequence column.
 */
heimdall::column_view_ptr create_spatial_column(heimdall::column_view& s, const std::vector<int64_t>& lengths);

}
