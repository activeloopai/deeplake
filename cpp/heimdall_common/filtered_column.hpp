#pragma once

#include <heimdall/column_view.hpp>

#include <icm/index_mapping.hpp>

namespace heimdall_common {

/**
 * @brief Create a filtered column view from a column view and an index mapping.
 * @param s The column view to filter.
 * @param i The index mapping to use for filtering.
 * @return A new column view that is a filtered version of the input column view.
 */
heimdall::column_view_ptr create_filtered_column(heimdall::column_view& s, icm::index_mapping_t<int64_t> i);

} // namespace heimdall_common
