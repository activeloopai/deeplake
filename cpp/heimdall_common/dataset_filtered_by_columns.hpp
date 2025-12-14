#pragma once

#include "chained_dataset_view.hpp"

#include <heimdall/column_view.hpp>

#include <memory>

namespace heimdall_common {

/**
 * @brief Create a dataset view that contains only the columns specified in the list.
 * @param source The dataset view to filter.
 * @param columns The list of column names to include in the new dataset view.
 * @return A new dataset view that contains only the columns specified in the list.
 */
heimdall::dataset_view_ptr create_dataset_filtered_by_columns(heimdall::dataset_view_ptr source,
                                                              std::vector<std::string> columns);

/**
 * @brief Create a dataset view that contains only the columns specified in the list.
 * @param columns The list of column names to include in the new dataset view.
 * @return A new dataset view that contains only the columns specified in the list.
 */
heimdall::dataset_view_ptr create_dataset_with_columns(std::vector<heimdall::column_view_ptr> columns);

} // namespace heimdall_common
