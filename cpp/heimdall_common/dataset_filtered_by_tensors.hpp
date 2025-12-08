#pragma once

#include "chained_dataset_view.hpp"

#include <heimdall/column_view.hpp>

#include <memory>

namespace heimdall_common {

/**
 * @brief Create a dataset view that contains only the tensors specified in the list.
 * @param source The dataset view to filter.
 * @param tensors The list of tensor names to include in the new dataset view.
 * @return A new dataset view that contains only the tensors specified in the list.
 */
heimdall::dataset_view_ptr create_dataset_filtered_by_tensors(heimdall::dataset_view_ptr source,
                                                              std::vector<std::string> tensors);

/**
 * @brief Create a dataset view that contains only the tensors specified in the list.
 * @param tensors The list of tensor names to include in the new dataset view.
 * @return A new dataset view that contains only the tensors specified in the list.
 */
heimdall::dataset_view_ptr create_dataset_with_tensors(std::vector<heimdall::column_view_ptr> tensors);

} // namespace heimdall_common
