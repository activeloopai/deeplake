#pragma once

#include <heimdall/dataset_view.hpp>
#include <icm/index_mapping.hpp>

namespace heimdall_common {

/**
 * @brief Create a filtered dataset from a source dataset and a set of indices.
 * @param source The source dataset.
 * @param indices The indices to include in the filtered dataset.
 * @return The filtered dataset.
 */
heimdall::dataset_view_ptr create_filtered_dataset(heimdall::dataset_view_ptr source,
                                                   const icm::index_mapping_t<int64_t>& indices);
}
