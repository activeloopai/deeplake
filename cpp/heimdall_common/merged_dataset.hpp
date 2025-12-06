#pragma once

#include <heimdall/dataset_view.hpp>

namespace heimdall_common {

/**
 * @brief Merges two identical datasets into one bigger dataset and returns it.
 * Datasets are considered to be identical if they contain the same count of tensors and there's an
 * identical mapping between tensors with the same tensor name, dtype, htype and dimensionality of samples.
 * Order of tensors in datasets is not considered during merge. If the datasets have identical but
 * different order of tensors, the order of the first will be preserved.
 *
 * @param ds1 The first dataset.
 * @param ds2 The second dataset.
 * @return dataset_view_ptr Result dataset.
 * @throw datasets_mismatch_uneven_tensors, datasets_mismatch_missing_tensor, tensor_mismatch,
 * tensor_mismatch_sequence_mismatch.
 */
heimdall::dataset_view_ptr create_merged_dataset(heimdall::dataset_view_ptr ds1, heimdall::dataset_view_ptr ds2);
}
