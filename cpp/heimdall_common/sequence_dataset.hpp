#pragma once

#include "group_dataset.hpp"

#include <memory>

namespace heimdall_common {
/**
 * @brief Creates a sequence dataset from the source dataset.
 * The sequence dataset will contain tensors with the same tensor name, dtype, and htype of samples
 * as the source dataset, but the samples will be merged into sequences of the given lengths. That will add an
 * additional dimension to the samples.
 *
 * @param source The source dataset.
 * @param lengths The lengths of the sequences.
 * @return dataset_view_ptr The sequence dataset.
 */
heimdall::dataset_view_ptr create_sequence_dataset(heimdall::dataset_view_ptr source,
                                                   const std::vector<int64_t>& lengths);
}
