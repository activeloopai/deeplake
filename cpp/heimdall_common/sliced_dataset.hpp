#pragma once

#include <heimdall/dataset_view.hpp>
#include <icm/indexable.hpp>

#include <memory>

namespace heimdall_common {

/**
 * @brief Creates a sliced dataset from the source dataset.
 * The sliced dataset will contain tensors with the same tensor name, dtype, and htype of samples
 * as the source dataset, but the samples will be sliced with the given slice.
 *
 * @param source The source dataset.
 * @param slice The slice of the samples.
 * @return dataset_view_ptr The sliced dataset.
 */
heimdall::dataset_view_ptr create_sliced_dataset(heimdall::dataset_view_ptr source, const icm::indexable_vector& slice);
}
