#pragma once

#include <heimdall/dataset_view.hpp>

#include "spatial_tensor.hpp"
#include "sequence_tensor.hpp"

#include <memory>

namespace heimdall_common {
/**
 * @brief Creates a group dataset from the source dataset.
 * The group dataset will contain tensors with the same tensor name, dtype, and htype of samples
 * as the source dataset, but the samples will be merged into either sequence or spatial group of the given lengths. That will add an
 * additional dimension to the samples.
 * @note Currently sequence and spatial tensors are supported.
 * @param source The source dataset.
 * @param lengths The lengths of the groups.
 * @return dataset_view_ptr The sequence dataset.
 */
template <typename T>
requires(std::is_same_v<T, spatial_tensor> || std::is_same_v<T, sequence_tensor>)
heimdall::dataset_view_ptr create_group_dataset(heimdall::dataset_view_ptr source, const std::vector<int64_t>& lengths);

/**
 * @brief Given the source group dataset, creates a dataset with the same tensors, but the groups are unrolled.
 * @param source The source group dataset.
 * @return dataset_view_ptr The unrolled dataset.
 */
heimdall::dataset_view_ptr ungroup_dataset(heimdall::dataset_view_ptr source);

/**
 * @brief Given the source group dataset, creates a dataset with the same tensors, but the sequences are split with the given ranges.
 * @param source The source group dataset.
 * @param ranges The ranges of the splits.
 * @return dataset_view_ptr The split dataset.
 */
heimdall::dataset_view_ptr ungroup_dataset_with_split_ranges(heimdall::dataset_view_ptr source, int64_t num_ranges);

/**
 * @brief Given the source group dataset, creates a dataset with the same tensors, but the sequences are split with
 * the given lengths.
 * @param source The source group dataset.
 * @param lengths The lengths of the splits.
 * @return dataset_view_ptr The split dataset.
 */
heimdall::dataset_view_ptr ungroup_dataset_with_lengths(heimdall::dataset_view_ptr source,
                                                        const std::vector<int64_t>& lengths);
} // namespace heimdall_common
