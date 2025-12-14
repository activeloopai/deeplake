#pragma once

#include <heimdall/dataset_view.hpp>

#include <memory>

namespace heimdall_common {

/**
 * @brief Creates a resized dataset from the source dataset.
 * The resized dataset will contain columns with the same column name, dtype, htype and dimensionality of samples
 * as the source dataset, but the samples count will be equal to the size parameter.
 *
 * @param source The source dataset.
 * @param size The size of the resized dataset.
 * @return dataset_view_ptr The resized dataset.
 */
heimdall::dataset_view_ptr create_resized_dataset(heimdall::dataset_view_ptr source, int64_t size);

/**
 * @brief Creates a resized dataset which will have the maximum samples count from the source dataset columns.
 */
heimdall::dataset_view_ptr create_max_view(heimdall::dataset_view_ptr source);

/**
 * @brief Creates a resized dataset which will have the minimum samples count from the source dataset columns.
 */
heimdall::dataset_view_ptr create_min_view(heimdall::dataset_view_ptr source);

}
