#pragma once

#include <heimdall/column_view.hpp>

namespace heimdall_common {

/**
 * @brief Merges two identical columns into one bigger column and returns it.
 * Tensors are considered to be identical if they have the same column name, dtype, htype and dimensionality of samples.
 *
 * @param t1 The first column.
 * @param t2 The second column.
 * @return column_view_ptr Result column.
 * @throw column_mismatch, column_mismatch_sequence_mismatch.
 */
heimdall::column_view_ptr create_merged_tensor(heimdall::column_view_ptr t1, heimdall::column_view_ptr t2);
}
