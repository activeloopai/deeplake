#pragma once

#include <heimdall/column_view.hpp>

#include "impl/sequence_column.hpp"

#include <vector>

namespace heimdall_common {

using impl::sequence_column;

/**
 * @brief Creates a sequence column from the source column.
 * @param s The source column.
 * @param sequence_lengths The lengths of the sequences.
 * @return column_view_ptr The sequence column.
 */
heimdall::column_view_ptr create_sequence_column(heimdall::column_view& s,
                                                 const std::vector<int64_t>& sequence_lengths);
}
