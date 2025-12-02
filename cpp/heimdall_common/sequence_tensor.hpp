#pragma once

#include <heimdall/column_view.hpp>

#include "impl/sequence_tensor.hpp"

#include <vector>

namespace heimdall_common {

using impl::sequence_tensor;

/**
 * @brief Creates a sequence tensor from the source tensor.
 * @param s The source tensor.
 * @param sequence_lengths The lengths of the sequences.
 * @return column_view_ptr The sequence tensor.
 */
heimdall::column_view_ptr create_sequence_tensor(heimdall::column_view& s,
                                                 const std::vector<int64_t>& sequence_lengths);
}
