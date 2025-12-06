#pragma once

#include "group_tensor.hpp"

namespace heimdall_common::impl {

class sequence_tensor : public group_tensor
{
public:
    sequence_tensor(heimdall::column_view_ptr source, const std::vector<int64_t>& sequence_lengths)
        : group_tensor(source, sequence_lengths)
    {
    }

public:
    inline const deeplake_core::type type() const noexcept override
    {
        return deeplake_core::type::sequence(source()->type());
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return true;
    }
};
} // namespace heimdall_common::impl
