#pragma once
#include "group_tensor.hpp"

namespace heimdall_common::impl {

class spatial_tensor : public group_tensor
{
public:
    spatial_tensor(heimdall::column_view_ptr source, const std::vector<int64_t>& sequence_lengths)
        : group_tensor(source, sequence_lengths)
    {
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return false;
    }

    inline const deeplake_core::type type() const noexcept override
    {
        auto t = source()->type();
        auto dt = t.data_type();
        auto dimensions = dt.dimensions();
        if (dimensions != nd::type::unknown_dimensions) {
            dimensions += 1;
        }
        return deeplake_core::type::generic(nd::type::array(dt.get_scalar_type(), dimensions));
    }

};

} // namespace heimdall_common::impl
