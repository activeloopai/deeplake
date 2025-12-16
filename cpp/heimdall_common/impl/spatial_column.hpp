#pragma once
#include "group_column.hpp"

namespace heimdall_common::impl {

class spatial_column : public group_column
{
public:
    spatial_column(heimdall::column_view_ptr source, const std::vector<int64_t>& sequence_lengths)
        : group_column(source, sequence_lengths)
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
