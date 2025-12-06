#pragma once

#include "column_view.hpp"

namespace heimdall {

class column : virtual public column_view
{
public:
    [[nodiscard]] virtual async::promise<void> update_row(int64_t row_offset, const nd::array& new_value) = 0;

    [[nodiscard]] virtual async::promise<void>
    update_rows(int64_t start_row_offset, int64_t end_row_offset, const nd::array& new_value) = 0;

    virtual void set_metadata(const std::string& key, const icm::json& value) = 0;

    virtual void create_index(const deeplake_core::index_type& index_type) = 0;

    virtual void drop_index(const deeplake_core::index_type& index_type) = 0;
};

}
