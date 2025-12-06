#pragma once

#include "row.hpp"
#include "row_range_view.hpp"

namespace heimdall {

class dataset;

class row_range : public row_range_view
{
public:
    row_range(std::shared_ptr<dataset> view, icm::index_mapping_t<int64_t> row_ids);

    impl::row_based_iterator begin() const
    {
        return impl::row_based_iterator(*mutable_view(), row_ids().begin(), impl::row_constructor);
    }

    impl::row_based_iterator end() const
    {
        return impl::row_based_iterator(*mutable_view(), row_ids().end(), impl::row_constructor);
    }

    [[nodiscard]] std::shared_ptr<dataset> mutable_view() const;

    [[nodiscard]] async::promise<void> set_value(const std::string& column_name, const nd::array& value);
};

}
