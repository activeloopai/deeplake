#pragma once

#include "row_view.hpp"

#include <async/promise.hpp>
#include <base/format.hpp>
#include <icm/index_mapping.hpp>
#include <nd/array.hpp>
#include <nd/type.hpp>

namespace heimdall {

class dataset_view;

class row_range_view
{
public:
    row_range_view(std::shared_ptr<dataset_view> view, icm::index_mapping_t<int64_t> row_ids)
        : view_(view)
        , row_ids_(std::move(row_ids))
    {
    }

    [[nodiscard]] const auto& row_ids() const
    {
        return row_ids_;
    }

    auto num_rows() const
    {
        return row_ids_.size();
    }

    impl::row_view_based_iterator cbegin() const
    {
        return impl::row_view_based_iterator(*view_, row_ids_.begin(), impl::row_view_constructor);
    }

    impl::row_view_based_iterator cend() const
    {
        return impl::row_view_based_iterator(*view_, row_ids_.end(), impl::row_view_constructor);
    }

    [[nodiscard]] const auto& view() const
    {
        return view_;
    }

    [[nodiscard]] std::string summary() const;

    [[nodiscard]] async::promise<nd::array> value(const std::string& column_name) const;

    [[nodiscard]] async::promise<nd::array> bytes(const std::string& column_name) const;

private:
    std::shared_ptr<dataset_view> view_;
    icm::index_mapping_t<int64_t> row_ids_;
};

} // namespace heimdall
