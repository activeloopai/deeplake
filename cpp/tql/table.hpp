#pragma once

#include <heimdall/column_view.hpp>
#include <heimdall/dataset_view.hpp>
#include <icm/index_mapping.hpp>
#include <icm/string_map.hpp>

namespace tql {

class table
{
public:
    table() = default;

    explicit table(heimdall::dataset_view_ptr ds)
        : sub_datasets_{{{}, std::move(ds)}}
    {
    }

    table(heimdall::dataset_view_ptr ds, std::string alias)
        : sub_datasets_{{std::move(alias), std::move(ds)}}
    {
    }

    explicit table(icm::string_map<heimdall::dataset_view_ptr>&& sub_datasets)
        : sub_datasets_(std::move(sub_datasets))
    {
        ASSERT(!sub_datasets_.empty());
    }

    table(table left, table right, std::string left_column_name, std::string right_column_name);

    heimdall::column_view& find_tensor(const std::string& ds_alias, const std::string& tensor_name);
    int64_t max_size() const
    {
        int64_t r = 0L;
        for (const auto& [_, ds] : sub_datasets_) {
            auto s = heimdall::max_size(*ds);
            if (s > r) {
                r = s;
            }
        }
        return r;
    }

    int64_t min_size() const
    {
        auto r = std::numeric_limits<int64_t>::max();
        for (const auto& [_, ds] : sub_datasets_) {
            auto s = heimdall::min_size(*ds);
            if (s < r) {
                r = s;
            }
        }
        return r;
    }

    table filtered(icm::index_mapping_t<int64_t> index) const;

    table transformed(std::function<heimdall::dataset_view_ptr(heimdall::dataset_view_ptr)> f) const;

    const icm::string_map<heimdall::dataset_view_ptr>& sub_datasets() const noexcept
    {
        return sub_datasets_;
    }

    icm::string_map<heimdall::dataset_view_ptr>& sub_datasets() noexcept
    {
        return sub_datasets_;
    }

private:
    icm::string_map<heimdall::dataset_view_ptr> sub_datasets_;
    std::string left_column_name_;
    std::string right_column_name_;
};

} // namespace tql
