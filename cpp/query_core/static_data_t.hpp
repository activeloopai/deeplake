#pragma once

#include <nd/array.hpp>

#include <map>

namespace query_core {

struct data_reference
{
    data_reference(std::string ds_alias, std::string column_name, int64_t i)
        : ds_alias(std::move(ds_alias))
        , column_name(std::move(column_name))
        , sample_index(i)
    {
    }

    friend std::strong_ordering operator<=>(const data_reference& lhs, const data_reference& rhs) noexcept = default;

    std::string ds_alias;
    std::string column_name;
    int64_t sample_index;
};

using static_data_t = std::map<data_reference, nd::array>;

} // namespace query_core
