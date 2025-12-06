#pragma once

/**
 * @file query_result.hpp
 * @brief Definition of the `query_result` struct.
 */

#include <icm/index_mapping.hpp>
#include <nd/array.hpp>

#include <cstdint>
#include <vector>

namespace query_core {

struct query_result
{
    query_result() = default;

    query_result(icm::index_mapping_t<int64_t> i)
        : indices(std::move(i))
    {
    }

    query_result(nd::array s, icm::index_mapping_t<int64_t> i)
        : scores(std::move(s))
        , indices(std::move(i))
    {
    }

    bool is_empty() const noexcept
    {
        return indices.empty();
    }

    nd::array scores;
    icm::index_mapping_t<int64_t> indices;
};

using query_results = std::vector<query_result>;

} // namespace query_core
