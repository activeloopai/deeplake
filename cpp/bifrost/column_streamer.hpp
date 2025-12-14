#pragma once

/**
 * @file column_streamer.hpp
 * @brief Definition of the column streamer class.
 */

#include "async_prefetcher.hpp"

#include <base/logger.hpp>

#include <heimdall/column_view.hpp>
#include <heimdall/dataset_view.hpp>
#include <heimdall_common/dataset_filtered_by_columns.hpp>

namespace bifrost {

/**
 * @brief Class to stream columns from a dataset.
 */
class column_streamer
{
public:
    column_streamer(heimdall::column_view_ptr column, uint64_t batch_size)
        : prefetcher_(heimdall_common::create_dataset_with_columns({column}), false, batch_size, {})
    {
        prefetcher_.start();
    }

    ~column_streamer() = default;

    nd::array next_batch()
    {
        auto b = prefetcher_.next_batch();
        return b.columns()[0].array();
    }

    bool empty() const noexcept
    {
        return prefetcher_.size() == 0;
    }

private:
    async_prefetcher prefetcher_;
};

} // namespace bifrost
